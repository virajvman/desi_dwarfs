"""server.py -- local Flask backend for the recon_vi_scarlet GUI.

Serves one object at a time from the HDF5 bundle (read-only) to the static
JS/canvas frontend, and autosaves a replayable decision CSV. All image
compositing happens client-side; only navigation and decision-save hit the
backend. Post-VI magnitudes are computed SERVER-side at save time (pure sums
of the model-frame component fluxes stored in the bundle) and baked into the
CSV row, so the CSV is analyzable without re-opening any bundle.

Run locally::

    python -m recon_vi_scarlet.server --bundle bundle.h5 --out decisions.csv \
        [--inspector NAME] [--port 8001]

then open http://127.0.0.1:8001/

Endpoints
---------
GET  /                      -> static/index.html
GET  /api/objects           -> index + per-object decision status (sidebar)
GET  /api/object/<i>        -> JSON metadata for object i (components, attrs,
                               array layout, existing decision)
GET  /api/object/<i>/arrays -> one binary blob: input cutout + galaxy cube +
                               not-galaxy cube + residual cube + component
                               patches (layout described by the metadata)
POST /api/decision          -> upsert one decision row (atomic CSV rewrite)
GET  /api/resume            -> {first_undecided, last_decided}
"""

import os
import csv
import math
import argparse
import threading
import datetime

import numpy as np
import h5py
from flask import Flask, Response, request, jsonify, send_from_directory


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

CSV_FIELDS = (
    "TARGETID", "BRICKNAME", "removed_comp_ids", "added_comp_ids",
    "n_components_changed", "lsb_in_galaxy", "bad_fit", "verdict", "inspected",
    "mag_g", "mag_r", "mag_z", "comment", "timestamp", "inspector",
)
VALID_VERDICTS = ("accept", "unsure", "remove", "")


# ----------------------------------------------------------------------
# Decision store (CSV; whole-table atomic rewrite, keyed by TARGETID)
# ----------------------------------------------------------------------

class DecisionStore:
    def __init__(self, path, inspector=""):
        self.path = path
        self.inspector = inspector
        self.lock = threading.Lock()
        self.rows = {}  # targetid(int) -> dict
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    tgid = int(row["TARGETID"])
                except (KeyError, ValueError):
                    continue
                for flag in ("inspected", "lsb_in_galaxy", "bad_fit"):
                    row[flag] = str(row.get(flag, "")).strip().lower() in ("1", "true", "yes")
                try:
                    row["n_components_changed"] = int(row.get("n_components_changed", 0) or 0)
                except ValueError:
                    row["n_components_changed"] = 0
                self.rows[tgid] = row

    def get(self, tgid):
        return self.rows.get(int(tgid))

    def _write_atomic(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for tgid in sorted(self.rows):
                r = self.rows[tgid]
                w.writerow({
                    "TARGETID": tgid,
                    "BRICKNAME": r.get("BRICKNAME", ""),
                    "removed_comp_ids": r.get("removed_comp_ids", ""),
                    "added_comp_ids": r.get("added_comp_ids", ""),
                    "n_components_changed": int(r.get("n_components_changed", 0) or 0),
                    "lsb_in_galaxy": int(bool(r.get("lsb_in_galaxy", False))),
                    "bad_fit": int(bool(r.get("bad_fit", False))),
                    "verdict": r.get("verdict", ""),
                    "inspected": int(bool(r.get("inspected", False))),
                    "mag_g": r.get("mag_g", ""),
                    "mag_r": r.get("mag_r", ""),
                    "mag_z": r.get("mag_z", ""),
                    "comment": r.get("comment", ""),
                    "timestamp": r.get("timestamp", ""),
                    "inspector": r.get("inspector", self.inspector),
                })
        os.replace(tmp, self.path)

    def upsert(self, payload, derived):
        """payload = client JSON; derived = server-computed
        {lsb_in_galaxy, mag_g, mag_r, mag_z}."""
        tgid = int(payload["TARGETID"])
        removed = [str(x) for x in payload.get("removed_comp_ids", []) if str(x)]
        added = [str(x) for x in payload.get("added_comp_ids", []) if str(x)]
        verdict = str(payload.get("verdict", "") or "")
        if verdict not in VALID_VERDICTS:
            raise ValueError("bad verdict {!r}".format(verdict))
        with self.lock:
            row = {
                "TARGETID": tgid,
                "BRICKNAME": str(payload.get("BRICKNAME", "")),
                "removed_comp_ids": ";".join(removed),
                "added_comp_ids": ";".join(added),
                "n_components_changed": len(removed) + len(added),
                "lsb_in_galaxy": bool(derived.get("lsb_in_galaxy", False)),
                "bad_fit": bool(payload.get("bad_fit", False)),
                "verdict": verdict,
                "inspected": bool(verdict),
                "mag_g": derived.get("mag_g", ""),
                "mag_r": derived.get("mag_r", ""),
                "mag_z": derived.get("mag_z", ""),
                "comment": str(payload.get("comment", "") or ""),
                "timestamp": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
                "inspector": self.inspector,
            }
            self.rows[tgid] = row
            self._write_atomic()
        return row


# ----------------------------------------------------------------------
# Bundle reader (read-only; one open handle guarded by a lock)
# ----------------------------------------------------------------------

class Bundle:
    CUBE_NAMES = ("input_cutout", "science_cube", "nondwarf_cube", "residual_cube")

    def __init__(self, path):
        self.lock = threading.Lock()
        self.f = h5py.File(path, "r")
        idx = self.f["index"][:]
        self.index = []
        for r in idx:
            self.index.append({
                "targetid": int(r["targetid"]),
                "brickname": _to_str(r["brickname"]),
                "box_size": int(r["box_size"]),
                "n_components": int(r["n_components"]),
                "n_members": int(r["n_members"]),
            })
        self.i_by_targetid = {r["targetid"]: i for i, r in enumerate(self.index)}

    def __len__(self):
        return len(self.index)

    def group(self, i):
        return self.f[str(self.index[i]["targetid"])]

    @staticmethod
    def _comp_sort_key(cid):
        return int(cid)

    def components_meta(self, g):
        comps = []
        if "components" not in g:
            return comps
        cg = g["components"]
        for key in sorted(cg.keys(), key=self._comp_sort_key):
            d = cg[key]
            y0, x0 = [int(v) for v in d.attrs["bbox"]]
            comps.append({
                "comp_id": int(d.attrs["comp_id"]),
                "bbox": [y0, x0], "h": int(d.shape[1]), "w": int(d.shape[2]),
                "initial_membership": bool(d.attrs["initial_membership"]),
                "type": _to_str(d.attrs["type"]),
                "is_star": bool(d.attrs["is_star"]),
                "xpix": _f(d.attrs["xpix"]), "ypix": _f(d.attrs["ypix"]),
                "ra": _f(d.attrs["ra"]), "dec": _f(d.attrs["dec"]),
                "flux_g": _f(d.attrs["flux_g"]),
                "flux_r": _f(d.attrs["flux_r"]),
                "flux_z": _f(d.attrs["flux_z"]),
                "gr": _f(d.attrs["gr"]), "rz": _f(d.attrs["rz"]),
                "gmm_prob": _f(d.attrs["gmm_prob"]),
            })
        return comps

    def metadata(self, i):
        with self.lock:
            g = self.group(i)
            base = g["science_cube"]
            cube_dtype = "float16" if base.dtype == np.float16 else "float32"
            return {
                "index": i,
                "targetid": int(self.index[i]["targetid"]),
                "brickname": _to_str(g.attrs["BRICKNAME"]),
                "box_size": int(g.attrs["box_size"]),
                "cube_dtype": cube_dtype,
                "gal_xpix": _f(g.attrs["gal_xpix"]),
                "gal_ypix": _f(g.attrs["gal_ypix"]),
                "gal_ra": _f(g.attrs["gal_ra"]),
                "gal_dec": _f(g.attrs["gal_dec"]),
                "gr_cut": _f(g.attrs.get("gr_cut", float("nan"))),
                "rz_cut": _f(g.attrs.get("rz_cut", float("nan"))),
                "n_components": int(g.attrs["n_components"]),
                "n_members": int(g.attrs["n_members"]),
                "n_stars": int(g.attrs.get("n_stars", 0)),
                "components": self.components_meta(g),
            }

    def arrays(self, i):
        """Concatenated binary blob in the order the metadata expects:
        input cutout, galaxy (science) cube, not-galaxy cube, residual cube
        (all cube_dtype), then each component patch (comp_id-sorted, always
        float32)."""
        with self.lock:
            g = self.group(i)
            cube_dtype = g["science_cube"].dtype
            parts = []
            for name in self.CUBE_NAMES:
                parts.append(np.ascontiguousarray(g[name][:], dtype=cube_dtype)
                             .tobytes())
            if "components" in g:
                cg = g["components"]
                for key in sorted(cg.keys(), key=self._comp_sort_key):
                    parts.append(np.ascontiguousarray(cg[key][:], dtype=np.float32)
                                 .tobytes())
            return b"".join(parts)

    def post_vi_photometry(self, targetid, removed_ids, added_ids):
        """Final member set = initial_membership minus removed plus added.
        Returns {lsb_in_galaxy, mag_g, mag_r, mag_z}; mags are pure sums of the
        model-frame per-component fluxes (NaN fluxes skipped; empty/nonpositive
        sum -> blank)."""
        i = self.i_by_targetid[int(targetid)]
        removed = {int(x) for x in removed_ids}
        added = {int(x) for x in added_ids}
        with self.lock:
            g = self.group(i)
            comps = self.components_meta(g)
        flux = np.zeros(3, dtype=np.float64)
        lsb_in = False
        for c in comps:
            cid = c["comp_id"]
            member = (c["initial_membership"] and cid not in removed) or cid in added
            if not member:
                continue
            if c["type"] == "starlet_lsb":
                lsb_in = True
            for b, name in enumerate(("flux_g", "flux_r", "flux_z")):
                v = c[name]
                if v is not None and math.isfinite(v):
                    flux[b] += v
        out = {"lsb_in_galaxy": lsb_in}
        for b, name in enumerate(("mag_g", "mag_r", "mag_z")):
            out[name] = ("{:.4f}".format(22.5 - 2.5 * math.log10(flux[b]))
                         if flux[b] > 0 else "")
        return out


def _to_str(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def _f(v):
    """JSON-safe float: NaN/Inf -> None (Flask would emit invalid JSON NaN
    tokens and the browser's r.json() would throw)."""
    v = float(v)
    return v if math.isfinite(v) else None


def _targetid_str(v):
    """JSON-safe TARGETID: decimal string preserves full int64 in JS."""
    return str(int(v))


# ----------------------------------------------------------------------
# App factory
# ----------------------------------------------------------------------

def create_app(bundle_path, out_path, inspector=""):
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
    bundle = Bundle(bundle_path)
    store = DecisionStore(out_path, inspector=inspector)
    # TARGETIDs known to the bundle. The decision guard rejects anything else, so
    # a stray/garbage POST can never create a row for an object we aren't serving.
    bundle_targetids = set(int(r["targetid"]) for r in bundle.index)

    @app.route("/")
    def index_page():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/api/objects")
    def api_objects():
        out = []
        for i, row in enumerate(bundle.index):
            dec = store.get(row["targetid"])
            status = ""
            verdict = ""
            edited = False
            bad_fit = False
            if dec is not None:
                verdict = dec.get("verdict", "") or ""
                edited = int(dec.get("n_components_changed", 0) or 0) > 0
                bad_fit = bool(dec.get("bad_fit", False))
                status = "inspected" if dec.get("inspected") else ("edited" if edited else "")
            out.append({
                "index": i,
                "targetid": _targetid_str(row["targetid"]),
                "brickname": row["brickname"],
                "n_components": row["n_components"],
                "n_members": row["n_members"],
                "status": status,
                "verdict": verdict,
                "edited": edited,
                "bad_fit": bad_fit,
            })
        return jsonify({"objects": out, "inspector": inspector})

    @app.route("/api/object/<int:i>")
    def api_object(i):
        if i < 0 or i >= len(bundle):
            return jsonify({"error": "index out of range"}), 404
        meta = bundle.metadata(i)
        tgid = int(bundle.index[i]["targetid"])
        meta["targetid"] = _targetid_str(tgid)
        dec = store.get(tgid)
        if dec is not None:
            meta["decision"] = {
                "removed_comp_ids": _split(dec.get("removed_comp_ids", "")),
                "added_comp_ids": _split(dec.get("added_comp_ids", "")),
                "verdict": dec.get("verdict", "") or "",
                "bad_fit": bool(dec.get("bad_fit", False)),
                "comment": dec.get("comment", "") or "",
                "inspected": bool(dec.get("inspected", False)),
            }
        else:
            meta["decision"] = None
        return jsonify(meta)

    @app.route("/api/object/<int:i>/arrays")
    def api_object_arrays(i):
        if i < 0 or i >= len(bundle):
            return Response("index out of range", status=404)
        blob = bundle.arrays(i)
        return Response(blob, mimetype="application/octet-stream")

    @app.route("/api/decision", methods=["POST"])
    def api_decision():
        payload = request.get_json(force=True)
        try:
            tgid = int(payload["TARGETID"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "missing or invalid TARGETID"}), 400
        if tgid not in bundle_targetids:
            return jsonify({"error": "TARGETID {} not in bundle".format(tgid)}), 400
        try:
            derived = bundle.post_vi_photometry(
                tgid,
                payload.get("removed_comp_ids", []),
                payload.get("added_comp_ids", []))
            row = store.upsert(payload, derived)
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "row": {
            "TARGETID": _targetid_str(row["TARGETID"]), "verdict": row["verdict"],
            "inspected": row["inspected"],
            "n_components_changed": row["n_components_changed"],
            "bad_fit": row["bad_fit"],
        }})

    @app.route("/api/resume")
    def api_resume():
        first_undecided = None
        last_decided = None
        for i, row in enumerate(bundle.index):
            dec = store.get(row["targetid"])
            decided = dec is not None and dec.get("inspected")
            if decided:
                last_decided = i
            elif first_undecided is None:
                first_undecided = i
        if first_undecided is None:
            first_undecided = 0
        return jsonify({"first_undecided": first_undecided,
                        "last_decided": last_decided,
                        "n_objects": len(bundle)})

    # Local dev tool: never let the browser cache the JS/CSS/HTML, so edits to
    # static/ show up on a plain reload (no hard-refresh needed).
    @app.after_request
    def _no_cache(resp):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    return app


def _split(s):
    return [x for x in str(s or "").split(";") if x]


def argument_parser():
    p = argparse.ArgumentParser(
        description="Local Flask backend for the recon_vi_scarlet GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bundle", required=True, help="HDF5 bundle from build_bundle.py")
    p.add_argument("--out", required=True, help="decision CSV (created/appended)")
    p.add_argument("--inspector", default="", help="recorded in each CSV row")
    p.add_argument("--host", default="127.0.0.1")
    # 8001 by default so recon_vi (8000) and this tool can run side by side
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--debug", action="store_true")
    return p


def main(argv=None):
    args = argument_parser().parse_args(argv)
    app = create_app(args.bundle, args.out, inspector=args.inspector)
    n = len(app.view_functions)  # touch app to ensure it built
    print("recon_vi_scarlet server: bundle={} out={}".format(args.bundle, args.out))
    print("  open http://{}:{}/".format(args.host, args.port))
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return n


if __name__ == "__main__":
    main()
