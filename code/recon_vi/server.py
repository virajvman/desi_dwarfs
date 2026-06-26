"""server.py -- local Flask backend for the recon_vi GUI.

Serves one object at a time from the HDF5 bundle (read-only) to the static
JS/canvas frontend, and autosaves a replayable decision CSV. All image
compositing happens client-side; only navigation and decision-save hit the
backend.

Run locally::

    python -m recon_vi.server --bundle bundle.h5 --out decisions.csv \
        [--inspector NAME] [--port 8000]

then open http://127.0.0.1:8000/

Endpoints
---------
GET  /                      -> static/index.html
GET  /api/objects           -> index + per-object decision status (sidebar)
GET  /api/object/<i>        -> JSON metadata for object i (sources, attrs,
                               array layout, existing decision)
GET  /api/object/<i>/arrays -> one binary blob: base cube + input cutout +
                               source flux patches (layout described by the
                               metadata endpoint)
POST /api/decision          -> upsert one decision row (atomic CSV rewrite)
GET  /api/resume            -> {first_undecided, last_decided}
"""

import os
import csv
import math
import json
import argparse
import threading
import datetime

import numpy as np
import h5py
from flask import Flask, Response, request, jsonify, send_from_directory


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

CSV_FIELDS = (
    "TARGETID", "BRICKNAME", "removed_objids", "added_objids",
    "n_sources_changed", "verdict", "inspected", "comment",
    "toggle_disabled", "infill_masked", "timestamp", "inspector",
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
                row["inspected"] = str(row.get("inspected", "")).strip().lower() in ("1", "true", "yes")
                row["toggle_disabled"] = str(row.get("toggle_disabled", "")).strip().lower() in ("1", "true", "yes")
                # default ON: a row predating this column means "infill" (absent -> True)
                _inf = str(row.get("infill_masked", "")).strip().lower()
                row["infill_masked"] = True if _inf == "" else _inf in ("1", "true", "yes")
                try:
                    row["n_sources_changed"] = int(row.get("n_sources_changed", 0) or 0)
                except ValueError:
                    row["n_sources_changed"] = 0
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
                    "removed_objids": r.get("removed_objids", ""),
                    "added_objids": r.get("added_objids", ""),
                    "n_sources_changed": int(r.get("n_sources_changed", 0) or 0),
                    "verdict": r.get("verdict", ""),
                    "inspected": int(bool(r.get("inspected", False))),
                    "comment": r.get("comment", ""),
                    "toggle_disabled": int(bool(r.get("toggle_disabled", False))),
                    "infill_masked": int(bool(r.get("infill_masked", True))),
                    "timestamp": r.get("timestamp", ""),
                    "inspector": r.get("inspector", self.inspector),
                })
        os.replace(tmp, self.path)

    def upsert(self, payload):
        tgid = int(payload["TARGETID"])
        removed = [str(x) for x in payload.get("removed_objids", []) if str(x)]
        added = [str(x) for x in payload.get("added_objids", []) if str(x)]
        verdict = str(payload.get("verdict", "") or "")
        if verdict not in VALID_VERDICTS:
            raise ValueError("bad verdict {!r}".format(verdict))
        with self.lock:
            row = {
                "TARGETID": tgid,
                "BRICKNAME": str(payload.get("BRICKNAME", "")),
                "removed_objids": ";".join(removed),
                "added_objids": ";".join(added),
                "n_sources_changed": len(removed) + len(added),
                "verdict": verdict,
                "inspected": bool(verdict),
                "comment": str(payload.get("comment", "") or ""),
                "toggle_disabled": bool(payload.get("toggle_disabled", False)),
                "infill_masked": bool(payload.get("infill_masked", True)),
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
    def __init__(self, path):
        self.lock = threading.Lock()
        self.f = h5py.File(path, "r")
        idx = self.f["index"][:]
        self.index = []
        for r in idx:
            self.index.append({
                "targetid": int(r["targetid"]),
                "brickname": _to_str(r["brickname"]),
                "recon_variant": _to_str(r["recon_variant"]),
                "toggle_disabled": bool(r["toggle_disabled"]),
                "n_sources": int(r["n_sources"]),
            })

    def __len__(self):
        return len(self.index)

    def group(self, i):
        return self.f[str(self.index[i]["targetid"])]

    def metadata(self, i):
        with self.lock:
            g = self.group(i)
            base_name = "recon_cube" if bool(g.attrs["toggle_disabled"]) else "science_cube"
            base = g[base_name]
            cube_dtype = "float16" if base.dtype == np.float16 else "float32"
            S = int(g.attrs["box_size"])
            sources = []
            if "sources" in g:
                for key in g["sources"]:
                    d = g["sources"][key]
                    y0, x0 = [int(v) for v in d.attrs["bbox"]]
                    h, w = int(d.shape[1]), int(d.shape[2])
                    sources.append({
                        "objid": int(d.attrs["source_objid_new"]),
                        "bbox": [y0, x0], "h": h, "w": w,
                        "initial_membership": bool(d.attrs["initial_membership"]),
                        "type": _to_str(d.attrs["type"]),
                        "xpix": _f(d.attrs["xpix"]), "ypix": _f(d.attrs["ypix"]),
                        "separation": _f(d.attrs["separation"]),
                        "mag_g": _f(d.attrs["mag_g"]),
                        "mag_r": _f(d.attrs["mag_r"]),
                        "mag_z": _f(d.attrs["mag_z"]),
                        "sersic": _f(d.attrs["sersic"]),
                        "shape_r": _f(d.attrs["shape_r"]),
                        "shape_e1": _f(d.attrs["shape_e1"]),
                        "shape_e2": _f(d.attrs["shape_e2"]),
                    })
            sources.sort(key=lambda s: s["objid"])
            nr = g.attrs.get("noise_rms_grz")
            if nr is None:
                noise = [None, None, None]   # old bundle w/o the attr -> infill off
            else:
                nr = np.asarray(nr, dtype=np.float64).ravel()
                # JSON has no NaN -> emit null for unusable bands (JS reads infill off)
                noise = [(float(nr[k]) if (k < nr.size and np.isfinite(nr[k])) else None)
                         for k in range(3)]
            return {
                "index": i,
                "targetid": int(self.index[i]["targetid"]),
                "brickname": _to_str(g.attrs["BRICKNAME"]),
                "recon_variant": _to_str(g.attrs["recon_variant"]),
                "toggle_disabled": bool(g.attrs["toggle_disabled"]),
                "box_size": S,
                "cube_dtype": cube_dtype,
                "gal_xpix": _f(g.attrs["gal_xpix"]),
                "gal_ypix": _f(g.attrs["gal_ypix"]),
                "gal_ra": _f(g.attrs["gal_ra"]),
                "gal_dec": _f(g.attrs["gal_dec"]),
                "target_objid": int(g.attrs.get("target_objid", -1)),
                "noise_rms_grz": noise,
                "sources": sources,
            }

    def arrays(self, i):
        """Concatenated binary blob in the order the metadata expects:
        base cube, input cutout, then each source patch (objid-sorted), patches
        always float32. Returns (bytes, cube_dtype_str)."""
        with self.lock:
            g = self.group(i)
            disabled = bool(g.attrs["toggle_disabled"])
            base = g["recon_cube"][:] if disabled else g["science_cube"][:]
            cutout = g["input_cutout"][:]
            cube_dtype = base.dtype  # float16 or float32, matched by both cubes
            parts = [np.ascontiguousarray(base, dtype=cube_dtype).tobytes(),
                     np.ascontiguousarray(cutout, dtype=cube_dtype).tobytes()]
            if "sources" in g:
                for key in sorted(g["sources"], key=lambda k: int(k)):
                    patch = g["sources"][key][:]
                    parts.append(np.ascontiguousarray(patch, dtype=np.float32).tobytes())
            return b"".join(parts)


def _to_str(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def _f(v):
    """JSON-safe float: NaN/Inf -> None. Flask's jsonify serializes non-finite
    floats as the literal tokens ``NaN``/``Infinity``, which are invalid JSON;
    the browser's r.json() then throws and the whole object load fails (e.g. a
    source with mag_z=NaN wedged objects 146/148). The frontend already renders
    null as "—" and guards overlays with isFinite, so null is safe here."""
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
    # a stray/garbage POST can never create a row for an object we aren't serving
    # (the write is already TARGETID-keyed; this just makes a bad key loud).
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
            if dec is not None:
                verdict = dec.get("verdict", "") or ""
                edited = int(dec.get("n_sources_changed", 0) or 0) > 0
                status = "inspected" if dec.get("inspected") else ("edited" if edited else "")
            out.append({
                "index": i,
                "targetid": _targetid_str(row["targetid"]),
                "brickname": row["brickname"],
                "toggle_disabled": row["toggle_disabled"],
                "n_sources": row["n_sources"],
                "status": status,
                "verdict": verdict,
                "edited": edited,
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
                "removed_objids": _split(dec.get("removed_objids", "")),
                "added_objids": _split(dec.get("added_objids", "")),
                "verdict": dec.get("verdict", "") or "",
                "comment": dec.get("comment", "") or "",
                "inspected": bool(dec.get("inspected", False)),
                "infill_masked": bool(dec.get("infill_masked", True)),
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
            row = store.upsert(payload)
        except (KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"ok": True, "row": {
            "TARGETID": _targetid_str(row["TARGETID"]), "verdict": row["verdict"],
            "inspected": row["inspected"],
            "n_sources_changed": row["n_sources_changed"],
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
        description="Local Flask backend for the recon_vi GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--bundle", required=True, help="HDF5 bundle from build_bundle.py")
    p.add_argument("--out", required=True, help="decision CSV (created/appended)")
    p.add_argument("--inspector", default="", help="recorded in each CSV row")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--debug", action="store_true")
    return p


def main(argv=None):
    args = argument_parser().parse_args(argv)
    app = create_app(args.bundle, args.out, inspector=args.inspector)
    n = len(app.view_functions)  # touch app to ensure it built
    print("recon_vi server: bundle={} out={}".format(args.bundle, args.out))
    print("  open http://{}:{}/".format(args.host, args.port))
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return n


if __name__ == "__main__":
    main()
