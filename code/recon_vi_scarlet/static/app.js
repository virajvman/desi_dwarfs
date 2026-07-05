// app.js -- recon_vi_scarlet frontend: navigation, component overlay,
// flux-space compositing, verdicts, autosave. Adapted from recon_vi's frontend;
// the core semantic change: every component lives in exactly one of TWO
// complementary model panels (galaxy | not-galaxy) and a toggle MOVES it
// between them (galaxy ± patch, not-galaxy ∓ patch). All compositing is
// client-side; only navigation and decision-save hit the backend.

(function () {
  "use strict";

  var FOV = 350;               // fiducial viewport size in array pixels
                               // (350 = the locked production fit box, so the
                               // whole cutout is visible on open; was 200)
  var MIN_HIT_PX = 8;          // min click tolerance (screen px)
  var VIEWER_TID_BASE = "https://www.legacysurvey.org/viewer/desi-spectrum/dr1/targetid";
  // Membership/provenance palette (same grammar as recon_vi): COLOR = state,
  // LINE STYLE = in galaxy (solid) / in not-galaxy (dashed), BADGE = your edits.
  var COLORS = {
    inGalaxy: "#2ee36a",   // component currently in the galaxy panel
    out: "#ff5d6c",        // component currently in the not-galaxy panel
    edited: "#ffb02e",     // changed by you this session (undo candidate)
    target: "#22e0e0",     // DESI target position crosshair (not a component)
    selected: "#ffe14d",   // pending selection
    hover: "#9aa0ff",      // hovered component's bbox outline
  };

  // ---- DOM ---------------------------------------------------------------
  var $ = function (id) { return document.getElementById(id); };
  var panels = {
    input: { canvas: null, off: null, label: "input" },
    galaxy: { canvas: null, off: null, label: "galaxy" },
    nondwarf: { canvas: null, off: null, label: "nondwarf" },
    residual: { canvas: null, off: null, label: "residual" },
    datamgal: { canvas: null, off: null, label: "datamgal" },
  };
  var els = {};

  // ---- State -------------------------------------------------------------
  var objects = [];            // /api/objects list
  var inspectorName = "";
  var cur = null;              // current object working state
  var view = { scale: 1, ox0: 0, oy0: 0 };  // shared offscreen viewport
  var hideOverlay = false;     // hold O for a clean, marker-free look (momentary)
  var showMarkers = true;      // persistent toggle: markers on ALL panels (m)
  var saveTimer = null;
  var loadToken = 0;           // guards against out-of-order/stale object loads
  var filterMode = false;      // accepted-only review pass: view+nav overlay only
  var filteredIndices = [];    // snapshot of raw indices with verdict==="accept"

  // ---- Init --------------------------------------------------------------
  function init() {
    panels.input.canvas = $("canvas-input");
    panels.galaxy.canvas = $("canvas-galaxy");
    panels.nondwarf.canvas = $("canvas-nondwarf");
    panels.residual.canvas = $("canvas-residual");
    panels.datamgal.canvas = $("canvas-datamgal");
    els = {
      markersBanner: $("markers-banner"),
      sidebar: $("sidebar-list"), btnFilter: $("btn-filter-accept"),
      counter: $("counter"), targetid: $("targetid"), brickname: $("brickname"),
      ncomp: $("ncomp-badge"),
      btnToGalaxy: $("btn-to-galaxy"), btnToNondwarf: $("btn-to-nondwarf"),
      btnUndo: $("btn-undo"), btnReset: $("btn-reset"),
      btnLsb: $("btn-lsb"), btnBadfit: $("btn-badfit"),
      btnMarkers: $("btn-markers"),
      btnAccept: $("btn-accept"), btnUnsure: $("btn-unsure"), btnReject: $("btn-reject"),
      btnPrev: $("btn-prev"), btnNext: $("btn-next"),
      btnResetView: $("btn-reset-view"), fov: $("fov-input"),
      jump: $("jump-input"), jumpGo: $("btn-jump"),
      comment: $("comment"), selinfo: $("sel-info"), status: $("status-line"),
      tooltip: $("tooltip"), verdictBadge: $("verdict-badge"),
    };
    resizeCanvases();
    wireEvents();
    bootstrap();
  }

  // Explicit square sizing for the 5 canvases: computed from the ACTUAL
  // available space (both width AND height) so the 2-row layout never
  // overflows past the controls below. A pure CSS aspect-ratio+flex-shrink
  // approach doesn't work here -- a flex child's default min-height:auto
  // pins it to its own content's intrinsic (aspect-ratio-derived) size, so
  // shrinking the row just overflows instead of producing a smaller square.
  // Sets style.width/height in px (NOT the canvas.width/height backing-store
  // attributes, which stay 400 and are what the click/hover math scales
  // against via getBoundingClientRect() -- see canvasToArray/onPanelClick).
  function resizeCanvases() {
    var panelsSection = document.getElementById("panels");
    var row = document.querySelector(".panel-row");
    var figcap = document.querySelector(".panel figcaption");
    if (!panelsSection || !row) return;
    var GAP = 10;
    var totalW = panelsSection.clientWidth;
    var totalH = panelsSection.clientHeight;
    var colW = (totalW - GAP * 2) / 3;   // 3 columns, 2 inter-column gaps
    var rowH = (totalH - GAP) / 2;       // 2 rows, 1 inter-row gap
    var capH = figcap ? figcap.getBoundingClientRect().height : 20;
    var size = Math.max(80, Math.floor(Math.min(colW, rowH - capH)));
    document.querySelectorAll(".panel canvas").forEach(function (cv) {
      cv.style.width = size + "px";
      cv.style.height = size + "px";
    });
  }

  function bootstrap() {
    fetch("/api/objects").then(function (r) { return r.json(); }).then(function (j) {
      objects = j.objects; inspectorName = j.inspector || "";
      buildSidebar();
      return fetch("/api/resume");
    }).then(function (r) { return r.json(); }).then(function (j) {
      openObject(j.first_undecided || 0);
    }).catch(function (e) { console.error(e); alert("Failed to load bundle: " + e); });
  }

  function tidStr(v) { return String(v); }
  function viewerUrl(tid) { return VIEWER_TID_BASE + tidStr(tid); }

  // ---- Sidebar -----------------------------------------------------------
  function buildSidebar() {
    var html = "";
    for (var i = 0; i < objects.length; i++) {
      var o = objects[i];
      html += '<div class="sb-item" data-i="' + i + '" id="sb-' + i + '">' +
        '<span class="sb-idx">' + i + '</span>' +
        '<span class="sb-tid">' + tidStr(o.targetid) + '</span>' +
        '<span class="sb-mark" id="sb-mark-' + i + '"></span></div>';
    }
    els.sidebar.innerHTML = html;
    els.sidebar.querySelectorAll(".sb-item").forEach(function (el) {
      el.addEventListener("click", function () {
        commitPending(); openObject(parseInt(el.getAttribute("data-i"), 10));
      });
    });
    for (var k = 0; k < objects.length; k++) updateSidebarItem(k);
  }

  function updateSidebarItem(i) {
    var o = objects[i];
    var mark = $("sb-mark-" + i);
    if (!mark) return;
    var t = "";
    if (o.verdict === "accept") t = "✓";
    else if (o.verdict === "unsure") t = "?";
    else if (o.verdict === "remove") t = "✗";
    else if (o.edited) t = "✎";
    if (o.bad_fit) t += "⚑";
    mark.textContent = t;
    mark.className = "sb-mark v-" + (o.verdict || (o.edited ? "edited" : "none"));
  }

  function highlightSidebar(i) {
    els.sidebar.querySelectorAll(".sb-item.active").forEach(function (el) {
      el.classList.remove("active");
    });
    var item = $("sb-" + i);
    if (item) { item.classList.add("active"); item.scrollIntoView({ block: "nearest" }); }
  }

  // ---- Accepted-only filter (view + navigation overlay) ------------------
  // Frontend-only. `filteredIndices` is a SNAPSHOT of the raw indices i whose
  // verdict==="accept", taken when the filter is turned on. Navigation and the
  // sidebar then operate within this subset, while objects[] and the raw
  // index<->TARGETID binding are left completely untouched -- so saves still hit
  // the correct CSV row regardless of what's displayed. Snapshot (not live): an
  // object downgraded mid-pass stays in the set until the filter is toggled
  // off->on, so the list you're stepping through never shifts under you.
  function computeAcceptedSnapshot() {
    var out = [];
    for (var i = 0; i < objects.length; i++) {
      if (objects[i] && objects[i].verdict === "accept") out.push(i);
    }
    return out;
  }

  // Next index to navigate to from raw index i (subset-aware); -1 if none.
  function nextNavIndex(i) {
    if (filterMode) {
      var pos = filteredIndices.indexOf(i);
      return (pos !== -1 && pos + 1 < filteredIndices.length) ? filteredIndices[pos + 1] : -1;
    }
    return i + 1;
  }

  function applyFilterToSidebar() {
    var inSet = {};
    for (var k = 0; k < filteredIndices.length; k++) inSet[filteredIndices[k]] = true;
    for (var i = 0; i < objects.length; i++) {
      var item = $("sb-" + i);
      if (item) item.classList.toggle("hidden", filterMode && !inSet[i]);
    }
  }

  function updateFilterBtn() {
    var b = els.btnFilter;
    if (!b) return;
    b.classList.toggle("on", filterMode);
    b.textContent = filterMode ? "Accepted only ✓" : "Accepted only";
  }

  function toggleAcceptFilter() {
    if (!filterMode) {
      var snap = computeAcceptedSnapshot();
      if (snap.length === 0) { flashTooltip("No accepted objects yet"); return; }
      filterMode = true;
      filteredIndices = snap;
      applyFilterToSidebar();
      updateFilterBtn();
      // If the current object isn't accepted, jump to the first one that is.
      if (!cur || filteredIndices.indexOf(cur.i) === -1) {
        commitPending();
        openObject(filteredIndices[0]);
      } else {
        refreshHeader();   // just refresh the "k / N accepted" counter
      }
    } else {
      // Pure view change: stay on the current object, expand back to the full list.
      filterMode = false;
      filteredIndices = [];
      applyFilterToSidebar();
      updateFilterBtn();
      refreshHeader();
    }
  }

  // ---- Object loading ----------------------------------------------------
  // Fetch with r.ok checking and a small retry: the Flask dev server can drop
  // or truncate a response under load, which used to surface as a generic
  // "Failed to load object" and — because cur never advanced on failure —
  // permanently wedge the Next button at the broken index. Transient reads
  // now just recover.
  function fetchOk(url, asBuffer, tries) {
    tries = tries || 3;
    return fetch(url).then(function (r) {
      if (!r.ok) {
        return r.text().then(function (t) {
          throw new Error("HTTP " + r.status + ": " + (t || "").slice(0, 200));
        });
      }
      return asBuffer ? r.arrayBuffer() : r.json();
    }).catch(function (e) {
      if (tries > 1) {
        return new Promise(function (res) { setTimeout(res, 250); })
          .then(function () { return fetchOk(url, asBuffer, tries - 1); });
      }
      throw e;
    });
  }

  function openObject(i) {
    if (i < 0 || i >= objects.length) return;
    // A pending debounced save belongs to the object we're leaving; every nav
    // path already commits it synchronously first, so kill the timer here so it
    // can never fire against the object we're about to load.
    if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
    var token = ++loadToken;   // only the latest openObject is allowed to commit
    var metaP = fetchOk("/api/object/" + i, false);
    var arrP = fetchOk("/api/object/" + i + "/arrays", true);
    Promise.all([metaP, arrP]).then(function (res) {
      if (token !== loadToken) return;   // a newer load superseded this one
      var meta = res[0], buffer = res[1];
      cur = buildState(i, meta, buffer);
      buildOffscreens();
      resetView();
      renderLive();
      drawAll();
      refreshHeader();
      prefetch(nextNavIndex(i));
    }).catch(function (e) {
      console.error("openObject(" + i + ") failed:", e);
      if (token === loadToken) {
        // cur is left unchanged (we don't silently skip a broken object). Since
        // cur.i is still the previous object, pressing Next/→ re-attempts this
        // same index, so the message doubles as a retry hint.
        flashTooltip("Failed to load object " + i + ": " +
          (e && e.message ? e.message : e) + " — press Next/→ to retry");
      }
    });
  }

  function prefetch(i) {
    if (i < 0 || i >= objects.length) return;
    fetchOk("/api/object/" + i, false).catch(function () {});
    fetchOk("/api/object/" + i + "/arrays", true).catch(function () {});
  }

  function buildState(i, meta, buffer) {
    var S = meta.box_size, N = S * S;
    var itemBytes = meta.cube_dtype === "float16" ? 2 : 4;
    var off = 0;
    var cutout = readCube(buffer, off, 3 * N, meta.cube_dtype); off += 3 * N * itemBytes;
    var galaxyBase = readCube(buffer, off, 3 * N, meta.cube_dtype); off += 3 * N * itemBytes;
    var nondwarfBase = readCube(buffer, off, 3 * N, meta.cube_dtype); off += 3 * N * itemBytes;
    var residual = readCube(buffer, off, 3 * N, meta.cube_dtype); off += 3 * N * itemBytes;

    // components arrive comp_id-sorted from the server; keep that order (it is
    // the patch order in the blob).
    var comps = meta.components;
    var baselineOn = {}, currentOn = {}, patchByCompId = {};
    var lsbId = null;
    for (var s = 0; s < comps.length; s++) {
      var c = comps[s];
      var plen = 3 * c.h * c.w;
      var patch = new Float32Array(buffer, off, plen); off += plen * 4;
      patchByCompId[c.comp_id] = patch;
      if (c.initial_membership) { baselineOn[c.comp_id] = true; currentOn[c.comp_id] = true; }
      if (c.type === "starlet_lsb") lsbId = c.comp_id;
    }

    // restore from a prior decision
    var verdict = "", comment = "", badFit = false;
    if (meta.decision) {
      verdict = meta.decision.verdict || "";
      comment = meta.decision.comment || "";
      badFit = !!meta.decision.bad_fit;
      meta.decision.removed_comp_ids.forEach(function (o) { delete currentOn[parseInt(o, 10)]; });
      meta.decision.added_comp_ids.forEach(function (o) { currentOn[parseInt(o, 10)] = true; });
    }

    return {
      i: i, meta: meta, S: S,
      cutout: cutout, galaxyBase: galaxyBase, nondwarfBase: nondwarfBase,
      residual: residual,
      comps: comps, patchByCompId: patchByCompId, lsbId: lsbId,
      baselineOn: baselineOn, currentOn: currentOn,
      selection: {}, undoStack: [], hoverId: null, outlineCache: {},
      verdict: verdict, comment: comment, badFit: badFit,
      dirty: false,
    };
  }

  function readCube(buffer, byteOffset, len, dtype) {
    if (dtype === "float16") {
      return ReconRGB.decodeHalfArray(new Uint16Array(buffer, byteOffset, len));
    }
    return new Float32Array(buffer.slice(byteOffset, byteOffset + len * 4));
  }

  // ---- Compositing -------------------------------------------------------
  // Toggling MOVES a component: galaxy ± patch and not-galaxy ∓ patch, so the
  // two panels always sum to the same full model (the bundle invariant).
  function compositeBoth() {
    var c = cur;
    var galaxy = new Float32Array(c.galaxyBase);      // copies
    var nondwarf = new Float32Array(c.nondwarfBase);
    for (var s = 0; s < c.comps.length; s++) {
      var comp = c.comps[s], id = comp.comp_id;
      var wasOn = !!c.baselineOn[id], isOn = !!c.currentOn[id];
      if (wasOn === isOn) continue;
      var sign = isOn ? +1 : -1;   // moved to galaxy -> +patch there, -patch other side
      var patch = c.patchByCompId[id];
      applyPatch(galaxy, c.S, patch, comp.bbox[0], comp.bbox[1], comp.h, comp.w, sign);
      applyPatch(nondwarf, c.S, patch, comp.bbox[0], comp.bbox[1], comp.h, comp.w, -sign);
    }
    return { galaxy: galaxy, nondwarf: nondwarf };
  }

  function applyPatch(cube, S, patch, y0, x0, h, w, sign) {
    var N = S * S;
    for (var b = 0; b < 3; b++) {
      var cubeBand = b * N, patchBand = b * h * w;
      for (var yy = 0; yy < h; yy++) {
        var cy = y0 + yy; if (cy < 0 || cy >= S) continue;
        var crow = cubeBand + cy * S;
        var prow = patchBand + yy * w;
        for (var xx = 0; xx < w; xx++) {
          var cx = x0 + xx; if (cx < 0 || cx >= S) continue;
          cube[crow + cx] += sign * patch[prow + xx];
        }
      }
    }
  }

  function buildOffscreens() {
    var c = cur, S = c.S;
    var octx = offCtx(S);
    panels.input.off = imageDataToCanvas(ReconRGB.sdssRgbImageData(c.cutout, S, octx), S);
    panels.residual.off = imageDataToCanvas(ReconRGB.sdssRgbImageData(c.residual, S, octx), S);
  }

  function renderLive() {
    var c = cur, S = c.S, N = S * S;
    var octx = offCtx(S);
    var both = compositeBoth();
    panels.galaxy.off = imageDataToCanvas(ReconRGB.sdssRgbImageData(both.galaxy, S, octx), S);
    panels.nondwarf.off = imageDataToCanvas(ReconRGB.sdssRgbImageData(both.nondwarf, S, octx), S);
    // data - galaxy: real data with the CURRENT member set subtracted, so
    // remaining neighbors/noise/fit error are all visible (unlike the
    // not-galaxy MODEL panel, which has no data noise in it).
    var dataMinusGal = new Float32Array(3 * N);
    for (var i = 0; i < 3 * N; i++) dataMinusGal[i] = c.cutout[i] - both.galaxy[i];
    panels.datamgal.off = imageDataToCanvas(ReconRGB.sdssRgbImageData(dataMinusGal, S, octx), S);
  }

  var _offCanvas = null;
  function offCtx(S) {
    if (!_offCanvas) _offCanvas = document.createElement("canvas");
    _offCanvas.width = S; _offCanvas.height = S;
    return _offCanvas.getContext("2d");
  }
  function imageDataToCanvas(imgData, S) {
    var cv = document.createElement("canvas");
    cv.width = S; cv.height = S;
    cv.getContext("2d").putImageData(imgData, 0, 0);
    return cv;
  }

  // ---- Viewport (shared across the panels; offscreen coords) -------------
  // offscreen pixel (ox, oy) corresponds to array (x = ox, y = S-1-oy).
  function resetView() {
    var c = cur, S = c.S;
    var fov = Math.min(FOV, S);
    var cw = panels.input.canvas.width || 400;
    view.scale = cw / fov;
    var gx = c.meta.gal_xpix, gy = c.meta.gal_ypix;
    if (!isFinite(gx)) gx = (S - 1) / 2;
    if (!isFinite(gy)) gy = (S - 1) / 2;
    var ogx = gx, ogy = (S - 1) - gy;       // gal center in offscreen coords
    view.ox0 = ogx - (cw / view.scale) / 2;
    view.oy0 = ogy - (cw / view.scale) / 2;
  }

  // Visible field of view, in array pixels across the canvas width.
  function currentFov() {
    return panels.input.canvas.width / view.scale;
  }

  // FOV bounds (array px): can't zoom out past the whole cube, or in past a few px.
  function fovBounds() {
    var maxFov = cur ? cur.S : FOV;          // whole cube is the most you can see
    return { min: Math.min(8, maxFov), max: maxFov };
  }
  // Clamp a zoom scale so the FOV stays within fovBounds().
  function clampScale(s) {
    var cw = panels.input.canvas.width, b = fovBounds();
    return Math.max(cw / b.max, Math.min(cw / b.min, s));
  }

  // Refresh the FOV readout, but never clobber it while the user is typing in it.
  function updateFovReadout() {
    if (!cur || !els.fov || document.activeElement === els.fov) return;
    els.fov.value = String(Math.round(currentFov()));
  }

  // Zoom to a given FOV (array px) about the current view center.
  function setFov(f) {
    if (!cur || !isFinite(f) || f <= 0) return;
    var b = fovBounds();
    f = Math.max(b.min, Math.min(b.max, f));
    var cw = panels.input.canvas.width, ch = panels.input.canvas.height;
    var cx = view.ox0 + (cw / view.scale) / 2;   // current center, offscreen coords
    var cy = view.oy0 + (ch / view.scale) / 2;
    view.scale = cw / f;
    view.ox0 = cx - (cw / view.scale) / 2;
    view.oy0 = cy - (ch / view.scale) / 2;
    drawAll();
  }

  function resetViewToFiducial() {
    if (!cur) return;
    resetView();
    drawAll();
  }

  function arrayToCanvas(x, y) {
    var S = cur.S;
    var ox = x, oy = (S - 1) - y;
    return [(ox - view.ox0) * view.scale, (oy - view.oy0) * view.scale];
  }

  // Each panel shows its own population of markers, so a crowded field splits
  // into two sparse, clickable sets: input = all, galaxy = current members,
  // not-galaxy = the rest. Hold O to hide every overlay for a clean look.
  var PANEL_SUBSET = { input: "all", galaxy: "in", nondwarf: "out" };

  function drawAll() {
    var on = !hideOverlay && showMarkers;
    drawPanel(panels.input, on ? "all" : null);
    drawPanel(panels.galaxy, on ? "in" : null);
    drawPanel(panels.nondwarf, on ? "out" : null);
    drawPanel(panels.residual, null);
    drawPanel(panels.datamgal, null);
    updateFovReadout();
  }

  function drawPanel(panel, subset) {
    var cv = panel.canvas, ctx = cv.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!panel.off) return;
    var sw = cv.width / view.scale, sh = cv.height / view.scale;
    ctx.drawImage(panel.off, view.ox0, view.oy0, sw, sh, 0, 0, cv.width, cv.height);
    if (subset) drawOverlay(ctx, subset);
  }

  // True if the component belongs to a panel's marker population.
  function inSubset(comp, subset) {
    if (subset === "all") return true;
    var inGalaxy = !!cur.currentOn[comp.comp_id];
    return subset === "in" ? inGalaxy : !inGalaxy;
  }

  // Marker radius (screen px): small fixed-ish circles — dense fields stay
  // selectable; the hover outline shows the component's real extent instead.
  function markerRadius(comp) {
    if (comp.type === "point" || comp.is_star) return 5;
    var rArr = 0.10 * Math.max(comp.h, comp.w);
    return Math.max(4, Math.min(12, rArr * view.scale));
  }

  function drawOverlay(ctx, subset) {
    var c = cur;
    for (var s = 0; s < c.comps.length; s++) {
      var comp = c.comps[s], id = comp.comp_id;
      if (id === c.lsbId) continue;   // full-frame LSB: sidebar chip, no marker
      if (!isFinite(comp.xpix) || !isFinite(comp.ypix)) continue;
      if (!inSubset(comp, subset)) continue;
      var pos = arrayToCanvas(comp.xpix, comp.ypix);
      var inGalaxy = !!c.currentOn[id];
      var baselineIn = !!c.baselineOn[id];
      var changed = inGalaxy !== baselineIn;
      var isSel = !!c.selection[id];

      // COLOR = state (selection > your-edit > in-galaxy > not-galaxy)
      var color = isSel ? COLORS.selected
        : changed ? COLORS.edited
          : inGalaxy ? COLORS.inGalaxy
            : COLORS.out;
      var r = markerRadius(comp);

      ctx.save();
      ctx.setLineDash(inGalaxy ? [] : [4, 3]);  // LINE STYLE = galaxy (solid) / not (dashed)
      ctx.lineWidth = isSel ? 2.4 : changed ? 2.2 : 1.5;
      ctx.strokeStyle = color;
      ctx.globalAlpha = (inGalaxy || changed || isSel) ? 1.0 : 0.85;
      if (comp.is_star) {
        drawDiamond(ctx, pos[0], pos[1], r);
      } else {
        ctx.beginPath();
        ctx.arc(pos[0], pos[1], r, 0, 2 * Math.PI);
        ctx.stroke();
      }
      ctx.restore();

      // BADGE = your session edit: + moved to galaxy, − moved to not-galaxy
      if (changed) drawBadge(ctx, pos, inGalaxy ? "+" : "−", color);
    }
    // hovered component: trace its actual flux outline so you see what would move
    if (c.hoverId != null && c.hoverId !== c.lsbId) {
      var hc = compById(c.hoverId);
      if (hc && inSubset(hc, subset)) drawSourceOutline(ctx, hc);
    }
    drawTargetCrosshair(ctx);
  }

  function compById(id) {
    for (var s = 0; s < cur.comps.length; s++) {
      if (cur.comps[s].comp_id === id) return cur.comps[s];
    }
    return null;
  }

  function drawDiamond(ctx, cx, cy, r) {
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx + r, cy);
    ctx.lineTo(cx, cy + r);
    ctx.lineTo(cx - r, cy);
    ctx.closePath();
    ctx.stroke();
  }

  // ---- Hover outline: the component's ACTUAL flux footprint ---------------
  // The patch is already client-side, so trace the isophote at a fraction of
  // its peak (band-summed) live: mark pixels above threshold, emit the
  // boundary edges between on/off pixels as line segments (array coords),
  // cache per component. A ~100x100 patch is a few hundred segments — trivial.
  var OUTLINE_FRAC = 0.08;   // isophote level as a fraction of the patch peak

  function computeOutline(comp) {
    var patch = cur.patchByCompId[comp.comp_id];
    var h = comp.h, w = comp.w, n = h * w;
    var sum = new Float32Array(n);
    var peak = 0;
    for (var p = 0; p < n; p++) {
      var v = patch[p] + patch[n + p] + patch[2 * n + p];
      sum[p] = v;
      if (v > peak) peak = v;
    }
    if (!(peak > 0)) return [];
    var thr = OUTLINE_FRAC * peak;
    var on = function (xx, yy) {
      return xx >= 0 && xx < w && yy >= 0 && yy < h && sum[yy * w + xx] > thr;
    };
    var y0 = comp.bbox[0], x0 = comp.bbox[1];
    var segs = [];
    for (var yy = 0; yy < h; yy++) {
      for (var xx = 0; xx < w; xx++) {
        if (!on(xx, yy)) continue;
        var x = x0 + xx, y = y0 + yy;   // pixel center, array coords
        if (!on(xx - 1, yy)) segs.push([x - 0.5, y - 0.5, x - 0.5, y + 0.5]);
        if (!on(xx + 1, yy)) segs.push([x + 0.5, y - 0.5, x + 0.5, y + 0.5]);
        if (!on(xx, yy - 1)) segs.push([x - 0.5, y - 0.5, x + 0.5, y - 0.5]);
        if (!on(xx, yy + 1)) segs.push([x - 0.5, y + 0.5, x + 0.5, y + 0.5]);
      }
    }
    return segs;
  }

  function drawSourceOutline(ctx, comp) {
    var segs = cur.outlineCache[comp.comp_id];
    if (!segs) segs = cur.outlineCache[comp.comp_id] = computeOutline(comp);
    if (!segs.length) return;
    ctx.save();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = COLORS.hover;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    for (var i = 0; i < segs.length; i++) {
      var a = arrayToCanvas(segs[i][0], segs[i][1]);
      var b = arrayToCanvas(segs[i][2], segs[i][3]);
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
    ctx.restore();
  }

  // The DESI target position is NOT a component (no guaranteed target seed by
  // design) — draw an open crosshair for orientation only.
  function drawTargetCrosshair(ctx) {
    var gx = cur.meta.gal_xpix, gy = cur.meta.gal_ypix;
    if (!isFinite(gx) || !isFinite(gy)) return;
    var pos = arrayToCanvas(gx, gy);
    var g = 5, L = 11;   // gap and arm length, screen px
    ctx.save();
    ctx.strokeStyle = COLORS.target;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(pos[0] - g - L, pos[1]); ctx.lineTo(pos[0] - g, pos[1]);
    ctx.moveTo(pos[0] + g, pos[1]); ctx.lineTo(pos[0] + g + L, pos[1]);
    ctx.moveTo(pos[0], pos[1] - g - L); ctx.lineTo(pos[0], pos[1] - g);
    ctx.moveTo(pos[0], pos[1] + g); ctx.lineTo(pos[0], pos[1] + g + L);
    ctx.stroke();
    ctx.restore();
  }

  function drawBadge(ctx, pos, sign, color) {
    var bx = pos[0] + 9, by = pos[1] - 9, r = 6.5;
    ctx.save();
    ctx.beginPath(); ctx.arc(bx, by, r, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(10,12,16,0.92)"; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = color; ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "bold 11px " + getComputedStyle(document.body).fontFamily;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(sign, bx, by);
    ctx.restore();
  }

  // ---- Hit testing & selection ------------------------------------------
  // Clicking works on the input AND both model panels; each panel picks only
  // from its own marker population (subset), so crowded fields are easy.
  function pickComponent(px, py, subset) {
    var c = cur, stack = [];
    for (var s = 0; s < c.comps.length; s++) {
      var comp = c.comps[s];
      if (comp.comp_id === c.lsbId) continue;   // LSB is toggled via its chip
      if (!isFinite(comp.xpix) || !isFinite(comp.ypix)) continue;
      if (!inSubset(comp, subset)) continue;
      var pos = arrayToCanvas(comp.xpix, comp.ypix);
      var d = Math.hypot(pos[0] - px, pos[1] - py);
      var tol = Math.max(MIN_HIT_PX, markerRadius(comp));
      if (d <= tol) { stack.push({ comp: comp, d: d }); }
    }
    if (stack.length === 0) return null;
    stack.sort(function (a, b) { return a.d - b.d; });
    return stack;  // nearest-first list of candidates within tolerance
  }

  function eventCanvasPos(ev, panel) {
    var rect = panel.canvas.getBoundingClientRect();
    return [(ev.clientX - rect.left) * (panel.canvas.width / rect.width),
            (ev.clientY - rect.top) * (panel.canvas.height / rect.height)];
  }

  // Picking is disabled while markers are hidden (no blind clicks).
  function panelInteractive(panelKey) {
    return !hideOverlay && showMarkers;
  }

  function onPanelClick(ev, panelKey) {
    if (!cur || !panelInteractive(panelKey)) return;
    var panel = panels[panelKey], subset = PANEL_SUBSET[panelKey];
    var p = eventCanvasPos(ev, panel);
    var stack = pickComponent(p[0], p[1], subset);
    if (!stack) { return; }

    // cycle through stacked components on repeated clicks near the same spot
    var pick = stack[0].comp;
    if (cur._lastClickStack && sameStack(cur._lastClickStack, stack)) {
      cur._cycleIdx = (cur._cycleIdx + 1) % stack.length;
      pick = stack[cur._cycleIdx].comp;
    } else {
      cur._cycleIdx = 0; cur._lastClickStack = stack;
    }

    if (cur.selection[pick.comp_id]) delete cur.selection[pick.comp_id];
    else cur.selection[pick.comp_id] = true;
    updateSelInfo();
    drawAll();
  }

  // Double-click = quick move: instantly send the component to the other
  // panel (undo-able), without the select-then-act step. The two single-click
  // events that precede a dblclick toggle the selection twice (net no-op).
  function onPanelDblClick(ev, panelKey) {
    if (!cur || !panelInteractive(panelKey)) return;
    var panel = panels[panelKey], subset = PANEL_SUBSET[panelKey];
    var p = eventCanvasPos(ev, panel);
    var stack = pickComponent(p[0], p[1], subset);
    if (!stack) { return; }
    var pick = stack[0].comp;
    cur.undoStack.push(snapshot());
    if (cur.currentOn[pick.comp_id]) delete cur.currentOn[pick.comp_id];
    else cur.currentOn[pick.comp_id] = true;
    delete cur.selection[pick.comp_id];
    cur._lastClickStack = null;
    afterEdit();
  }

  function sameStack(a, b) {
    if (a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) if (a[i].comp.comp_id !== b[i].comp.comp_id) return false;
    return true;
  }

  function setSelected(on) {
    var ids = Object.keys(cur.selection);
    if (ids.length === 0) { flashTooltip("No components selected"); return; }
    cur.undoStack.push(snapshot());
    ids.forEach(function (id) {
      id = parseInt(id, 10);
      if (on) cur.currentOn[id] = true; else delete cur.currentOn[id];
    });
    cur.selection = {};
    cur._lastClickStack = null;
    afterEdit();
  }

  function snapshot() {
    return JSON.parse(JSON.stringify({ currentOn: cur.currentOn }));
  }
  function undo() {
    if (cur.undoStack.length === 0) { flashTooltip("Nothing to undo"); return; }
    var snap = cur.undoStack.pop();
    cur.currentOn = snap.currentOn;
    cur.selection = {};
    afterEdit();
  }
  function resetBaseline() {
    cur.undoStack.push(snapshot());
    cur.currentOn = {};
    Object.keys(cur.baselineOn).forEach(function (id) { cur.currentOn[parseInt(id, 10)] = true; });
    cur.selection = {};
    afterEdit();
  }

  function afterEdit() {
    cur.dirty = true;
    renderLive();
    drawAll();
    updateSelInfo();
    updateLsbBtn();
    scheduleSave(false);
  }

  // ---- LSB chip (the full-frame component has no clickable marker) --------
  function toggleLsb() {
    if (!cur) return;
    if (cur.lsbId == null) {
      flashTooltip("No LSB component in this fit");
      return;
    }
    cur.undoStack.push(snapshot());
    if (cur.currentOn[cur.lsbId]) delete cur.currentOn[cur.lsbId];
    else cur.currentOn[cur.lsbId] = true;
    afterEdit();
  }
  function updateLsbBtn() {
    var b = els.btnLsb;
    if (!b) return;
    if (!cur || cur.lsbId == null) {
      b.textContent = "LSB: n/a";
      b.className = "act off";
      b.disabled = true;
      b.title = "no starlet LSB component in this fit";
      return;
    }
    b.disabled = false;
    var inGal = !!cur.currentOn[cur.lsbId];
    b.textContent = "LSB: " + (inGal ? "galaxy" : "not-galaxy");
    b.className = "act " + (inGal ? "on" : "off");
    b.title = "move the full-frame LSB component between the panels (hotkey: l)";
  }

  // ---- Marker toggle (declutter ALL panels, input included) ---------------
  function toggleModelMarkers() {
    showMarkers = !showMarkers;
    updateMarkersBtn();
    drawAll();
  }
  function updateMarkersBtn() {
    var b = els.btnMarkers;
    if (!b) return;
    b.textContent = "Markers: " + (showMarkers ? "on" : "off");
    b.className = "act " + (showMarkers ? "on" : "off");
    // Unmissable banner: with markers off, every panel ignores clicks/hover,
    // which otherwise looks exactly like a broken picker rather than a toggle.
    if (els.markersBanner) els.markersBanner.style.display = showMarkers ? "none" : "block";
  }

  // ---- Bad-fit flag (model itself wrong -> refit candidate) ---------------
  function toggleBadFit() {
    if (!cur) return;
    cur.badFit = !cur.badFit;
    cur.dirty = true;
    updateBadFitBtn();
    scheduleSave(false);
  }
  function updateBadFitBtn() {
    var b = els.btnBadfit;
    if (!b) return;
    b.disabled = !cur;
    b.textContent = cur && cur.badFit ? "bad fit ⚑" : "bad fit";
    b.className = "act badfit " + (cur && cur.badFit ? "on" : "off");
  }

  // ---- Deltas / status ---------------------------------------------------
  function deltas() {
    var removed = [], added = [];
    for (var s = 0; s < cur.comps.length; s++) {
      var id = cur.comps[s].comp_id;
      var wasOn = !!cur.baselineOn[id], isOn = !!cur.currentOn[id];
      if (wasOn && !isOn) removed.push(id);
      else if (!wasOn && isOn) added.push(id);
    }
    return { removed: removed, added: added };
  }

  function updateSelInfo() {
    var nsel = Object.keys(cur.selection).length;
    els.selinfo.textContent = nsel + " selected";
    var d = deltas();
    els.status.textContent = "to not-galaxy " + d.removed.length +
      " · to galaxy " + d.added.length +
      " · changed " + (d.removed.length + d.added.length);
    els.verdictBadge.textContent = cur.verdict || "—";
    els.verdictBadge.className = "badge v-" + (cur.verdict || "none");
  }

  // ---- Saving ------------------------------------------------------------
  function scheduleSave(immediate) {
    if (saveTimer) clearTimeout(saveTimer);
    if (immediate) { doSave(); return; }
    saveTimer = setTimeout(doSave, 400);
  }

  function doSave() {
    if (!cur) return;
    // Neutralize any pending debounced save: once we save synchronously here, a
    // later timer fire would re-run doSave against whatever `cur` is THEN (e.g.
    // the next object after navigation), writing a spurious row for it.
    if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
    var d = deltas();
    // Capture identity/state at call time: setVerdict fires doSave() and then
    // immediately navigates, so by the time this POST resolves `cur` may already
    // be the next object. Snapshotting here keeps the sidebar/objects[] update
    // (and the TARGETID we send) bound to the row we are actually saving.
    var savedI = cur.i;
    var savedVerdict = cur.verdict || "";
    var savedEdited = (d.removed.length + d.added.length) > 0;
    var savedBadFit = !!cur.badFit;
    var payload = {
      TARGETID: tidStr(cur.meta.targetid), BRICKNAME: cur.meta.brickname,
      removed_comp_ids: d.removed, added_comp_ids: d.added,
      verdict: savedVerdict, comment: els.comment.value || "",
      bad_fit: savedBadFit,
    };
    return fetch("/api/decision", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(function (r) { return r.json(); }).then(function (resp) {
      if (resp && resp.error) {   // e.g. server-side TARGETID guard rejected the write
        console.error("save rejected:", resp.error);
        flashTooltip("Save rejected: " + resp.error);
        return;
      }
      var o = objects[savedI];
      o.verdict = savedVerdict;
      o.edited = savedEdited;
      o.bad_fit = savedBadFit;
      o.status = savedVerdict ? "inspected" : (o.edited ? "edited" : "");
      updateSidebarItem(savedI);
    }).catch(function (e) { console.error("save failed", e); });
  }

  function commitPending() {
    if (cur && cur.dirty) { doSave(); }
  }

  function setVerdict(v) {
    if (!cur) return;   // buttons are live during the bootstrap window before cur loads
    cur.verdict = v;
    cur.dirty = true;
    updateSelInfo();
    doSave();
    advanceToNext();
  }

  // ---- Navigation --------------------------------------------------------
  // Advance to the next object after a verdict. When filtered, step within the
  // accepted snapshot; stopping (not falling through to hidden objects) at the
  // end of the set.
  function advanceToNext() {
    var ni = nextNavIndex(cur.i);
    if (ni < 0 || ni >= objects.length) {
      if (filterMode) flashTooltip("End of accepted set");
      return;
    }
    openObject(ni);
  }

  function go(delta) {
    if (!cur) return;
    commitPending();
    if (filterMode) {
      var pos = filteredIndices.indexOf(cur.i);
      if (pos === -1) {   // cur fell outside the subset (shouldn't happen w/ snapshot)
        openObject(filteredIndices[delta > 0 ? 0 : filteredIndices.length - 1]);
        return;
      }
      var next = pos + delta;
      if (next < 0 || next >= filteredIndices.length) {
        flashTooltip(delta > 0 ? "End of accepted set" : "Start of accepted set");
        return;
      }
      openObject(filteredIndices[next]);
      return;
    }
    openObject(cur.i + delta);
  }
  function jumpTo() {
    var v = (els.jump.value || "").trim();
    if (!v) return;
    commitPending();
    var targetIdx = -1;
    var asIdx = parseInt(v, 10);
    if (!isNaN(asIdx) && asIdx >= 0 && asIdx < objects.length && String(asIdx) === v) {
      targetIdx = asIdx;                       // raw bundle index (matches sidebar #)
    } else {                                   // else treat as TARGETID
      for (var i = 0; i < objects.length; i++) {
        if (tidStr(objects[i].targetid) === v) { targetIdx = i; break; }
      }
    }
    if (targetIdx === -1) { flashTooltip("No index/TARGETID " + v); return; }
    if (filterMode && filteredIndices.indexOf(targetIdx) === -1) {
      flashTooltip("Not in accepted set — clear the filter (f) to open it");
      return;
    }
    openObject(targetIdx);
  }

  // ---- Header / tooltip --------------------------------------------------
  function refreshHeader() {
    var c = cur;
    if (!c) return;   // filter can be toggled (f) during the bootstrap window before cur loads
    if (filterMode) {
      var pos = filteredIndices.indexOf(c.i);
      els.counter.textContent = (pos === -1 ? "–" : (pos + 1)) +
        " / " + filteredIndices.length + " accepted";
    } else {
      els.counter.textContent = (c.i + 1) + " / " + objects.length;
    }
    var tid = tidStr(c.meta.targetid);
    els.targetid.textContent = tid;
    els.targetid.href = viewerUrl(tid);
    els.brickname.textContent = c.meta.brickname;
    els.ncomp.textContent = c.meta.n_members + " / " + c.meta.n_components + " in galaxy";
    els.ncomp.className = "badge v-iso";
    els.comment.value = c.comment || "";
    updateLsbBtn();
    updateBadFitBtn();
    updateSelInfo();
    highlightSidebar(c.i);
  }

  var tipTimer = null;
  function flashTooltip(msg) {
    els.tooltip.textContent = msg;
    els.tooltip.style.opacity = "1";
    els.tooltip.style.transform = "translateX(-50%)";
    els.tooltip.style.left = "50%";
    els.tooltip.style.top = "12px";
    if (tipTimer) clearTimeout(tipTimer);
    tipTimer = setTimeout(function () { els.tooltip.style.opacity = "0"; }, 1600);
  }

  function fluxMag(flux) {
    if (flux == null || !isFinite(flux) || flux <= 0) return null;
    return 22.5 - 2.5 * Math.log10(flux);
  }

  function onHover(ev, panelKey) {
    if (!cur || !panelInteractive(panelKey)) { return; }
    var panel = panels[panelKey], subset = PANEL_SUBSET[panelKey];
    var p = eventCanvasPos(ev, panel);
    var stack = pickComponent(p[0], p[1], subset);
    var newHover = stack ? stack[0].comp.comp_id : null;
    if (newHover !== cur.hoverId) {
      cur.hoverId = newHover;
      drawAll();
    }
    if (!stack) { els.tooltip.style.opacity = "0"; return; }
    var comp = stack[0].comp;
    var inGalaxy = !!cur.currentOn[comp.comp_id];
    var changed = inGalaxy !== !!cur.baselineOn[comp.comp_id];
    var state = (inGalaxy ? "in galaxy" : "in not-galaxy") + (changed ? " · edited" : "");
    var typeStr = comp.type + (comp.is_star ? " ★" : "");
    els.tooltip.innerHTML =
      "comp " + comp.comp_id + " · " + typeStr + "<br>" +
      "mag g/r/z " + fmt(fluxMag(comp.flux_g)) + "/" + fmt(fluxMag(comp.flux_r)) +
      "/" + fmt(fluxMag(comp.flux_z)) + "<br>" +
      "g−r " + fmt(comp.gr) + " · r−z " + fmt(comp.rz) +
      " · P<sub>gmm</sub> " + fmt(comp.gmm_prob) + "<br>" + state;
    els.tooltip.style.opacity = "1";
    els.tooltip.style.transform = "none";
    els.tooltip.style.left = (ev.clientX + 14) + "px";
    els.tooltip.style.top = (ev.clientY + 14) + "px";
  }
  function fmt(v) { return (v == null || !isFinite(v)) ? "—" : v.toFixed(2); }

  // ---- Pan / zoom --------------------------------------------------------
  function onWheel(ev) {
    ev.preventDefault();
    if (!cur) return;
    var rect = ev.currentTarget.getBoundingClientRect();
    var px = (ev.clientX - rect.left) * (ev.currentTarget.width / rect.width);
    var py = (ev.clientY - rect.top) * (ev.currentTarget.height / rect.height);
    var before = [px / view.scale + view.ox0, py / view.scale + view.oy0];
    var factor = ev.deltaY < 0 ? 1.15 : 1 / 1.15;
    view.scale = clampScale(view.scale * factor);
    view.ox0 = before[0] - px / view.scale;
    view.oy0 = before[1] - py / view.scale;
    drawAll();
  }

  var dragging = false, dragLast = null;
  function onDown(ev) { dragging = true; dragLast = [ev.clientX, ev.clientY]; }
  function onUp() { dragging = false; }
  function onMove(ev) {
    if (dragging) {
      var dx = ev.clientX - dragLast[0], dy = ev.clientY - dragLast[1];
      dragLast = [ev.clientX, ev.clientY];
      var rect = ev.currentTarget.getBoundingClientRect();
      var sx = ev.currentTarget.width / rect.width;
      view.ox0 -= dx * sx / view.scale;
      view.oy0 -= dy * sx / view.scale;
      drawAll();
    }
  }

  // ---- Wiring ------------------------------------------------------------
  function wireEvents() {
    ["input", "galaxy", "nondwarf"].forEach(function (key) {
      panels[key].canvas.addEventListener("click", function (ev) {
        if (dragMoved) { dragMoved = false; return; }
        onPanelClick(ev, key);
      });
      panels[key].canvas.addEventListener("dblclick", function (ev) {
        onPanelDblClick(ev, key);
      });
      panels[key].canvas.addEventListener("mousemove", function (ev) {
        onHover(ev, key);
      });
    });
    [panels.input, panels.galaxy, panels.nondwarf, panels.residual, panels.datamgal].forEach(function (p) {
      p.canvas.addEventListener("wheel", onWheel, { passive: false });
      p.canvas.addEventListener("mousedown", onDown);
      p.canvas.addEventListener("mousemove", onMove);
    });
    window.addEventListener("mouseup", onUp);

    var resizeTimer = null;
    window.addEventListener("resize", function () {
      if (resizeTimer) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(resizeCanvases, 100);
    });

    els.btnToNondwarf.addEventListener("click", function () { setSelected(false); });
    els.btnToGalaxy.addEventListener("click", function () { setSelected(true); });
    els.btnUndo.addEventListener("click", undo);
    els.btnReset.addEventListener("click", resetBaseline);
    els.btnLsb.addEventListener("click", toggleLsb);
    els.btnBadfit.addEventListener("click", toggleBadFit);
    els.btnMarkers.addEventListener("click", toggleModelMarkers);
    els.btnAccept.addEventListener("click", function () { setVerdict("accept"); });
    els.btnUnsure.addEventListener("click", function () { setVerdict("unsure"); });
    els.btnReject.addEventListener("click", function () { setVerdict("remove"); });
    els.btnPrev.addEventListener("click", function () { go(-1); });
    els.btnNext.addEventListener("click", function () { go(1); });
    els.btnFilter.addEventListener("click", toggleAcceptFilter);
    els.btnResetView.addEventListener("click", resetViewToFiducial);
    els.fov.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { setFov(parseFloat(els.fov.value)); els.fov.blur(); }
    });
    els.fov.addEventListener("blur", updateFovReadout);
    els.jumpGo.addEventListener("click", jumpTo);
    els.jump.addEventListener("keydown", function (e) { if (e.key === "Enter") jumpTo(); });
    els.comment.addEventListener("blur", function () { cur.dirty = true; scheduleSave(true); });

    window.addEventListener("keydown", function (e) {
      if (e.target === els.comment || e.target === els.jump || e.target === els.fov) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;   // let OS/browser shortcuts through (e.g. Cmd/Ctrl+F find)
      if ((e.key === "o" || e.key === "O") && !hideOverlay) { hideOverlay = true; drawAll(); }
      if (e.key === "l" || e.key === "L") { toggleLsb(); }
      if (e.key === "m" || e.key === "M") { toggleModelMarkers(); }
      if (e.key === "b" || e.key === "B") { toggleBadFit(); }
      if (e.key === "f" || e.key === "F") { toggleAcceptFilter(); }
    });
    window.addEventListener("keyup", function (e) {
      if (e.key === "o" || e.key === "O") { hideOverlay = false; drawAll(); }
    });

    // Defensive reset: "hold O" only clears on keyup. If the window loses
    // focus while O is held (alt-tab, clicking a devtools panel, etc.), the
    // browser never delivers that keyup and hideOverlay gets stuck true --
    // every panel on every object then shows no markers and ignores clicks,
    // looking exactly like a broken picker. Force it back off on any focus loss.
    window.addEventListener("blur", function () {
      dragging = false;
      if (hideOverlay) { hideOverlay = false; drawAll(); }
    });
    document.addEventListener("visibilitychange", function () {
      if (document.hidden && hideOverlay) { hideOverlay = false; drawAll(); }
    });
  }

  // distinguish a click from a drag-release
  var dragMoved = false;
  window.addEventListener("mousemove", function () { if (dragging) dragMoved = true; });

  document.addEventListener("DOMContentLoaded", init);
})();
