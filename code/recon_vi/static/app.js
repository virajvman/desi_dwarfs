// app.js -- recon_vi frontend: navigation, source overlay, flux-space
// compositing, verdicts, autosave. All compositing is client-side; only
// navigation and decision-save hit the backend.

(function () {
  "use strict";

  var PIXSCALE = 0.262;        // arcsec/pix (DR9/DR10 grz)
  var FOV = 200;               // fiducial viewport size in array pixels
  var MIN_HIT_PX = 8;          // min click tolerance (screen px)
  // Membership/provenance palette (saturated so it pops on RGB, like the LS
  // viewer). Channels are kept orthogonal: COLOR = state, LINE STYLE = in/out
  // of cube (solid/dashed), BADGE = your session edits. Type stays on hover.
  var COLORS = {
    inCube: "#2ee36a",     // in the fiducial cube
    out: "#ff5d6c",        // subtracted / excluded from the cube
    edited: "#ffb02e",     // changed by you this session (undo candidate)
    target: "#22e0e0",     // protected target source
    selected: "#ffe14d",   // pending selection
    flash: "#9aa0ff",
  };

  // ---- DOM ---------------------------------------------------------------
  var $ = function (id) { return document.getElementById(id); };
  var panels = {
    input: { canvas: null, off: null, label: "input" },
    base: { canvas: null, off: null, label: "base" },
    live: { canvas: null, off: null, label: "live" },
  };
  var els = {};

  // ---- State -------------------------------------------------------------
  var objects = [];            // /api/objects list
  var inspectorName = "";
  var cur = null;              // current object working state
  var view = { scale: 1, ox0: 0, oy0: 0 };  // shared offscreen viewport
  var flashOn = false;
  var saveTimer = null;

  // ---- Init --------------------------------------------------------------
  function init() {
    panels.input.canvas = $("canvas-input");
    panels.base.canvas = $("canvas-base");
    panels.live.canvas = $("canvas-live");
    els = {
      sidebar: $("sidebar-list"),
      counter: $("counter"), targetid: $("targetid"), brickname: $("brickname"),
      variant: $("variant-badge"),
      btnRemove: $("btn-remove"), btnAdd: $("btn-add"),
      btnUndo: $("btn-undo"), btnReset: $("btn-reset"),
      btnAccept: $("btn-accept"), btnUnsure: $("btn-unsure"), btnReject: $("btn-reject"),
      btnPrev: $("btn-prev"), btnNext: $("btn-next"),
      jump: $("jump-input"), jumpGo: $("btn-jump"),
      comment: $("comment"), selinfo: $("sel-info"), status: $("status-line"),
      tooltip: $("tooltip"), verdictBadge: $("verdict-badge"),
    };
    wireEvents();
    bootstrap();
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

  // ---- Sidebar -----------------------------------------------------------
  function buildSidebar() {
    var html = "";
    for (var i = 0; i < objects.length; i++) {
      var o = objects[i];
      html += '<div class="sb-item" data-i="' + i + '" id="sb-' + i + '">' +
        '<span class="sb-idx">' + i + '</span>' +
        '<span class="sb-tid">' + o.targetid + '</span>' +
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
    mark.textContent = t;
    mark.className = "sb-mark v-" + (o.verdict || (o.edited ? "edited" : "none"));
    var item = $("sb-" + i);
    if (item) item.classList.toggle("disabled", !!o.toggle_disabled);
  }

  function highlightSidebar(i) {
    els.sidebar.querySelectorAll(".sb-item.active").forEach(function (el) {
      el.classList.remove("active");
    });
    var item = $("sb-" + i);
    if (item) { item.classList.add("active"); item.scrollIntoView({ block: "nearest" }); }
  }

  // ---- Object loading ----------------------------------------------------
  function openObject(i) {
    if (i < 0 || i >= objects.length) return;
    var metaP = fetch("/api/object/" + i).then(function (r) { return r.json(); });
    var arrP = fetch("/api/object/" + i + "/arrays").then(function (r) { return r.arrayBuffer(); });
    Promise.all([metaP, arrP]).then(function (res) {
      var meta = res[0], buffer = res[1];
      cur = buildState(i, meta, buffer);
      buildOffscreens();
      resetView();
      renderLive();
      drawAll();
      refreshHeader();
      prefetch(i + 1);
    }).catch(function (e) { console.error(e); });
  }

  function prefetch(i) {
    if (i < 0 || i >= objects.length) return;
    fetch("/api/object/" + i).catch(function () {});
    fetch("/api/object/" + i + "/arrays").catch(function () {});
  }

  function buildState(i, meta, buffer) {
    var S = meta.box_size, N = S * S;
    var itemBytes = meta.cube_dtype === "float16" ? 2 : 4;
    var off = 0;
    var baseCube = readCube(buffer, off, 3 * N, meta.cube_dtype); off += 3 * N * itemBytes;
    var cutout = readCube(buffer, off, 3 * N, meta.cube_dtype); off += 3 * N * itemBytes;

    var sources = meta.sources.slice().sort(function (a, b) { return a.objid - b.objid; });
    var baselineOn = {}, currentOn = {}, patchByObjid = {};
    for (var s = 0; s < sources.length; s++) {
      var src = sources[s];
      var plen = 3 * src.h * src.w;
      var patch = new Float32Array(buffer, off, plen); off += plen * 4;
      patchByObjid[src.objid] = patch;
      if (src.initial_membership) { baselineOn[src.objid] = true; currentOn[src.objid] = true; }
    }

    // restore from a prior decision
    var verdict = "", comment = "";
    if (meta.decision) {
      verdict = meta.decision.verdict || "";
      comment = meta.decision.comment || "";
      meta.decision.removed_objids.forEach(function (o) { delete currentOn[parseInt(o, 10)]; });
      meta.decision.added_objids.forEach(function (o) { currentOn[parseInt(o, 10)] = true; });
    }

    return {
      i: i, meta: meta, S: S,
      baseCube: baseCube, cutout: cutout,
      sources: sources, patchByObjid: patchByObjid,
      baselineOn: baselineOn, currentOn: currentOn,
      selection: {}, undoStack: [],
      verdict: verdict, comment: comment,
      liveImageData: null, baseImageData: null, cutoutImageData: null,
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
  function composite() {
    var c = cur, N = c.S * c.S;
    var out = new Float32Array(c.baseCube);  // copy
    for (var s = 0; s < c.sources.length; s++) {
      var src = c.sources[s], id = src.objid;
      var wasOn = !!c.baselineOn[id], isOn = !!c.currentOn[id];
      if (wasOn === isOn) continue;
      var sign = isOn ? +1 : -1;  // added -> +model, removed -> -model
      applyPatch(out, c.S, c.patchByObjid[id], src.bbox[0], src.bbox[1], src.h, src.w, sign);
    }
    return out;
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
    c.cutoutImageData = ReconRGB.sdssRgbImageData(c.cutout, S, octx);
    c.baseImageData = ReconRGB.sdssRgbImageData(c.baseCube, S, octx);
    panels.input.off = imageDataToCanvas(c.cutoutImageData, S);
    panels.base.off = imageDataToCanvas(c.baseImageData, S);
  }

  function renderLive() {
    var c = cur, S = c.S;
    var octx = offCtx(S);
    var live = c.meta.toggle_disabled ? c.baseCube : composite();
    c.liveImageData = ReconRGB.sdssRgbImageData(live, S, octx);
    panels.live.off = imageDataToCanvas(c.liveImageData, S);
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

  // ---- Viewport (shared across the 3 panels; offscreen coords) ----------
  // offscreen pixel (ox, oy) corresponds to array (x = ox, y = S-1-oy).
  function resetView() {
    var c = cur, S = c.S;
    var fov = Math.min(FOV, S);
    var cw = panels.input.canvas.width || 460;
    view.scale = cw / fov;
    var gx = c.meta.gal_xpix, gy = c.meta.gal_ypix;
    if (!isFinite(gx)) gx = (S - 1) / 2;
    if (!isFinite(gy)) gy = (S - 1) / 2;
    var ogx = gx, ogy = (S - 1) - gy;       // gal center in offscreen coords
    view.ox0 = ogx - (cw / view.scale) / 2;
    view.oy0 = ogy - (cw / view.scale) / 2;
  }

  function arrayToCanvas(x, y) {
    var S = cur.S;
    var ox = x, oy = (S - 1) - y;
    return [(ox - view.ox0) * view.scale, (oy - view.oy0) * view.scale];
  }
  function canvasToArray(px, py) {
    var S = cur.S;
    var ox = px / view.scale + view.ox0;
    var oy = py / view.scale + view.oy0;
    return [ox, (S - 1) - oy];
  }

  function drawAll() {
    drawPanel(panels.input, true);
    drawPanel(panels.base, flashOn);
    drawPanel(panels.live, flashOn);
  }

  function drawPanel(panel, withOverlay) {
    var cv = panel.canvas, ctx = cv.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!panel.off) return;
    var sw = cv.width / view.scale, sh = cv.height / view.scale;
    ctx.drawImage(panel.off, view.ox0, view.oy0, sw, sh, 0, 0, cv.width, cv.height);
    if (withOverlay && !cur.meta.toggle_disabled) drawOverlay(ctx);
  }

  function drawOverlay(ctx) {
    var c = cur;
    for (var s = 0; s < c.sources.length; s++) {
      var src = c.sources[s], id = src.objid;
      if (!isFinite(src.xpix) || !isFinite(src.ypix)) continue;
      var pos = arrayToCanvas(src.xpix, src.ypix);
      var inCube = !!c.currentOn[id];
      var baselineIn = !!c.baselineOn[id];
      var changed = inCube !== baselineIn;
      var isTarget = (id === c.meta.target_objid) || (src.separation < 1.0);
      var isSel = !!c.selection[id];

      // COLOR = state (selection > target > your-edit > kept > subtracted)
      var color = isSel ? COLORS.selected
        : isTarget ? COLORS.target
          : changed ? COLORS.edited
            : inCube ? COLORS.inCube
              : COLORS.out;
      var aScreen = (src.shape_r > 0 ? src.shape_r / PIXSCALE : 0) * view.scale;

      ctx.save();
      ctx.setLineDash(inCube ? [] : [4, 3]);   // LINE STYLE = in (solid) / out (dashed)
      ctx.lineWidth = isTarget ? 2.6 : isSel ? 2.4 : changed ? 2.2 : 1.5;
      ctx.strokeStyle = color;
      ctx.globalAlpha = (inCube || isTarget || changed || isSel) ? 1.0 : 0.85;
      if (src.type === "PSF" || aScreen < 3) {
        ctx.beginPath();
        ctx.arc(pos[0], pos[1], Math.max(4, aScreen), 0, 2 * Math.PI);
        ctx.stroke();
      } else {
        drawEllipse(ctx, pos[0], pos[1], aScreen, src);
      }
      ctx.restore();

      // BADGE = your session edit: + added back, − removed
      if (changed) drawBadge(ctx, pos, (inCube && !baselineIn) ? "+" : "−", color);
    }
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

  function drawEllipse(ctx, cx, cy, aScreen, src) {
    var ee = Math.hypot(src.shape_e1, src.shape_e2);
    var ba = (1 - ee) / (1 + ee);              // b/a
    var theta = 0.5 * Math.atan2(src.shape_e2, src.shape_e1);
    var bScreen = aScreen * ba;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-theta);   // y is flipped on canvas; sign adjustable if mirrored
    ctx.beginPath();
    ctx.ellipse(0, 0, Math.max(2, aScreen), Math.max(1.5, bScreen), 0, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  // ---- Hit testing & selection ------------------------------------------
  function pickSource(px, py) {
    var c = cur, best = null, bestD = Infinity, stack = [];
    for (var s = 0; s < c.sources.length; s++) {
      var src = c.sources[s];
      if (!isFinite(src.xpix) || !isFinite(src.ypix)) continue;
      var pos = arrayToCanvas(src.xpix, src.ypix);
      var d = Math.hypot(pos[0] - px, pos[1] - py);
      var aScreen = (src.shape_r > 0 ? src.shape_r / PIXSCALE : 0) * view.scale;
      var tol = Math.max(MIN_HIT_PX, aScreen);
      if (d <= tol) { stack.push({ src: src, d: d }); }
      if (d < bestD) { bestD = d; best = src; }
    }
    if (stack.length === 0) return null;
    stack.sort(function (a, b) { return a.d - b.d; });
    return stack;  // nearest-first list of candidates within tolerance
  }

  function onPanelClick(ev) {
    if (!cur || cur.meta.toggle_disabled) return;
    var rect = panels.input.canvas.getBoundingClientRect();
    var scaleX = panels.input.canvas.width / rect.width;
    var scaleY = panels.input.canvas.height / rect.height;
    var px = (ev.clientX - rect.left) * scaleX;
    var py = (ev.clientY - rect.top) * scaleY;
    var stack = pickSource(px, py);
    if (!stack) { return; }

    // cycle through stacked sources on repeated clicks near the same spot
    var pick = stack[0].src;
    if (cur._lastClickStack && sameStack(cur._lastClickStack, stack)) {
      cur._cycleIdx = (cur._cycleIdx + 1) % stack.length;
      pick = stack[cur._cycleIdx].src;
    } else {
      cur._cycleIdx = 0; cur._lastClickStack = stack;
    }

    var isTarget = (pick.objid === cur.meta.target_objid) || (pick.separation < 1.0);
    if (isTarget && !ev.shiftKey) {
      flashTooltip("Target source — Shift-click to select it");
      return;
    }
    if (cur.selection[pick.objid]) delete cur.selection[pick.objid];
    else cur.selection[pick.objid] = true;
    updateSelInfo();
    drawPanel(panels.input, true);
  }

  function sameStack(a, b) {
    if (a.length !== b.length) return false;
    for (var i = 0; i < a.length; i++) if (a[i].src.objid !== b[i].src.objid) return false;
    return true;
  }

  function setSelected(on) {
    var ids = Object.keys(cur.selection);
    if (ids.length === 0) { flashTooltip("No sources selected"); return; }
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
    scheduleSave(false);
  }

  // ---- Deltas / status ---------------------------------------------------
  function deltas() {
    var removed = [], added = [];
    for (var s = 0; s < cur.sources.length; s++) {
      var id = cur.sources[s].objid;
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
    els.status.textContent = "removed " + d.removed.length + " · added " + d.added.length +
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
    var d = deltas();
    var payload = {
      TARGETID: cur.meta.targetid, BRICKNAME: cur.meta.brickname,
      removed_objids: d.removed, added_objids: d.added,
      verdict: cur.verdict || "", comment: els.comment.value || "",
      toggle_disabled: cur.meta.toggle_disabled,
    };
    return fetch("/api/decision", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(function (r) { return r.json(); }).then(function () {
      var o = objects[cur.i];
      o.verdict = cur.verdict || "";
      o.edited = (d.removed.length + d.added.length) > 0;
      o.status = cur.verdict ? "inspected" : (o.edited ? "edited" : "");
      updateSidebarItem(cur.i);
    }).catch(function (e) { console.error("save failed", e); });
  }

  function commitPending() {
    if (cur && cur.dirty) { doSave(); }
  }

  function setVerdict(v) {
    cur.verdict = v;
    cur.dirty = true;
    updateSelInfo();
    doSave();
    if (cur.i + 1 < objects.length) openObject(cur.i + 1);
  }

  // ---- Navigation --------------------------------------------------------
  function go(delta) {
    commitPending();
    openObject(cur.i + delta);
  }
  function jumpTo() {
    var v = (els.jump.value || "").trim();
    if (!v) return;
    commitPending();
    var asIdx = parseInt(v, 10);
    if (!isNaN(asIdx) && asIdx >= 0 && asIdx < objects.length &&
        String(asIdx) === v) { openObject(asIdx); return; }
    // else treat as TARGETID
    for (var i = 0; i < objects.length; i++) {
      if (String(objects[i].targetid) === v) { openObject(i); return; }
    }
    flashTooltip("No index/TARGETID " + v);
  }

  // ---- Header / tooltip --------------------------------------------------
  function refreshHeader() {
    var c = cur;
    els.counter.textContent = (c.i + 1) + " / " + objects.length;
    els.targetid.textContent = c.meta.targetid;
    els.brickname.textContent = c.meta.brickname;
    els.variant.textContent = c.meta.toggle_disabled ? "view-only (z<0.005)" : c.meta.recon_variant;
    els.variant.className = "badge " + (c.meta.toggle_disabled ? "v-disabled" : "v-iso");
    els.comment.value = c.comment || "";
    var disabled = c.meta.toggle_disabled;
    [els.btnRemove, els.btnAdd, els.btnUndo, els.btnReset].forEach(function (b) {
      b.disabled = disabled;
    });
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

  function onHover(ev) {
    if (!cur || cur.meta.toggle_disabled) { return; }
    var rect = panels.input.canvas.getBoundingClientRect();
    var px = (ev.clientX - rect.left) * (panels.input.canvas.width / rect.width);
    var py = (ev.clientY - rect.top) * (panels.input.canvas.height / rect.height);
    var stack = pickSource(px, py);
    if (!stack) { els.tooltip.style.opacity = "0"; return; }
    var src = stack[0].src;
    var inCube = !!cur.currentOn[src.objid];
    var changed = inCube !== !!cur.baselineOn[src.objid];
    var state = (inCube ? "in cube" : "subtracted") + (changed ? " · edited" : "");
    els.tooltip.innerHTML =
      "objid " + src.objid + " · " + src.type + "<br>" +
      "mag_r " + fmt(src.mag_r) + " · sep " + fmt(src.separation) + '"<br>' + state;
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
    view.scale *= factor;
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
    panels.input.canvas.addEventListener("click", function (ev) {
      if (dragMoved) { dragMoved = false; return; }
      onPanelClick(ev);
    });
    panels.input.canvas.addEventListener("mousemove", onHover);
    [panels.input, panels.base, panels.live].forEach(function (p) {
      p.canvas.addEventListener("wheel", onWheel, { passive: false });
      p.canvas.addEventListener("mousedown", onDown);
      p.canvas.addEventListener("mousemove", onMove);
    });
    window.addEventListener("mouseup", onUp);

    els.btnRemove.addEventListener("click", function () { setSelected(false); });
    els.btnAdd.addEventListener("click", function () { setSelected(true); });
    els.btnUndo.addEventListener("click", undo);
    els.btnReset.addEventListener("click", resetBaseline);
    els.btnAccept.addEventListener("click", function () { setVerdict("accept"); });
    els.btnUnsure.addEventListener("click", function () { setVerdict("unsure"); });
    els.btnReject.addEventListener("click", function () { setVerdict("remove"); });
    els.btnPrev.addEventListener("click", function () { go(-1); });
    els.btnNext.addEventListener("click", function () { go(1); });
    els.jumpGo.addEventListener("click", jumpTo);
    els.jump.addEventListener("keydown", function (e) { if (e.key === "Enter") jumpTo(); });
    els.comment.addEventListener("blur", function () { cur.dirty = true; scheduleSave(true); });

    window.addEventListener("keydown", function (e) {
      if (e.target === els.comment || e.target === els.jump) return;
      if ((e.key === "o" || e.key === "O") && !flashOn) { flashOn = true; drawAll(); }
    });
    window.addEventListener("keyup", function (e) {
      if (e.key === "o" || e.key === "O") { flashOn = false; drawAll(); }
    });
  }

  // distinguish a click from a drag-release
  var dragMoved = false;
  window.addEventListener("mousemove", function () { if (dragging) dragMoved = true; });

  document.addEventListener("DOMContentLoaded", init);
})();
