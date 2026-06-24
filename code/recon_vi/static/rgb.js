// rgb.js -- faithful JS port of desi_lowz_funcs.sdss_rgb() + a float16 decoder.
//
// The cubes are laid out (3, S, S) in band order [g, r, z] (matching every
// np.array([... "g","r","z"]) in the pipeline). sdss_rgb maps:
//   g (cube band 0) -> scale 6.0 -> RGB plane 2 (blue)
//   r (cube band 1) -> scale 3.4 -> RGB plane 1 (green)
//   z (cube band 2) -> scale 2.2 -> RGB plane 0 (red)
// with m = 0.03, Q = 20, the mean taken over the 3 bands, then clip(0, 1).
//
// CRITICAL: these scales/m are the FUNCTION DEFAULTS of sdss_rgb, NOT the
// internal `rgbscales` fallback (g=2.5,r=1.5,z=0.4,m=0.02). aperture_cogs calls
// sdss_rgb(total_model, ["g","r","z"]) with no `scales=`/`m=`, so the defaults
// (scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03) win -- they overwrite
// the fallback via rgbscales.update(scales). The GUI must use the SAME numbers
// or every recon image renders ~2.4x too faint with the wrong color balance.

(function (global) {
  "use strict";

  var M = 0.03;
  var Q = 20.0;
  var SQRTQ = Math.sqrt(Q);
  // (scale, rgbPlane) per band, in cube band order g, r, z.
  var SCALE_G = 6.0, PLANE_G = 2;
  var SCALE_R = 3.4, PLANE_R = 1;
  var SCALE_Z = 2.2, PLANE_Z = 0;

  // Decode an IEEE-754 half (Uint16) to a JS number.
  function halfToFloat(h) {
    var s = (h & 0x8000) >> 15;
    var e = (h & 0x7c00) >> 10;
    var f = h & 0x03ff;
    if (e === 0) {
      return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    } else if (e === 0x1f) {
      return f ? NaN : (s ? -Infinity : Infinity);
    }
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function decodeHalfArray(u16) {
    var out = new Float32Array(u16.length);
    for (var i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
    return out;
  }

  // cube: Float32Array length 3*S*S, [g(0..N), r(N..2N), z(2N..3N)].
  // Returns an ImageData (S x S), flipped vertically so row 0 of the array is
  // drawn at the BOTTOM (matplotlib origin="lower").
  function sdssRgbImageData(cube, S, ctx) {
    var N = S * S;
    var img = ctx.createImageData(S, S);
    var data = img.data;
    var gOff = 0, rOff = N, zOff = 2 * N;
    for (var y = 0; y < S; y++) {
      // flip: array row y -> image row (S-1-y)
      var dstRow = (S - 1 - y) * S;
      var srcRow = y * S;
      for (var x = 0; x < S; x++) {
        var p = srcRow + x;
        var ig = cube[gOff + p], ir = cube[rOff + p], iz = cube[zOff + p];

        var ag = ig * SCALE_G + M; if (ag < 0) ag = 0;
        var ar = ir * SCALE_R + M; if (ar < 0) ar = 0;
        var az = iz * SCALE_Z + M; if (az < 0) az = 0;
        var I = (ag + ar + az) / 3.0;
        var fI = Math.asinh(Q * I) / SQRTQ;
        var Idiv = I > 0 ? I : 1e-6;

        var rgb = [0, 0, 0];
        rgb[PLANE_G] = (ig * SCALE_G + M) * fI / Idiv;
        rgb[PLANE_R] = (ir * SCALE_R + M) * fI / Idiv;
        rgb[PLANE_Z] = (iz * SCALE_Z + M) * fI / Idiv;

        var di = (dstRow + x) * 4;
        data[di] = clip255(rgb[0]);
        data[di + 1] = clip255(rgb[1]);
        data[di + 2] = clip255(rgb[2]);
        data[di + 3] = 255;
      }
    }
    return img;
  }

  function clip255(v) {
    v = v * 255.0;
    if (v < 0) return 0;
    if (v > 255) return 255;
    return v;
  }

  global.ReconRGB = {
    halfToFloat: halfToFloat,
    decodeHalfArray: decodeHalfArray,
    sdssRgbImageData: sdssRgbImageData,
  };
})(window);
