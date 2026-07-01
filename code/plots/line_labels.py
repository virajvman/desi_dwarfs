"""
line_labels.py
==============
Overlay emission-line and stellar-absorption identifications on a spectrum,
in the style of deep-BCD line-ID figures (e.g. Izotov et al. SBS 0335-052).

Pieces
------
  LINES        : nebular EMISSION list (air wavelengths, Angstrom), each with a
                 short readable `id`, a display `label`, a `cat`egory, `strong`.
  ABSORPTION   : stellar/ISM ABSORPTION list, same structure.
  label_lines  : draw rotated labels + connector ticks, with auto anti-overlap
                 staggering for un-listed lines and a per-id STYLE dict for
                 full manual control of the ones you fine-tune.
  mark_band    : bracket + label for BAND features (D4000, G band, TiO, ...).
  air_to_vac   : air->vacuum shift (set air_to_vacuum=True for the DESI frame).

Fine-tuning model (STYLE dict, keyed by id)
-------------------------------------------
  Anchor of the label = (lambda + dx[Angstrom],  fiducial + dy[axes-fraction]).
  fiducial defaults to 0.5 = vertical centre of the panel's ylim.
  Connector: far end = `end`=(x[A], y[frac]); defaults to (true lambda, end_y).
             near end auto-sits a fixed `gap` (axes-frac) short of the label.
             -> a slanted connector appears whenever you move the label with
                `dx` (it leans from the real line position to the moved text)
                or you set `end`'s x yourself.
  loc="top": floating label just above the top edge, NO connector (use for
             lines whose peak runs off the panel). `dy` nudges it, `angle`
             still rotates it.

  Per-id keys:  hide, loc, dx, dy, angle, end, color, label, gap
  Example:
      STYLE = {
          "OIII5007": dict(loc="top"),                 # clipped -> above panel
          "Halpha":   dict(loc="top", dx=+6),
          "NII6584":  dict(dy=+0.14, dx=+7, end=(6583.5, 0.60)),   # slanted
          "SII6731":  dict(dy=+0.05, color="firebrick"),
          "NII6548":  dict(hide=True),                  # remove this one
      }

Wavelengths are AIR (as tabulated by NMSU/NIST for lambda>2000 A).
DESI is VACUUM: pass air_to_vacuum=True to shift ticks onto the vacuum frame.
"""

from dataclasses import dataclass


# ----------------------------------------------------------------------
# Air <-> vacuum (SDSS / Ciddor 1996, good to <0.01 A in the optical)
# ----------------------------------------------------------------------
def air_to_vac(lam_air):
    """Convert air wavelength(s) in Angstrom to vacuum."""
    s = 1.0e4 / lam_air
    n = (1.0 + 0.00008336624212083
         + 0.02408926869968 / (130.1065924522 - s * s)
         + 0.0001599740894897 / (38.92568793293 - s * s))
    return lam_air * n


# ----------------------------------------------------------------------
# Line container.  id is the STYLE-dict key; label is what gets drawn.
# ----------------------------------------------------------------------
@dataclass
class Line:
    id: str
    wave: float
    label: str
    cat: str
    strong: bool = False


L = Line

# ---- EMISSION -------------------------------------------------------
# categories: balmer, paschen, oxygen, forbidden, helium, highion, wr, other
LINES = [
    # Balmer
    L("H9",       3835.391, r"H9",                    "balmer"),
    L("H8",       3889.064, r"H8",                    "balmer"),
    L("Hepsilon", 3970.079, r"H$\epsilon$,$\,$[NeIII]",           "balmer"),
    L("Hdelta",   4101.742, r"H$\delta$",             "balmer",   True),
    L("Hgamma",   4340.471, r"H$\gamma$",             "balmer",   True),
    L("Hbeta",    4861.333, r"H$\beta$",              "balmer",   True),
    L("Halpha",   6562.819, r"H$\alpha$",             "balmer",   True),
    L("H10",      3797.904, r"H10",                   "balmer"),
    L("H11",      3770.637, r"H11",                   "balmer"),
    # Oxygen
    L("OII3726",  3727.5, r"[O II]",  "oxygen",   True), #average OII value
    L("OIII4363", 4363.210, r"[O III] $\lambda$4363", "oxygen",   True),
    L("OIII4959", (4958.911+5007)/2, r"[O III]", "oxygen",   True),
    L("OI6300",   6300.304, r"[O I] $\lambda$6300",   "oxygen",   True),
    L("OI6364",   6363.776, r"[O I] $\lambda$6364",   "oxygen"),
    L("OII7330",  7325, r"[O II]",  "oxygen"), #L("OII7320",  7319.990, r"[O II] $\lambda$7320",  "oxygen"), #
    L("NI7468", 7468 , r"N I 7468", "other", True),
    # Other forbidden (low ionization nebular)
    L("NeIII3869", 3868.760, r"[Ne III]", "forbidden", True), #    L("NeIII3967", 3967.470, r"[Ne III]", "forbidden"),
    L("SII4069",   4068.600, r"[S II] $\lambda$4069",   "forbidden"),
    L("NII5755",   5754.590, r"[N II] $\lambda$5755",   "forbidden"),
    L("ClIII5518", 5517.709, r"[Cl III]", "forbidden"),
    L("ClIII5538", 5537.873, r"", "forbidden"),
    L("SIII6312",  6312.060, r"[S III] $\lambda$6312",  "forbidden", True),
    L("NII6548",   6548.050, r"[N II] $\lambda$6548",   "forbidden", True),
    L("NII6584",   6583.460, r"[N II] $\lambda$6584",   "forbidden", True),
    L("SII6716",   6722, r"[S II]",   "forbidden", True),
    L("ArIII7136", 7135.790, r"[Ar III] $\lambda$7136", "forbidden", True),
    L("ArIII7751", 7751.060, r"[Ar III] $\lambda$7751", "forbidden"),
    L("SIII9069",  9068.600, r"[S III] $\lambda$9069",  "forbidden", True),
    L("NI5200",    5200.257, r"[N I] $\lambda$5200",    "forbidden"),
    L("ClII8579",  8578.700, r"[Cl II] $\lambda$8579",  "forbidden"),
    # Helium
    L("HeI4026",  4026.190, r"He I $\lambda$4026",   "helium"),
    L("HeI4471",  4471.479, r"He I $\lambda$4471",   "helium"),
    L("HeII4686", 4685.710, r"He II $\lambda$4686",  "helium",   True),
    L("HeI5876",  5875.624, r"He I $\lambda$5876",   "helium",   True),
    L("HeI6678",  6678.150, r"He I $\lambda$6678",   "helium"),
    L("HeI7065",  7065.196, r"He I $\lambda$7065",   "helium"),
    L("HeI7281",  7281.349, r"He I",   "helium"),
    # Paschen
    L("Pa12", 8750.472, r"Pa12", "paschen"),
    L("Pa11", 8862.782, r"Pa11", "paschen"),
    # Permitted metals occasionally seen in emission
    L("OI8446",   8446.359, r"O I $\lambda$8446",    "other"),
    # Wolf-Rayet bump features
    L("NIII4640",  4640.640, r"N III $\lambda$4640",    "wr"),
    L("CIII4650",  4650.250, r"C III $\lambda$4650",    "wr"),
    L("FeIII4658", 4658.050, r"[Fe III] $\lambda$4658", "wr"),
    # High-ionization / coronal (mostly AGN; off by default via exclude)
    L("ArIV4711",  4711.260, r"[Ar IV] $\lambda$4711",  "highion"),
    L("ArIV4740",  4740.120, r"[Ar IV] $\lambda$4740",  "highion"),
    # other lines near the OIII doublet
    L("FeIII4986", 4987.3, r"[Fe III] $\lambda 4986$", "other" ),
    L("HeI4922", 4923.3, r"He I $\lambda 4922$", "other")
]

# ---- ABSORPTION -----------------------------------------------------
# categories: caii, balmer_abs, gband, metals, molecular
ABSORPTION = [
    L("CaK",        3933.663, r"Ca II K $\lambda$3934",   "caii",      True),
    L("CaH",        3968.468, r"Ca II H $\lambda$3968",   "caii",      True),
    L("CN4150",     4150.000, r"CN $\lambda$4150",        "molecular"),
    L("CaI4227",    4226.730, r"Ca I $\lambda$4227",      "metals"),
    L("Gband",      4304.400, r"G band $\lambda$4304",    "gband",     True),
    L("FeI4383",    4383.500, r"Fe I $\lambda$4383",      "metals"),
    L("FeI5270",    5270.000, r"Fe I $\lambda$5270",      "metals"),
    L("FeI5335",    5335.000, r"Fe I $\lambda$5335",      "metals"),
    L("FeI5406",    5406.000, r"Fe I $\lambda$5406",      "metals"),
    L("Mgb",        5175.000, r"Mg b $\lambda$5175",      "metals",    True),
    L("NaD",        5892.900, r"Na D $\lambda$5893",      "metals",    True),
    L("TiO5449",    5449.000, r"TiO $\lambda$5449",       "molecular"),
    L("TiO6159",    6159.000, r"TiO $\lambda$6159",       "molecular"),
    L("TiO7053",    7053.000, r"TiO $\lambda$7053",       "molecular"),
    L("CaT8498",    8498.020, r"Ca II $\lambda$8498",     "caii",      True),
    L("CaT8542",    8542.090, r"Ca II $\lambda$8542",     "caii",      True),
    L("CaT8662",    8662.140, r"Ca II $\lambda$8662",     "caii",      True),
    L("MgI8807",    8806.800, r"Mg I $\lambda$8807",      "metals"),
]


# ----------------------------------------------------------------------
# Font.  usetex=True -> exact Computer Modern (needs a LaTeX install).
# ----------------------------------------------------------------------
def configure_cm_font(usetex=False):
    import matplotlib as mpl
    if usetex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["cmr10", "Computer Modern Roman"]
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["mathtext.fontset"] = "cm"
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
        mpl.rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# The workhorse.
# ----------------------------------------------------------------------
def label_lines(ax, lines=LINES, *, style=None,
                air_to_vacuum=False,
                categories=None, exclude=None, strong_only=False, hide=None,
                fiducial=0.5, angle=90, top_angle=0, color="0.15",
                fontsize=6.0, lw=0.5, gap=0.03, end_y=0.20, top_pad=0.02,
                stagger=True, level_step=0.05, max_levels=6, min_dx_frac=0.010):
    """
    Draw line-ID labels on `ax`.  x is data (wavelength, A); y is axes-fraction
    (0 bottom, 1 top), so the same call works on every panel regardless of its
    flux scale.
 
    Selection
    ---------
    strong_only, categories, exclude, hide  -- filter which lines appear.
        A line listed in `style` is ALWAYS kept (unless hidden), overriding
        these filters, so you can pull in one extra line just by styling it.
    hide        : iterable of ids to drop.  (Also: style[id]=dict(hide=True),
                  or exclude=[...] for whole categories, or delete from LINES.)
 
    Placement
    ---------
    fiducial    : axes-fraction y that dy is measured from (default 0.5 centre).
    Un-styled lines get auto anti-overlap staggering (stagger=True).
    Per-line control via `style` (dict keyed by id); recognised keys:
        hide  (bool)          remove this line
        loc   ("inside"/"top")  "top" = floating label above panel, no connector
        dx    (Angstrom)      horizontal shift of the label
        dy    (axes-frac)     vertical shift from fiducial (forces manual place)
        angle (deg)           text rotation (default 90 inside, top_angle=0 on top)
        end   ((x_A, y_frac)) far end of the connector (default (true_lambda,end_y))
        noline (bool)         draw the text only, no connector line
        color (str)           text + connector colour for this line
        label (str)           override the drawn text
        gap   (axes-frac)     per-line gap between connector and text
 
    Returns a dict {id: (kind, x_data, y_frac_anchor)} of what was drawn.
    """
    from matplotlib.transforms import blended_transform_factory
 
    style = style or {}
    hide = set(hide or [])
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    x0, x1 = ax.get_xlim()
    xlo, xhi = min(x0, x1), max(x0, x1)
    span = xhi - xlo if xhi != xlo else 1.0
 
    def to_ang(xf):
        return xlo + xf * span
 
    def to_frac(xa):
        return (xa - xlo) / span
 
    # ---- select ----
    items = []          # [Line, w_true, w_label, style]
    for ln in lines:
        s = style.get(ln.id, {})
        if ln.id in hide or s.get("hide"):
            continue
        styled = ln.id in style
        if not styled:
            if strong_only and not ln.strong:
                continue
            if categories is not None and ln.cat not in categories:
                continue
            if exclude is not None and ln.cat in exclude:
                continue
        w_true = air_to_vac(ln.wave) if air_to_vacuum else ln.wave
        w_lab = w_true + s.get("dx", 0.0)
        if not (xlo <= w_lab <= xhi):
            continue
        items.append([ln, w_true, w_lab, s])
    items.sort(key=lambda t: t[2])
 
    # ---- greedy stagger for lines without a manual dy / loc="top" ----
    min_dx = min_dx_frac * span
    last_at = []
    for it in items:
        ln, w_true, w_lab, s = it
        manual_v = ("dy" in s) or (s.get("loc") == "top")
        if manual_v:
            it.append(None)
            continue
        if not stagger:
            it.append(0)
            continue
        lev = None
        for k in range(len(last_at)):
            if w_lab - last_at[k] >= min_dx:
                lev = k
                break
        if lev is None:
            if len(last_at) < max_levels:
                lev = len(last_at)
                last_at.append(w_lab)
            else:
                lev = min(range(len(last_at)), key=lambda j: last_at[j])
                last_at[lev] = w_lab
        else:
            last_at[lev] = w_lab
        it.append(lev)
 
    # ---- draw ----
    placed = {}
    for ln, w_true, w_lab, s, lev in items:
        col = s.get("color", color)
        lab = s.get("label", ln.label)
        loc = s.get("loc", "inside")
        ang = s.get("angle", top_angle if loc == "top" else angle)
 
        if loc == "top":
            yt = 1.0 + top_pad + s.get("dy", 0.0)
            ax.text(w_lab, yt, lab, transform=trans, rotation=ang,
                    ha="center", va="bottom", fontsize=fontsize,
                    color=col, clip_on=False, zorder=6)
            placed[ln.id] = ("top", w_lab, yt)
            continue
 
        # inside: anchor + connector
        dy = s.get("dy", (lev or 0) * level_step)
        y_anchor = fiducial + dy
        g = s.get("gap", gap)
        ex, ey = s.get("end", (w_true, end_y))
 
        if not s.get("noline"):
            # unit vector from far-end to label anchor, in (x-frac, y-frac) space
            axf, ayf = to_frac(w_lab), y_anchor
            exf = to_frac(ex)
            vx, vy = axf - exf, ayf - ey
            norm = (vx * vx + vy * vy) ** 0.5 or 1.0
            ux, uy = vx / norm, vy / norm
            nxf, ny = axf - g * ux, ayf - g * uy     # near end, gap short of text
            ax.plot([ex, to_ang(nxf)], [ey, ny], transform=trans,
                    color=col, lw=lw, zorder=5, clip_on=False,
                    solid_capstyle="butt")
        ax.text(w_lab, y_anchor, lab, transform=trans, rotation=ang,
                ha="center", va="bottom", fontsize=fontsize,
                color=col, zorder=6, clip_on=False)
        placed[ln.id] = ("inside", w_lab, y_anchor)
 
    return placed


def mark_band(ax, w0, w1, label, *, y=0.90, drop=0.02, color="0.15",
              fontsize=6.0, air_to_vacuum=False):
    """
    Bracket over [w0, w1] (Angstrom) with a centred label, for BAND features a
    single tick can't represent:
        mark_band(ax, 3850, 4080, r"D4000")
        mark_band(ax, 4285, 4320, r"G band")
        mark_band(ax, 3630, 3660, r"Balmer jump")
        mark_band(ax, 7050, 7300, r"TiO")
    y, drop are axes fractions (bracket at y, tips drop below it).
    """
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    if air_to_vacuum:
        w0, w1 = air_to_vac(w0), air_to_vac(w1)
    ax.plot([w0, w0, w1, w1], [y - drop, y, y, y - drop], transform=trans,
            color=color, lw=0.6, clip_on=False, zorder=6)
    ax.text(0.5 * (w0 + w1), y + 0.005, label, transform=trans, ha="center",
            va="bottom", fontsize=fontsize, color=color, clip_on=False, zorder=6)