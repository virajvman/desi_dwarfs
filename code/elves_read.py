#!/usr/bin/env python3
"""
Catalog of CONFIRMED ELVES dwarfs: g-band mag, g-r colour, distance (Mpc).

Sources (in ~/DESI2_LOWZ/desi_dwarfs/data/elves):
  Carlsten+2022 (VizieR J/ApJ/933/47, CDS): ReadMe, table6.dat, table9.dat
  Li+2025 ELVES-Dwarf Table 4:              li25_elves_dwarfs.txt

Output: confirmed_elves_dwarfs.csv
  name, host, survey, m_g, g_r, converted_from_gi, dist_mpc, dist_sbf_mpc

Colour: homogenised to g-r. ELVES table9 stores g and a second band (r or i,
per `Filt`); g-i rows (CFHT/MegaCam) are converted with Carlsten+2021a Eq.1
(ApJ 922, 267), (g-i) = 1.53(g-r) - 0.032, i.e. (g-r) = ((g-i)+0.032)/1.53.
DECaLS rows and Li+25 are native g-r and pass through unchanged.

Distance: dist_mpc = adopted HOST distance; dist_sbf_mpc = individual SBF measurement.
"""
import os
import re
import numpy as np
import pandas as pd
from astropy.io import ascii

DATA_DIR = os.path.expanduser("~/DESI2_LOWZ/desi_dwarfs/data/elves")
OUT_CSV  = os.path.join(DATA_DIR, "confirmed_elves_dwarfs.csv")

SCHEMA = ["name", "host", "survey", "m_g", "g_r",
          "converted_from_gi", "dist_mpc", "dist_sbf_mpc"]

# Carlsten+2021a (ApJ 922, 267) Eq.1, MIST-derived for the CFHT system
def gi_to_gr(gi):
    return (gi + 0.032) / 1.53


# ----------------------------------------------------------------------
# 1. Carlsten+2022 main ELVES catalog (CDS format)
# ----------------------------------------------------------------------
def _carlsten_join(t6, t9):
    d6 = t6[["Name", "Host", "D-Host", "D-SBF"]].to_pandas()
    d6.columns = ["name", "host", "dist_mpc", "dist_sbf_mpc"]

    d9 = t9[["Name", "gmag", "rimag", "Filt"]].to_pandas()
    d9.columns = ["name", "m_g", "_ri", "_filt"]

    raw = d9["m_g"] - d9["_ri"]                       # g - (r or i)
    band = d9["_filt"].astype(str).str.strip().str.lower()
    is_i = band.str[-1].eq("i")                       # second band is i -> g-i
    print("table9 Filt values:", sorted(band.dropna().unique()))
    d9["g_r"] = np.where(is_i, gi_to_gr(raw), raw)
    d9["converted_from_gi"] = is_i

    out = d6.merge(d9[["name", "m_g", "g_r", "converted_from_gi"]],
                   on="name", how="left")
    out["survey"] = "ELVES (Carlsten+22)"
    print(f"Carlsten: {len(out)} confirmed; photometry matched for "
          f"{out['m_g'].notna().sum()}, of which converted from g-i: "
          f"{int(out['converted_from_gi'].fillna(False).sum())}")
    return out[SCHEMA]


def read_carlsten(data_dir):
    readme = os.path.join(data_dir, "ReadMe")
    t6 = ascii.read(os.path.join(data_dir, "table6.dat"), readme=readme, format="cds")
    t9 = ascii.read(os.path.join(data_dir, "table9.dat"), readme=readme, format="cds")
    return _carlsten_join(t6, t9)


# ----------------------------------------------------------------------
# 2. Li+2025 ELVES-Dwarf Table 4  (native g-r)
# ----------------------------------------------------------------------
STATUS = {"Conf.", "Rej.", "Unconf.", "No Obs."}
HOST_RE = re.compile(r"^(.*?)\s*\(\s*([\d.]+)\s*Mpc\s*\)\s*$")


def _val(s):
    s = s.strip()
    if not s or s.lower() == "cdots":
        return np.nan
    s = s.split("+or-")[0].strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _dsbf(s):
    s = s.strip()
    if not s or s.lower() == "cdots":
        return np.nan
    if s[0] in "<>":
        return np.nan
    s = re.split(r"[_^]", s.replace("$", ""), maxsplit=1)[0]
    s = s.replace("{", "").replace("}", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_li25(path):
    rows, host, host_d = [], None, np.nan
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            nonempty = [x for x in fields if x.strip()]
            if len(nonempty) == 1:
                m = HOST_RE.match(nonempty[0].strip())
                if m:
                    host, host_d = m.group(1).strip(), float(m.group(2))
                continue
            if len(fields) < 13 or fields[5].strip() not in STATUS:
                continue
            rows.append(dict(
                name=fields[0].strip(), host=host,
                survey="ELVES-Dwarf (Li+25)", status=fields[5].strip(),
                m_g=_val(fields[8]), g_r=_val(fields[9]),
                converted_from_gi=False,
                dist_mpc=host_d, dist_sbf_mpc=_dsbf(fields[3]),
            ))
    df = pd.DataFrame(rows)
    conf = df[df.status == "Conf."].copy()
    print(f"Li+25: {len(conf)} confirmed")
    return conf[SCHEMA]


# ----------------------------------------------------------------------
def main():
    pieces = []
    try:
        pieces.append(read_carlsten(DATA_DIR))
    except Exception as e:
        print(f"[warn] Carlsten tables skipped: {e}")
    pieces.append(read_li25(os.path.join(DATA_DIR, "li25_elves_dwarfs.txt")))

    cat = pd.concat(pieces, ignore_index=True)
    cat.to_csv(OUT_CSV, index=False)
    print(f"\nTotal confirmed dwarfs: {len(cat)}")
    print(cat.groupby("survey").size().to_string())
    print(f"wrote {OUT_CSV}\n")
    print(cat.head(8).to_string(index=False))


if __name__ == "__main__":
    main()