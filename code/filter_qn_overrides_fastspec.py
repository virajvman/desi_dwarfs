"""
Pre-filter a fastspecfit sample file to remove targets whose redshift would be
overridden by the QuasarNet (QN) afterburner.

The flagging logic mirrors `DESISpectra.update_qso_redshifts` in
`py/fastspecfit/io.py` exactly, so the targets dropped here are precisely the
ones whose Z column fastspecfit would replace with QN's Z_NEW at read time.

Two override paths (both implemented):

  Path A -- primary QSO targets
      (DESI_TARGET & QSO_bit) != 0
      AND IS_QSO_QN_NEW_RR
      AND max(C_LYA, C_CIV, C_CIII, C_MgII, C_Hbeta, C_Halpha) > QN_thresh

  Path B -- WISE-variable QSO secondary targets
      (SCND_TARGET & WISE_VAR_QSO_bit) != 0
      AND IS_QSO_QN_NEW_RR
      AND (SPECTYPE == 'QSO' OR IS_QSO_MGII OR max(C_*) > QN_thresh)

  Special case for cmx: SV0_QSO or MINI_SV_QSO target bits in CMX_TARGET.

QN_thresh is 0.95 for {fuji, guadalupe, himalayas, iron} and 0.99 otherwise
(see `py/fastspecfit/io.py:960-968`).

For crash-avoidance the default also requires QN's Z_NEW > Z_OVERRIDE_THRESH
(default 1.5), since only high-z overrides trigger the `nLineFree == 0`
corner-case crash in `EMFit.optimize`. Set --z-thresh=0 to disable this
extra filter and match fastspecfit's override decision exactly.

Usage (interactive Perlmutter compute node recommended):

    python filter_qn_overrides.py \
        --samplefile /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs.fits \
        --outfile    /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs_qnsafe.fits \
        --specprod   iron \
        --nproc      32 \
        --report     /pscratch/sd/v/virajvm/catalog_dr1_dwarfs/desi_dr1_dwarfs_qn_dropped.fits
"""

import argparse
import os
import time
from multiprocessing import Pool

import fitsio
import numpy as np
from astropy.table import Table, unique


QN_LINE_COLS = ['C_LYA', 'C_CIV', 'C_CIII', 'C_MgII', 'C_Hbeta', 'C_Halpha']
Z_OVERRIDE_THRESH = 1.5  # default Z_NEW cut (set 0 to disable)

# Per-specprod QN confidence threshold, matching py/fastspecfit/io.py:960-968.
QN_THRESH_BY_SPECPROD = {
    'fuji': 0.95, 'guadalupe': 0.95, 'himalayas': 0.95, 'iron': 0.95,
    # Jura, Kibo, Loa, ... default to 0.99 (see else-branch upstream)
}
QN_THRESH_DEFAULT = 0.99


def _qn_thresh(specprod):
    return QN_THRESH_BY_SPECPROD.get(specprod, QN_THRESH_DEFAULT)


def _file_for(redux_root, specprod, survey, program, healpix, prefix):
    h = int(healpix)
    return os.path.join(
        redux_root, specprod, 'healpix', survey, program,
        str(h // 100), str(h),
        f'{prefix}-{survey}-{program}-{h}.fits',
    )


def _decode_array(arr):
    """Return a str ndarray for bytes-or-str input."""
    if arr.dtype.kind in ('S', 'O'):
        return np.array([
            x.decode('ascii') if isinstance(x, bytes) else str(x)
            for x in arr
        ])
    return arr


def _read_fibermap_targets(redrockfile, fm_ext='FIBERMAP'):
    """Read the FIBERMAP HDU but only the columns relevant for QSO targeting.

    Returns the full row order (one row per target in healpix coadds).
    """
    with fitsio.FITS(redrockfile) as F:
        all_cols = F[fm_ext].get_colnames()
    candidate = [
        'TARGETID',
        # primary target bitmasks across survey eras
        'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET',
        'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_MWS_TARGET', 'SV1_SCND_TARGET',
        'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_MWS_TARGET', 'SV2_SCND_TARGET',
        'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET', 'SV3_SCND_TARGET',
        'CMX_TARGET',
    ]
    cols = [c for c in candidate if c in all_cols]
    return Table(fitsio.read(redrockfile, ext=fm_ext, columns=cols))


def _empty_result(survey, program, healpix):
    return (
        survey, program, int(healpix),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array([], dtype='U6'),  # 'pathA' / 'pathB'
    )


def _scan_one_group(arg):
    """Apply the fastspecfit QN-override logic to one (survey, program, healpix)
    group and return the TARGETIDs that would have their z overridden.

    Parameters
    ----------
    arg : tuple
        (redux_root, specprod, survey, program, healpix, z_thresh)

    Returns
    -------
    (survey, program, healpix, targetids, znews, max_confs, paths)
        Empty arrays if nothing flagged.
    """
    redux_root, specprod, survey, program, healpix, z_thresh = arg

    qnfile = _file_for(redux_root, specprod, survey, program, healpix, 'qso_qn')
    mgiifile = _file_for(redux_root, specprod, survey, program, healpix, 'qso_mgii')
    redrockfile = _file_for(redux_root, specprod, survey, program, healpix, 'redrock')

    if not (os.path.isfile(qnfile) and os.path.isfile(redrockfile)):
        return _empty_result(survey, program, healpix)

    # Imported per-worker to keep multiprocessing happy on macOS spawn.
    from desitarget.targets import main_cmx_or_sv

    try:
        fm = _read_fibermap_targets(redrockfile)
    except Exception:
        return _empty_result(survey, program, healpix)

    try:
        surv_target, surv_mask, surv = main_cmx_or_sv(fm, scnd=True)
    except Exception:
        # Fibermap doesn't have target columns we can interpret; skip.
        return _empty_result(survey, program, healpix)

    n = len(fm)
    if surv == 'cmx':
        desi_target = surv_target[0]
        desi_mask = surv_mask[0]
        bits = 0
        for name in ('SV0_QSO', 'MINI_SV_QSO'):
            if name in desi_mask.names():
                bits |= int(desi_mask[name])
        IQSO = (fm[desi_target] & bits) != 0
        IWISE_VAR_QSO = np.zeros(n, bool)
    else:
        desi_target, _bgs_target, _mws_target, scnd_target = surv_target
        desi_mask, _bgs_mask, _mws_mask, scnd_mask = surv_mask
        IQSO = (fm[desi_target] & int(desi_mask['QSO'])) != 0
        if 'WISE_VAR_QSO' in scnd_mask.names():
            IWISE_VAR_QSO = (fm[scnd_target] & int(scnd_mask['WISE_VAR_QSO'])) != 0
        else:
            IWISE_VAR_QSO = np.zeros(n, bool)

    if not (np.any(IQSO) or np.any(IWISE_VAR_QSO)):
        return _empty_result(survey, program, healpix)

    # Load QN; rows are aligned with FIBERMAP (and REDSHIFTS) row-by-row.
    try:
        qn = fitsio.read(
            qnfile, ext='QN_RR',
            columns=['TARGETID', 'Z_NEW', 'IS_QSO_QN_NEW_RR'] + QN_LINE_COLS,
        )
    except Exception:
        return _empty_result(survey, program, healpix)

    if len(qn) != n or not np.array_equal(qn['TARGETID'], fm['TARGETID']):
        # Sanity: fastspecfit asserts this; skip if files are misaligned.
        return _empty_result(survey, program, healpix)

    max_conf = np.max(np.stack([qn[c] for c in QN_LINE_COLS]), axis=0)
    qn_thresh = _qn_thresh(specprod)
    qn_099 = max_conf > qn_thresh
    qn_new_rr = qn['IS_QSO_QN_NEW_RR'].astype(bool)

    # Path A: primary QSO targets.
    iqso = IQSO & qn_new_rr & qn_099

    # Path B: WISE-variable QSO secondary targets, requires MgII or SPECTYPE.
    iwise = np.zeros(n, bool)
    if np.any(IWISE_VAR_QSO):
        if os.path.isfile(mgiifile):
            try:
                mgii = fitsio.read(mgiifile, ext='MGII',
                                   columns=['TARGETID', 'IS_QSO_MGII'])
                if len(mgii) == n and np.array_equal(mgii['TARGETID'], fm['TARGETID']):
                    is_qso_mgii = mgii['IS_QSO_MGII'].astype(bool)
                else:
                    is_qso_mgii = np.zeros(n, bool)
            except Exception:
                is_qso_mgii = np.zeros(n, bool)
        else:
            is_qso_mgii = np.zeros(n, bool)

        try:
            zb = fitsio.read(redrockfile, ext='REDSHIFTS',
                             columns=['TARGETID', 'SPECTYPE'])
            if len(zb) == n and np.array_equal(zb['TARGETID'], fm['TARGETID']):
                spectype = _decode_array(zb['SPECTYPE'])
                is_qso_spectype = (spectype == 'QSO')
            else:
                is_qso_spectype = np.zeros(n, bool)
        except Exception:
            is_qso_spectype = np.zeros(n, bool)

        iwise = (
            (is_qso_spectype | is_qso_mgii | qn_099)
            & (IWISE_VAR_QSO & qn_new_rr)
        )

    flagged = iqso | iwise

    if z_thresh and z_thresh > 0:
        flagged &= (qn['Z_NEW'] > z_thresh)

    if not np.any(flagged):
        return _empty_result(survey, program, healpix)

    paths = np.array(['pathA' if a and not b else ('pathB' if b else 'both')
                      for a, b in zip(iqso[flagged], iwise[flagged])], dtype='U6')
    return (
        survey, program, int(healpix),
        fm['TARGETID'][flagged].astype(np.int64),
        qn['Z_NEW'][flagged].astype(np.float32),
        max_conf[flagged].astype(np.float32),
        paths,
    )


def _make_key(survey, program, healpix, targetid):
    return f'{survey}|{program}|{int(healpix)}|{int(targetid)}'


def _decode(s):
    return s.decode('ascii') if isinstance(s, bytes) else str(s)


def filter_sample(samplefile, outfile, specprod='iron', nproc=32,
                  redux_root=None, report=None,
                  z_thresh=Z_OVERRIDE_THRESH):
    if redux_root is None:
        redux_root = os.environ.get(
            'DESI_SPECTRO_REDUX', '/global/cfs/cdirs/desi/spectro/redux')

    print(f'Reading sample {samplefile}')
    sample = Table.read(samplefile)
    print(f'  rows: {len(sample):,}')

    sample['SURVEY'] = np.array([_decode(s) for s in sample['SURVEY']])
    sample['PROGRAM'] = np.array([_decode(s) for s in sample['PROGRAM']])

    uniq = unique(sample['SURVEY', 'PROGRAM', 'HEALPIX'])
    groups = [
        (redux_root, specprod, row['SURVEY'], row['PROGRAM'],
         int(row['HEALPIX']), z_thresh)
        for row in uniq
    ]
    print(f'Unique (survey, program, healpix) groups: {len(groups):,}')
    print(f'Using QN confidence threshold: {_qn_thresh(specprod):.2f} '
          f'(specprod={specprod})')
    if z_thresh and z_thresh > 0:
        print(f'Restricting to QN Z_NEW > {z_thresh} (crash-avoidance mode)')
    else:
        print('Z_NEW filter disabled: flagging every QN-override target')

    print(f'Scanning {len(groups):,} healpix groups with {nproc} processes...')
    t0 = time.time()
    if nproc <= 1:
        results = [_scan_one_group(g) for g in groups]
    else:
        with Pool(nproc) as p:
            results = p.map(_scan_one_group, groups,
                            chunksize=max(1, len(groups) // (nproc * 4)))
    dt = time.time() - t0
    print(f'  scan took {dt:.1f} s ({len(groups) / max(dt, 1e-9):.0f} groups/s)')

    bad_keys = set()
    drop_rows = []
    for survey, program, healpix, tids, znews, mcs, paths in results:
        for tid, znew, mc, path in zip(tids, znews, mcs, paths):
            bad_keys.add(_make_key(survey, program, healpix, tid))
            drop_rows.append((survey, program, healpix, int(tid),
                              float(znew), float(mc), str(path)))
    print(f'Flagged {len(bad_keys):,} (group, targetid) entries '
          f'as QN-override candidates')

    sample_keys = np.array([
        _make_key(s, p, h, t) for s, p, h, t in zip(
            sample['SURVEY'], sample['PROGRAM'],
            sample['HEALPIX'], sample['TARGETID'])
    ])
    mask = np.array([k in bad_keys for k in sample_keys])

    n_bad = int(mask.sum())
    print(f'Dropping {n_bad:,} / {len(sample):,} sample rows '
          f'({100. * n_bad / max(len(sample), 1):.3f}%)')

    out = sample[~mask]
    print(f'Writing {len(out):,} rows to {outfile}')
    out.write(outfile, overwrite=True)

    if report and drop_rows:
        rep = Table(rows=drop_rows,
                    names=('SURVEY', 'PROGRAM', 'HEALPIX', 'TARGETID',
                           'QN_Z_NEW', 'QN_MAX_CONF', 'PATH'))
        print(f'Writing drop report ({len(rep):,} rows) to {report}')
        rep.write(report, overwrite=True)

    return out


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split('\n\n', 1)[0],
    )
    p.add_argument('--samplefile', required=True,
                   help='Input sample FITS file with SURVEY/PROGRAM/HEALPIX/TARGETID.')
    p.add_argument('--outfile', required=True,
                   help='Output FITS file with QN-flagged targets removed.')
    p.add_argument('--specprod', default='iron',
                   help='DESI spectroscopic production (e.g. iron, loa).')
    p.add_argument('--redux-root', default=None,
                   help='Override DESI_SPECTRO_REDUX root.')
    p.add_argument('--nproc', type=int, default=32,
                   help='Parallel workers for QN file reads.')
    p.add_argument('--z-thresh', type=float, default=Z_OVERRIDE_THRESH,
                   help='Drop only if QN Z_NEW exceeds this. Set 0 to disable '
                        '(matches fastspecfit override decision exactly).')
    p.add_argument('--report', default=None,
                   help='Optional FITS path: per-target report of dropped entries.')
    args = p.parse_args()

    filter_sample(
        samplefile=args.samplefile,
        outfile=args.outfile,
        specprod=args.specprod,
        redux_root=args.redux_root,
        nproc=args.nproc,
        report=args.report,
        z_thresh=args.z_thresh,
    )


if __name__ == '__main__':
    main()
