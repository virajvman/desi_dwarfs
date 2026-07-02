"""recon_vi_scarlet -- SCARLET component visual-inspection & fine-tuning GUI.

Sibling of ``recon_vi`` (which edits Tractor-source reconstructions); this tool
finetunes the SCARLET component GROUPING: every component lives in exactly one
of two complementary model panels (galaxy | not-galaxy) and a toggle MOVES it
between them. The decision CSV records component add/remove deltas relative to
the fitter's ``initial_membership`` plus post-VI magnitudes.

Two halves:

* ``build_bundle`` -- packs a hand-picked VI catalog's objects out of the
  stage-2 per-brick bundle store (``scarlet_photo.consolidate`` output) into a
  single HDF5 bundle to ``scp`` down.
* ``server`` -- run locally; a thin Flask backend that serves one object at a
  time to the static JS/canvas frontend in ``static/`` and autosaves a
  replayable decision CSV.

See ``README.md`` for the end-to-end workflow. ``recon_vi`` is never imported
or modified.
"""

__all__ = ["build_bundle", "server"]
