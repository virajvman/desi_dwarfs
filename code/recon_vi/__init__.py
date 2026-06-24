"""recon_vi -- reconstructed-cube visual-inspection & fine-tuning GUI.

Two halves:

* ``build_bundle`` -- run on NERSC; packs a hand-picked VI catalog + the
  per-object reconstruction artifacts into a single HDF5 bundle to ``scp`` down.
* ``server`` -- run locally; a thin Flask backend that serves one object at a
  time to the static JS/canvas frontend in ``static/`` and autosaves a
  replayable decision CSV.

See ``README.md`` for the end-to-end workflow.
"""

__all__ = ["build_bundle", "server"]
