"""
scarlet_photo -- consolidated SCARLET photometry pipeline for DESI dwarfs.

Deliberately THIN: importing this package must NOT pull in scarlet/autograd/
proxmin or any heavy fitter module, so that the stage-2 store + consolidator
(`bundle_store`, `consolidate`) can be imported standalone inside the NERSC
container (numpy/h5py only). Only `config` is re-exported here.

Two execution environments:
  * stage 1 (fit): `inputs`, `detect`, `fit`, `grouping`, `photometry`,
    `fragment`, `pipeline`, `driver` -- need the full scarlet stack.
  * stage 2 (consolidate): `bundle_store`, `consolidate` -- numpy/h5py only,
    container-safe (h5py/astropy imported inside functions).
"""

from .config import ScarletConfig, BANDS, PIXSCALE

__all__ = ["ScarletConfig", "BANDS", "PIXSCALE"]
