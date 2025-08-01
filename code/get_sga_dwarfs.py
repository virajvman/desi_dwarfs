'''
In this script, we read the catalogs that overlap with SGA and remove objects overlap with massive galaxies. And keep those that overlap with dwarf galaxies in the catalog. 

To get a stellar mass in SGA, we need a redshift. Where ever there is a missing redshift, we can update with DESI catalog. Once we update it, if objects still are missing redshifts, then we just assume it is not a dwarf and remove it.

For objects that are dwarfs, it is still not confirmed that they are dwarfs. There are some objects where the photometry seems iffy. 
'''



bgsb_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_bright_filter_zsucc_zrr02_allfracflux_W_SGA.fits")
bgsf_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_bgs_faint_filter_zsucc_zrr03_allfracflux_W_SGA.fits")
elg_cat = Table.read("/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_elg_filter_zsucc_zrr05_allfracflux_W_SGA.fits")
#there are no LOWZ matches here by construction!

tot_cat = vstack([bgsb_cat, bgsf_cat, elg_cat])



