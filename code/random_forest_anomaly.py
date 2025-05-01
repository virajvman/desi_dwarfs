'''
In this script, we apply the RF based algorithm from http://wise-obs.tau.ac.il/~dovip/weird-galaxies/about.html. This is an interesting comparison point to the NMF+PCA based anomaly detection. Those are good for visualizing the full spread of the galaxy, but not particulary good at identifying anomalies!
'''


if __name__ == '__main__':

    overwrite_templates = False
    
    rng = np.random.default_rng(42)

    ##################
    ##PART 1: Data preparation: Load,download the DESI spectra. Resample it!
    ##################
    # if overwrite_templates:
    print_stage("Loading the DESI spectra")

    save_dered = "/pscratch/sd/v/virajvm/catalog_dr1_dwarfs/iron_spectra/spectra_files/desi_y1_dwarf_clean_deredshift.h5"

    with h5py.File(save_dered, "r") as f:
        all_tgids = f["TARGETID"][:]
        all_zreds = f["Z"][:]
        wave_rest = f["WAVE_REST"][:]
        all_fluxs_out = f["FLUX"][:]
        all_flux_ivars_out = f["FLUX_IVAR"][:]


    print("wave rest shape", wave_rest.shape)
    print("flux out shape", all_fluxs_out.shape)
    print("flux_ivar out shape", all_flux_ivars_out.shape)
    print("zreds out shape", all_zreds.shape)

    #only select a handful for galaxies for initial testing purposes

    #translate them into the appropriate shape and to float32 for memory usage
    all_fluxs_out = all_fluxs_out.T.astype(np.float32)
    all_flux_ivars_out = all_flux_ivars_out.T.astype(np.float32)
    wave_rest = wave_rest.astype(np.float32)


    ##################
    ##PART 1: Constructing the shuffled spectra matrix!!!
    ##################
    
        