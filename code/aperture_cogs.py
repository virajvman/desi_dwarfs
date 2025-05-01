'''
In this script, we will do a curve of growth analysis on our objects that are really shreds!

We are working in a different script here as we need the tractor/astrometry packages to construct the psf model!

Basic steps are following what SGA catalog did:
1) Identify the range of apertures within which we will do our photometry
2) Mask relevant pixels (if star or residuals after subtracting model image are very large?)
3) If we have identified sources within aperture that we want to subtract, we can create an image with all the masked pixels and subtracted sources for reference?

'''



        ###
        #curve of growth analysis
        #TODO: add the fitting function to get asymptotic magnitude
        #TODO: why does z-band magnitude dip down some times?
        #TODO: I am not masking enough pixels of the background sources I think ... do this better ...

        #The COG analysis relies on subtracting tractor models from the outskirts
        #So we will do this in a different script for clarity and ability to working the tractor package 
        #we will save all the relevant files so we can directly just load them!
        ###

        ##Instead of cog, include the 
        
        # radii = np.linspace(2.25,4.75,10)

        # tot_subtract_sources = { "g": tot_subtract_sources_g, "r": tot_subtract_sources_r, "z": tot_subtract_sources_z  }
                
        # cog_mags = {"g":[], "r": [], "z": []}
        
        # for radius in radii:
        #     aperture_for_phot_i = get_elliptical_aperture( segment_map_v2, star_mask, 2, sigma = radius )

        #     ## let us plot all these apertures for reference
        #     aperture_for_phot_i.plot(ax = ax[1], color = "r", lw = 1, ls = "dotted")
            
        #     for bi in "grz":
        #         phot_table_i = aperture_photometry(data[bi] , aperture_for_phot_i, mask = ~aperture_mask.astype(bool))
        #         new_mag_i = 22.5 - 2.5*np.log10( phot_table_i["aperture_sum"].data[0] - tot_subtract_sources[bi] )
        #         cog_mags[bi].append(new_mag_i)


        # all_cogs = np.concatenate( (cog_mags["g"],cog_mags["r"],cog_mags["z"]) )

    
        # all_cogs = all_cogs[ ~np.isinf(all_cogs) & ~np.isnan(all_cogs)]
        
        # if len(all_cogs) > 0:
        #     ax[2].scatter(radii, cog_mags["g"],color = "mediumblue")
        #     ax[2].scatter(radii, cog_mags["r"],color = "forestgreen")
        #     ax[2].scatter(radii, cog_mags["z"],color = "firebrick")
        #     ax[2].set_ylabel(r"$m(<r)$ mag",fontsize = 14)
        #     ax[2].set_xlim([2, 5.5])
        #     ax[2].vlines(x = 3.5, ymin=np.min(all_cogs) - 0.25, ymax = np.max(all_cogs) + 0.25, color = "k",ls = "dotted")
        #     ax[2].set_ylim([ np.max(all_cogs) + 0.25,np.min(all_cogs) - 0.25  ] )
