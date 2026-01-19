Generating SSP Libraries
=========================

This page demonstrates how to create custom Simple Stellar Population (SSP) spectral grids using GalSyn. These grids are the foundational building blocks for synthesizing galaxy light from hydrodynamical simulations.

As detailed in Abdurro'uf et al. (2026), the package offers two modes for spectral assignment:

    * On-the-fly generation: Calculates spectra for each particle during the synthesis process. This provides maximum precision but is computationally expensive.
    * Pre-computed grids (Recommended): Uses a multi-dimensional HDF5 grid of spectra across a range of stellar ages and metallicities. This method significantly improves computational efficiency by utilizing fast interpolation.

In GalSyn, there are two widely-used Stellar Population Synthesis (SPS) codes:

    * **FSPS**: Offers extensive control over stellar isochrones (e.g., MIST, Padova), spectral libraries (e.g., MILES, BaSeL), and various IMFs.
    * **Bagpipes**: Utilizes the Bruzual & Charlot (2003) models with a fixed Kroupa (2001) IMF.


Here is an example script for generating SSP grids using FSPS:

.. code-block:: python

    import numpy as np
    from galsyn.ssp_generator_fsps import generate_ssp_grid

    # Output file name
    ssp_grid_fsps = 'ssp_fsps.hdf5'    # output file name

    # Define the age and metallicity grids
    # Logarithmic grid from 1 Myr to 13.8 Gyr (approx age of universe)
    ages_gyr = np.logspace(np.log10(0.001), np.log10(13.8), 300)

    # Linear grid for log10(Z/Z_sun)
    logzsol_grid = np.linspace(-2.0, 0.2, 300)

    imf_type = 1  # Chabrier (2003) IMF
    n_jobs = 5    # number of CPU cores to be used for the calculations
    generate_ssp_grid(output_filename=ssp_grid_fsps,
                          ages_gyr=ages_gyr,
                          logzsol_grid=logzsol_grid,
                          imf_type=imf_type,
                          overwrite=True,
                          n_jobs=n_jobs,
                          rest_wave_min=500,
                          rest_wave_max=30000)


And here is an example script for Bagpipes SSP grids:

.. code-block:: python

    from galsyn.ssp_generator_bagpipes import generate_ssp_grid_bagpipes

    # Output file name for Bagpipes-based library
    ssp_grid_bagpipes = 'ssp_bagpipes.hdf5'

    # Define the age and metallicity grids
    # Using similar resolution to your FSPS setup for consistency
    ages_gyr = np.logspace(np.log10(0.001), np.log10(13.8), 300)
    logzsol_grid = np.linspace(-2.0, 0.2, 300)

    # Ionization parameter for nebular emission modeling
    gas_logu = -2.0 

    # Number of CPU cores to use
    n_jobs = 5

    # Generate the grid
    # Note: Bagpipes backend in GalSyn is optimized for BC03 models 
    # and a Kroupa (2001) IMF by default.
    generate_ssp_grid_bagpipes(
        output_filename=ssp_grid_bagpipes,
        ages_gyr=ages_gyr,
        logzsol_grid=logzsol_grid,
        gas_logu=gas_logu,
        overwrite=True,
        n_jobs=n_jobs
    )