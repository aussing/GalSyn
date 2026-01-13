import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian1DKernel
from photutils.psf.matching import resize_psf
from scipy.integrate import simpson
from scipy import stats
from scipy.interpolate import interp1d
from .imgutils import convert_flux_map

class GalSynMockObservation_imaging:
    """
    A class to simulate observational effects on synthetic galaxy images,
    including PSF convolution, noise injection, RMS image generation,
    and resampling to a desired pixel scale.

    Parameters:
    -----------
    fits_file_path : str
        Path to the input FITS file.
    filters : list
        List of filter names (e.g., ['jwst_nircam_f115w', 'jwst_nircam_f200w']) for which images will be processed.
    psf_paths : dict
        Dictionary where keys are filter names and values are paths to PSF FITS images.
    psf_pixel_scales : dict
        Dictionary where keys are filter names and values are pixel scales of PSF images in arcsec.
    mag_zp : dict
        Dictionary of magnitude zero-points for each filter.
    limiting_magnitude : dict
        Dictionary of limiting magnitudes for each filter.
    snr_limit : dict
        Dictionary of signal-to-noise ratios at the limiting magnitude for each filter.
    aperture_radius_arcsec : dict
        Dictionary of radii for the circular aperture (in arcsec) used in measuring the magnitude limit for each filter.
    exposure_time : dict
        Dictionary of exposure times in seconds for each filter.
    filter_transmission_path : dict
        Dictionary of paths to text files containing the transmission function for filters.
    desired_pixel_scales : dict
        Dictionary where keys are filter names and values are the desired final pixel
        scales in arcsec for the resampled images.

    """

    def __init__(self, fits_file_path, filters, psf_paths, psf_pixel_scales, mag_zp, limiting_magnitude, snr_limit,
                 aperture_radius_arcsec, exposure_time, filter_transmission_path, desired_pixel_scales):
        """
        Initializes the GalSynMockObservation_imaging with input parameters.

        Parameters:
        -----------
        fits_file_path : str
            Path to the input FITS file.
        filters : list
            List of filter names (e.g., ['jwst_nircam_f115w', 'jwst_nircam_f200w']) for which images will be processed.
        psf_paths : dict
            Dictionary where keys are filter names and values are paths to PSF FITS images.
        psf_pixel_scales : dict
            Dictionary where keys are filter names and values are pixel scales of PSF images in arcsec.
        mag_zp : dict
            Dictionary of magnitude zero-points for each filter.
        limiting_magnitude : dict
            Dictionary of limiting magnitudes for each filter.
        snr_limit : dict
            Dictionary of signal-to-noise ratios at the limiting magnitude for each filter.
        aperture_radius_arcsec : dict
            Dictionary of radii for the circular aperture (in arcsec) used in measuring the magnitude limit for each filter.
        exposure_time : dict
            Dictionary of exposure times in seconds for each filter.
        filter_transmission_path : dict
            Dictionary of paths to text files containing the transmission function for filters.
        desired_pixel_scales : dict
            Dictionary where keys are filter names and values are the desired final pixel
            scales in arcsec for the resampled images.
        """

        self.fits_file_path = fits_file_path
        self.filters = filters
        self.psf_paths = psf_paths
        self.psf_pixel_scales = psf_pixel_scales
        self.mag_zp = mag_zp
        self.limiting_magnitude = limiting_magnitude
        self.snr_limit = snr_limit
        self.aperture_radius_arcsec = aperture_radius_arcsec
        self.exposure_time = exposure_time
        self.filter_transmission_path = filter_transmission_path
        self.desired_pixel_scales = desired_pixel_scales

        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        self.initial_pixel_scale_arcsec = self.image_header['PIXSIZE']
        self.original_flux_unit = self.image_header['BUNIT']

        self.sci_images = {} 
        self.rms_images = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.hdul.close()

    def _rebin_map_flux(self, data, initial_scale, final_scale):
        """
        Rebins a 2D array by an integer or fractional factor while conserving flux.
        Uses bilinear interpolation for fractional factors with empirical 
        flux normalization.
        """
        if np.isclose(initial_scale, final_scale):
            return data
        
        factor = final_scale / initial_scale
        
        # If integer factor, use optimized reshaping
        if np.isclose(factor % 1, 0):
            f = int(np.round(factor))
            y, x = data.shape
            new_y, new_x = y // f, x // f
            reshaped = data[:new_y*f, :new_x*f].reshape(new_y, f, new_x, f)
            return reshaped.sum(axis=(1, 3))
        else:
            # Fractional rebinning using bilinear interpolation
            from scipy.ndimage import zoom
            # Use order=1 for smoother sub-pixel area weighting
            resampled = zoom(data, 1/factor, order=1, prefilter=False)
            
            # Apply theoretical area scaling
            resampled *= (factor**2)
            
            # Enforce exact empirical flux conservation
            sum_in = np.nansum(data)
            sum_out = np.nansum(resampled)
            if sum_out != 0 and not np.isnan(sum_out):
                resampled *= (sum_in / sum_out)
                
            return resampled

    def _get_flux_data(self, filter_name, dust_attenuation=True):
        """
        Retrieves flux data for a given filter from the FITS file.
        This function converts the stored flux (in its original `self.original_flux_unit`)
        back to `erg/s/cm^2/Angstrom` for internal calculations.

        IMPORTANT: If the original FITS unit is 'MJy/sr', this function returns
        spectral flux per pixel (i.e., erg/s/cm^2/Angstrom/pixel).
        If the original FITS unit is 'nJy' or 'AB magnitude', this function returns
        spectral flux density per unit wavelength (erg/s/cm^2/Angstrom).
        The interpretation as 'per pixel' or 'per unit area' is crucial for later steps.
        For resampling, the data needs to be explicitly converted to a surface brightness
        unit if it's not already.
        """
        ext_name = f"{'DUST_' if dust_attenuation else 'NODUST_'}{filter_name.upper()}"
        flux_data = self.hdul[ext_name].data

        _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
        wave_eff = filter_wave_pivot_data[filter_name]

        c_angstrom_s = 2.99792458e18     # speed of light in Angstrom/s

        if self.original_flux_unit == 'erg/s/cm2/A':
            return flux_data
        elif self.original_flux_unit == 'nJy':
            return (flux_data / 1e9 * 1e-23) * c_angstrom_s / wave_eff**2
        elif self.original_flux_unit == 'AB magnitude':
            f_nu = 10**((flux_data + 48.6)/(-2.5))
            return f_nu * c_angstrom_s / wave_eff**2
        elif self.original_flux_unit == 'MJy/sr':
            pixel_area_sr = (self.initial_pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2
            f_nu_per_pixel = flux_data * pixel_area_sr * 1e6
            return (f_nu_per_pixel * 1e-23) * c_angstrom_s / wave_eff**2
        else:
            raise ValueError(f"Unsupported unit: {self.original_flux_unit}")

    def _load_filter_transmission_from_paths_local(self, filters_list, filter_transmission_path_dict):
        filter_transmission_data = {}
        filter_wave_pivot_data = {}
        for f_name in filters_list:
            data = np.loadtxt(filter_transmission_path_dict[f_name])
            wave, trans = data[:, 0], data[:, 1]
            filter_transmission_data[f_name] = {'wave': wave, 'trans': trans}
            num = simpson(wave * trans, wave)
            den = simpson(trans / np.where(wave != 0, wave, 1e-10), wave)
            filter_wave_pivot_data[f_name] = np.sqrt(num / den) if den > 0 else np.nan
        return filter_transmission_data, filter_wave_pivot_data

    def process_images(self, dust_attenuation=None, apply_noise_to_image=True):
        print("\nStarting full image processing pipeline...")
        process_types = [True, False] if dust_attenuation is None else [dust_attenuation]
        
        for current_dust_attenuation in process_types:
            for f_name in self.filters:
                print(f"Processing Filter: {f_name}")

                # 1. Load and convert to erg/s/cm2/A per pixel
                flux_per_pixel = self._get_flux_data(f_name, current_dust_attenuation)

                # 2. Resample (Flux Conserving)
                desired_scale = self.desired_pixel_scales[f_name]
                resampled_flux = self._rebin_map_flux(flux_per_pixel, self.initial_pixel_scale_arcsec, desired_scale)

                # 3. PSF Convolution (at new pixel scale)
                psf_path = self.psf_paths[f_name]
                with fits.open(psf_path) as ph:
                    psf_data = ph[0].data
                
                # Ensure PSF pixel size matches image pixel size before convolution
                psf_resampled = resize_psf(psf_data, self.psf_pixel_scales[f_name], desired_scale)
                psf_resampled /= np.sum(psf_resampled)
                convolved_flux = convolve_fft(resampled_flux, psf_resampled, boundary='fill', fill_value=0.0)

                # 4. Noise Injection (at final resolution)
                _, filter_wave_pivot_data = self._load_filter_transmission_from_paths_local(self.filters, self.filter_transmission_path)
                wave_eff = filter_wave_pivot_data[f_name]
                c_angstrom_s = 2.99792458e18
                
                mag_zp = self.mag_zp[f_name]
                lim_mag = self.limiting_magnitude[f_name]
                snr_lim = self.snr_limit[f_name]
                exp_time = self.exposure_time[f_name]
                ap_rad = self.aperture_radius_arcsec[f_name]

                # Background noise calc
                C_ap = exp_time * (10**(0.4 * (mag_zp - lim_mag)))
                sigma_bg_ap_sq = np.maximum(0, (C_ap / snr_lim)**2 - C_ap)
                ap_area_pix2 = np.pi * (ap_rad / desired_scale)**2
                sigma_bg_per_pixel = np.sqrt(sigma_bg_ap_sq / ap_area_pix2)

                # Counts conversion
                f_nu_pixel = convolved_flux * wave_eff**2 / c_angstrom_s
                pix_mag_AB = -2.5 * np.log10(np.clip(f_nu_pixel, 1e-50, None)) - 48.6
                counts_expected = np.maximum(0, exp_time * (10**(0.4 * (mag_zp - pix_mag_AB))))

                flux_per_count = (10**((mag_zp - 2.5 * np.log10(1.0/exp_time) + 48.6)/(-2.5))) * c_angstrom_s / wave_eff**2

                if apply_noise_to_image:
                    noisy_counts = stats.poisson.rvs(counts_expected) + np.random.normal(0, sigma_bg_per_pixel, size=convolved_flux.shape)
                    final_flux_erg = noisy_counts * flux_per_count
                else:
                    final_flux_erg = convolved_flux

                final_rms_erg = np.sqrt(counts_expected + sigma_bg_per_pixel**2) * flux_per_count

                # 5. Convert back to original units
                key = f"{f_name}_{'dust' if current_dust_attenuation else 'nodust'}"
                self.sci_images[key] = convert_flux_map(final_flux_erg, wave_eff, to_unit=self.original_flux_unit, pixel_scale_arcsec=desired_scale)
                self.rms_images[f"{key}_rms"] = convert_flux_map(final_rms_erg, wave_eff, to_unit=self.original_flux_unit, pixel_scale_arcsec=desired_scale)

    def save_results_to_fits(self, output_fits_path):
        """
        Saves all processed (noise-injected and resampled) images and RMS images
        to a new FITS file, maintaining the original flux units.
        """
        hdul_out = fits.HDUList()
        prihdr = fits.Header()
        prihdr['BUNIT'] = self.original_flux_unit
        hdul_out.append(fits.PrimaryHDU(header=prihdr))

        for key, img_data in self.sci_images.items():
            ext_hdr = fits.Header()
            parts = key.rsplit('_', 1)
            ext_hdr['EXTNAME'] = f"SCI_{parts[1].upper()}_{parts[0].upper()}"
            ext_hdr['BUNIT'] = self.original_flux_unit
            hdul_out.append(fits.ImageHDU(data=img_data, header=ext_hdr))

        for key, rms_data in self.rms_images.items():
            ext_hdr = fits.Header()
            base_key = key.rsplit('_', 1)[0]
            parts = base_key.rsplit('_', 1)
            ext_hdr['EXTNAME'] = f"RMS_{parts[1].upper()}_{parts[0].upper()}"
            ext_hdr['BUNIT'] = self.original_flux_unit
            hdul_out.append(fits.ImageHDU(data=rms_data, header=ext_hdr))

        hdul_out.writeto(output_fits_path, overwrite=True)
        hdul_out.close()
        print(f"Results saved to {output_fits_path}")


class GalSynMockObservation_ifu:
    """
    A class to simulate observational effects on synthetic IFU data cubes,
    including wavelength cutting and regridding, spatial resampling, spectral smoothing, spatial PSF convolution,
    noise injection, and RMS data cube generation.

    Parameters:
    -----------
    fits_file_path : str
        Path to the FITS file output from galsyn_run_fsps (containing OBS_SPEC_NODUST/DUST and WAVELENGTH_GRID).
    desired_wave_grid : np.ndarray
        1D numpy array of the desired final wavelength grid (in Angstrom) for the data cube.
    psf_cube_path : str
        Path to the FITS file containing the 3D PSF data cube (wavelength, y, x).
        The wavelength axis of the PSF cube must match `desired_wave_grid`.
    psf_pixel_scale : float
        Pixel scale of the PSF cube in arcsec.
    spectral_resolution_R : float
        Desired constant spectral resolution (R = lambda / d_lambda) for the output cube.
    mag_zp : float or callable
        Magnitude zero-point. Can be a constant float or a function of wavelength (in Angstrom).
    limiting_magnitude_wave_func : callable
        A function that takes wavelength (in Angstrom) as input and returns
        the limiting magnitude at that wavelength. This magnitude is assumed
        to be per area of the `final_pixel_scale_arcsec`.
    snr_limit : float or callable
        Signal-to-noise ratio corresponding to the limiting magnitude. Can be a constant float or a function of wavelength.
    final_pixel_scale_arcsec : float
        Desired final spatial pixel size in arcsec for the output data cube.
    exposure_time : float or callable
        Exposure time in seconds. Can be a constant float or a function of wavelength.

    """

    def __init__(self, fits_file_path, desired_wave_grid, psf_cube_path, psf_pixel_scale,
                 spectral_resolution_R, mag_zp, limiting_magnitude_wave_func, snr_limit,
                 final_pixel_scale_arcsec, exposure_time):
        """
        Initializes the GalSynMockObservation_ifu with input parameters.

        Parameters:
        -----------
        fits_file_path : str
            Path to the FITS file output from galsyn_run_fsps (containing OBS_SPEC_NODUST/DUST and WAVELENGTH_GRID).
        desired_wave_grid : np.ndarray
            1D numpy array of the desired final wavelength grid (in Angstrom) for the data cube.
        psf_cube_path : str
            Path to the FITS file containing the 3D PSF data cube (wavelength, y, x).
            The wavelength axis of the PSF cube must match `desired_wave_grid`.
        psf_pixel_scale : float
            Pixel scale of the PSF cube in arcsec.
        spectral_resolution_R : float
            Desired constant spectral resolution (R = lambda / d_lambda) for the output cube.
        mag_zp : float or callable
            Magnitude zero-point. Can be a constant float or a function of wavelength (in Angstrom).
        limiting_magnitude_wave_func : callable
            A function that takes wavelength (in Angstrom) as input and returns
            the limiting magnitude at that wavelength. This magnitude is assumed
            to be per area of the `final_pixel_scale_arcsec`.
        snr_limit : float or callable
            Signal-to-noise ratio corresponding to the limiting magnitude. Can be a constant float or a function of wavelength.
        final_pixel_scale_arcsec : float
            Desired final spatial pixel size in arcsec for the output data cube.
        exposure_time : float or callable
            Exposure time in seconds. Can be a constant float or a function of wavelength.
        """

        self.hdul = fits.open(fits_file_path)
        self.image_header = self.hdul[0].header
        self.initial_pixel_scale_arcsec = self.image_header['PIXSIZE']
        self.original_flux_unit = self.image_header.get('BUNIT', 'erg/s/cm2/A')
        self.original_wave_grid = self.hdul['WAVELENGTH_GRID'].data['WAVELENGTH']
        
        self.desired_wave_grid = desired_wave_grid
        self.psf_cube_path = psf_cube_path
        self.psf_pixel_scale = psf_pixel_scale
        self.spectral_resolution_R = spectral_resolution_R
        self.mag_zp, self.limiting_magnitude_wave_func, self.snr_limit = mag_zp, limiting_magnitude_wave_func, snr_limit
        self.final_pixel_scale_arcsec, self.exposure_time = final_pixel_scale_arcsec, exposure_time

        self.initial_datacube_nodust = self.hdul['OBS_SPEC_NODUST'].data if 'OBS_SPEC_NODUST' in self.hdul else None
        self.initial_datacube_dust = self.hdul['OBS_SPEC_DUST'].data if 'OBS_SPEC_DUST' in self.hdul else None
        self.sci_datacubes, self.rms_datacubes = {}, {}

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.hdul.close()

    def _get_wave_dependent_param(self, param, wavelength):
        return param(wavelength) if callable(param) else param

    def _get_flux_data_ifufunc(self, input_datacube):
        """
        Retrieves flux data from the IFU FITS file and converts it to
        erg/s/cm^2/A for internal calculations.

        Parameters:
        -----------
        input_datacube : np.ndarray
            The input data cube to process.
        dust_attenuation : bool
            Indicates whether to process the dust-attenuated or non-attenuated cube.

        Returns:
        --------
        np.ndarray
            The data cube in units of erg/s/cm^2/A.
        """
        if self.original_flux_unit == 'erg/s/cm2/A': 
            return input_datacube
        
        elif self.original_flux_unit == 'MJy/sr':
            c_angstrom_s = 2.99792458e18
            # Calculate pixel area in steradians using the initial pixel scale
            pixel_area_sr = (self.initial_pixel_scale_arcsec * np.pi / (180.0 * 3600.0))**2
            output_cube = np.zeros_like(input_datacube)
            
            for i, wave_eff in enumerate(self.original_wave_grid):
                # 1. Convert MJy/sr to Jy/pixel: (data * 1e6) * pixel_area_sr
                # 2. Convert Jy to erg/s/cm^2/Hz: Jy * 1e-23
                # 3. Convert f_nu to f_lambda: (f_nu * c) / lambda^2
                f_nu_per_pixel = input_datacube[i] * 1e6 * pixel_area_sr
                f_nu_erg_s_cm2_Hz = f_nu_per_pixel * 1e-23
                output_cube[i] = f_nu_erg_s_cm2_Hz * c_angstrom_s / wave_eff**2
            return output_cube
        else: 
            raise ValueError(f"Unsupported unit: {self.original_flux_unit}")

    def process_datacube(self, dust_attenuation=None, apply_noise_to_cube=True):
        print("\nStarting IFU Pipeline: Resample -> Smoothing -> PSF -> Noise")
        process_types = [True, False] if dust_attenuation is None else [dust_attenuation]
        
        for current_dust_attenuation in process_types:
            input_cube_raw = self.initial_datacube_dust if current_dust_attenuation else self.initial_datacube_nodust
            if input_cube_raw is None: 
                continue
            
            print(f"Processing IFU data for dust_attenuation={current_dust_attenuation}")

            # 1. Internal Conversion & Spectral Grid Alignment
            print("  Cutting/interpolating data cube to desired wavelength grid...")
            cube_erg = self._get_flux_data_ifufunc(input_cube_raw)
            reshaped = cube_erg.reshape(cube_erg.shape[0], -1).T
            interp_cube_flat = np.zeros((reshaped.shape[0], len(self.desired_wave_grid)))
            for i in range(reshaped.shape[0]):
                interp_cube_flat[i] = interp1d(self.original_wave_grid, reshaped[i], bounds_error=False, fill_value=0.0)(self.desired_wave_grid)
            cube_wave_cut = interp_cube_flat.T.reshape(len(self.desired_wave_grid), cube_erg.shape[1], cube_erg.shape[2])
            print(f"  Data cube cut to wavelength grid. New shape: {cube_wave_cut.shape}")

            # 2. Spatial Resampling (Flux Conserving)
            #print(f"  Resampling data cube spatially to {self.final_pixel_scale_arcsec:.4f} arcsec...")
            #factor = self.final_pixel_scale_arcsec / self.initial_pixel_scale_arcsec
            #new_ny, new_nx = int(np.round(cube_wave_cut.shape[1] / factor)), int(np.round(cube_wave_cut.shape[2] / factor))
            #resampled_cube = np.zeros((len(self.desired_wave_grid), new_ny, new_nx))
            #from scipy.ndimage import zoom
            #for i in range(len(self.desired_wave_grid)):
            #    resampled_cube[i] = zoom(cube_wave_cut[i], 1/factor, order=0)
            #    if resampled_cube[i].sum() != 0: resampled_cube[i] *= (cube_wave_cut[i].sum() / resampled_cube[i].sum())

            # 2. Spatial Resampling (Flux Conserving)
            print(f"  Resampling data cube spatially to {self.final_pixel_scale_arcsec:.4f} arcsec...")
            factor = self.final_pixel_scale_arcsec / self.initial_pixel_scale_arcsec
            
            from scipy.ndimage import zoom
            # First, calculate a single zoom to get the output shape
            sample_slice = zoom(cube_wave_cut[0], 1/factor, order=1, prefilter=False)
            new_ny, new_nx = sample_slice.shape
            
            resampled_cube = np.zeros((len(self.desired_wave_grid), new_ny, new_nx))
            
            for i in range(len(self.desired_wave_grid)):
                # Resample current slice
                resampled_slice = zoom(cube_wave_cut[i], 1/factor, order=1, prefilter=False)
                
                # Area scaling
                resampled_slice *= (factor**2)
                
                # Empirical normalization per spectral slice
                sum_in = np.nansum(cube_wave_cut[i])
                sum_out = np.nansum(resampled_slice)
                
                if sum_out != 0 and not np.isnan(sum_out):
                    resampled_slice *= (sum_in / sum_out)
                
                resampled_cube[i] = resampled_slice
                

            # 3. Spectral Smoothing
            print(f"  Smoothing spectra to R={self.spectral_resolution_R}...")
            smoothed_cube = np.zeros_like(resampled_cube)
            if len(self.desired_wave_grid) > 1:
                delta_lambda = np.mean(np.diff(self.desired_wave_grid))
                sigma_lambda = self.desired_wave_grid / self.spectral_resolution_R / (2 * np.sqrt(2 * np.log(2)))
                mean_sigma_pix = np.mean(sigma_lambda / delta_lambda)
                if mean_sigma_pix > 0:
                    gauss_kernel = Gaussian1DKernel(stddev=mean_sigma_pix)
                    for iy in range(resampled_cube.shape[1]):
                        for ix in range(resampled_cube.shape[2]):
                            smoothed_cube[:, iy, ix] = convolve_fft(resampled_cube[:, iy, ix], gauss_kernel, boundary='fill', fill_value=0.0)
                else: smoothed_cube = resampled_cube
            else: smoothed_cube = resampled_cube

            # 4. PSF Convolution
            print("  Convolving each spatial slice with PSF...")
            with fits.open(self.psf_cube_path) as ph: psf_cube_data = ph[0].data
            convolved_cube = np.zeros_like(smoothed_cube)
            for i in range(len(self.desired_wave_grid)):
                psf_slice = resize_psf(psf_cube_data[i], self.psf_pixel_scale, self.final_pixel_scale_arcsec)
                psf_slice /= np.sum(psf_slice)
                convolved_cube[i] = convolve_fft(smoothed_cube[i], psf_slice, boundary='fill', fill_value=0.0)

            # 5. Noise Simulation
            print("  Simulating and injecting noise...")
            final_sci, final_rms = np.zeros_like(convolved_cube), np.zeros_like(convolved_cube)
            c_angstrom_s = 2.99792458e18
            for i, wave in enumerate(self.desired_wave_grid):
                mag_zp, lim_mag, snr_lim, exp_time = self._get_wave_dependent_param(self.mag_zp, wave), self.limiting_magnitude_wave_func(wave), self._get_wave_dependent_param(self.snr_limit, wave), self._get_wave_dependent_param(self.exposure_time, wave)
                counts_expected = np.maximum(0, exp_time * (10**(0.4 * (mag_zp - (-2.5 * np.log10(np.clip(convolved_cube[i] * wave**2 / c_angstrom_s, 1e-50, None)) - 48.6)))))
                sigma_bg_sq = np.maximum(0, (exp_time * (10**(0.4 * (mag_zp - lim_mag))) / snr_lim)**2 - exp_time * (10**(0.4 * (mag_zp - lim_mag)))) / (factor**2)
                flux_per_count = (10**((mag_zp - 2.5 * np.log10(1.0/exp_time) + 48.6)/(-2.5))) * c_angstrom_s / wave**2
                if apply_noise_to_cube: final_sci[i] = (stats.poisson.rvs(counts_expected) + np.random.normal(0, np.sqrt(sigma_bg_sq), size=counts_expected.shape)) * flux_per_count
                else: final_sci[i] = convolved_cube[i]
                final_rms[i] = np.sqrt(counts_expected + sigma_bg_sq) * flux_per_count

            # 6. Reconvert Units
            print(f"  Converting final data cubes back to original units ({self.original_flux_unit})...")
            key = 'dust' if current_dust_attenuation else 'nodust'
            self.sci_datacubes[key], self.rms_datacubes[key] = np.zeros_like(final_sci), np.zeros_like(final_rms)
            for i, wave in enumerate(self.desired_wave_grid):
                self.sci_datacubes[key][i] = convert_flux_map(final_sci[i], wave, to_unit=self.original_flux_unit, pixel_scale_arcsec=self.final_pixel_scale_arcsec)
                self.rms_datacubes[key][i] = convert_flux_map(final_rms[i], wave, to_unit=self.original_flux_unit, pixel_scale_arcsec=self.final_pixel_scale_arcsec)

    def save_results_to_fits(self, output_fits_path):
        """
        Saves all processed data cubes and the RMS data cubes
        from `self.sci_datacubes` and `self.rms_datacubes` to a new FITS file.
        The data is saved in flux per pixel units (erg/s/cm2/A/pixel).
        """
        """Saves IFU cubes with headers matching the old module's structure."""
        hdul_out = fits.HDUList()
        prihdr = fits.Header()
        prihdr['BUNIT'], prihdr['COMMENT'] = self.original_flux_unit, 'Mock IFU Observation Results'
        prihdr['RES_R'], prihdr['PIXSIZE'] = self.spectral_resolution_R, self.final_pixel_scale_arcsec
        central_wave = self.desired_wave_grid[len(self.desired_wave_grid) // 2]
        prihdr['ZP_MAG'] = (self._get_wave_dependent_param(self.mag_zp, central_wave), 'ZP at central wavelength')
        prihdr['SNR_LIM'] = (self._get_wave_dependent_param(self.snr_limit, central_wave), 'SNR at central wavelength')
        prihdr['MAG_LIM'] = (float(self.limiting_magnitude_wave_func(central_wave)), 'mag limit at central wavelength')
        prihdr['EXP_TIME'] = (self._get_wave_dependent_param(self.exposure_time, central_wave), 'Exposure time (s) at central wavelength')
        hdul_out.append(fits.PrimaryHDU(header=prihdr))

        for key, cube in self.sci_datacubes.items():
            if cube is not None: hdul_out.append(fits.ImageHDU(data=cube, header=self._create_ifu_header(f'SCI_{key.upper()}', cube.shape, self.original_flux_unit)))
        for key, cube in self.rms_datacubes.items():
            if cube is not None: hdul_out.append(fits.ImageHDU(data=cube, header=self._create_ifu_header(f'RMS_{key.upper()}', cube.shape, self.original_flux_unit)))
        
        col = fits.Column(name='WAVELENGTH', format='D', array=self.desired_wave_grid)
        hdul_out.append(fits.BinTableHDU.from_columns([col], name='WAVELENGTH_GRID'))
        hdul_out.writeto(output_fits_path, overwrite=True)
        hdul_out.close()
        print(f"IFU results saved to {output_fits_path}")

    def _create_ifu_header(self, extname, shape, flux_unit):
        """Helper to create a standard header for IFU cube extensions."""
        hdr = fits.Header()
        hdr['EXTNAME'] = extname
        hdr['BUNIT'] = flux_unit
        hdr['CTYPE1'] = 'WAVE'
        hdr['CRPIX1'] = 1.0 
        hdr['CRVAL1'] = self.desired_wave_grid[0]
        hdr['CDELT1'] = (self.desired_wave_grid[1] - self.desired_wave_grid[0]) if len(self.desired_wave_grid) > 1 else 0.0
        hdr['CUNIT1'] = 'Angstrom'
        #hdr['CTYPE2'] = 'DEC--TAN'
        hdr['CRPIX2'] = shape[1] / 2.0 + 0.5 
        hdr['CDELT2'] = self.final_pixel_scale_arcsec
        hdr['CUNIT2'] = 'arcsec'
        #hdr['CTYPE3'] = 'RA---TAN'
        hdr['CRPIX3'] = shape[2] / 2.0 + 0.5 
        hdr['CDELT3'] = self.final_pixel_scale_arcsec
        hdr['CUNIT3'] = 'arcsec'
        return hdr