"""This is a script that defines the core Event Table class"""


from load_event_table import *
from scipy.stats import wasserstein_distance
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import cmdstanpy
from fit_gmm import fit_gmm
from scipy.stats import norm
cmdstanpy.set_cmdstan_path('/Users/glebiagelskii/.cmdstan/cmdstan-2.37.0')
class Event_Table():
    meta_calibration_parameters = {'Refeyn OneMP':{'Medium':[3.34E-5, 0 ], 'Full':[2.8E-5, 0 ], 'Small':[2.8E-5, 0 ], 'Regular':[2.8E-5, 0 ]}, 'Refeyn TwoMP':{'Medium':[4E-5, 10 ], 'Full':[3.7E-5,10], 'Small':[3.7E-5, 10 ], 'Regular':[3.7E-5, 10 ]}}
    MS1000_guesses = {'means':[90, 180, 360], 'sigmas':[12, 8, 12]}
    def __init__(self, frame,	y_det,	x_det,	contrasts_det,	contrasts,	x_fit,	y_fit,	contrasts_se,	r2_fit,	res_fit,x,	y,	masses_kDa, calibration_required, instrument_name, ROI_mode, calibration_results=None, calibrated=0, gradient_kDa_per_contrast=None, intercept_kDa_per_contrast=None, r2=None, calibrated_peaks=None, calibrated_core_sigma=None):
        self.frame = frame
        self.y_det = y_det
        self.x_det = x_det
        self.contrasts_det = contrasts_det
        self.contrasts = contrasts
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.contrasts_se = contrasts_se
        self.r2_fit = r2_fit
        self.res_fit = res_fit
        self.x = x
        self.y = y
        self.masses_kDa = masses_kDa
        self.fit_required = calibration_required 
        self.instrument_name = instrument_name
        self.ROI_mode = ROI_mode

        self.calibration_results = calibration_results
        self.calibrated = calibrated
        self.gradient_kDa_per_contrast = gradient_kDa_per_contrast
        self.intercept_kDa_per_contrast = intercept_kDa_per_contrast
        self.r2 = r2
        self.calibrated_peaks = calibrated_peaks
        self.calibrated_core_sigmas = calibrated_core_sigma
    
    
    @classmethod

    def from_path(cls, file_path: str, ROI: str, Instrument:str, needs_calibration: bool):

        df = load_event_table(Path(file_path))

        if table_header_format_is_correct(df):
            pass
        else:
            raise ValueError('Review Header')
        
        if Instrument == 'Refeyn OneMP' or Instrument=='Refeyn TwoMP':
            pass
        else:
            raise ValueError(f'Intrument name cannot be {Instrument}, only Refeyn OneMP or Refeyn TwoMP')
        if ROI=='Regular' or ROI=='Full' or ROI=='Small' or ROI=='Medium':
            pass
        else:
            raise ValueError(f'ROI name cannot be {ROI}, only Medium, Small, Full or Regular')
            
        return cls(
            frame=df['frame'].values,
            y_det=df['y'].values,
            x_det=df['x'].values,
            contrasts_det=df['contrasts_det'].values if 'contrasts_det' in df else None,
            contrasts=df['contrasts'].values,
            x_fit=df['x_fit'].values,
            y_fit=df['y_fit'].values,
            contrasts_se=df['contrasts_se'].values,
            r2_fit=df['r2_fit'].values,
            res_fit=df['res_fit'].values,
            x=df['x'].values,
            y=df['y'].values,
            masses_kDa=df['masses_kDa'].values if 'masses_kDa' in df else None,
            calibration_required=needs_calibration,
            instrument_name=Instrument,
            ROI_mode=ROI,
            calibration_results=None,
            calibrated=0,
            gradient_kDa_per_contrast=None,
            intercept_kDa_per_contrast=None,
            r2=None, calibrated_peaks=None,
            calibrated_core_sigma=None
        )
       

    def calibrate(self, select_by_metric:str, reject_great_residuals:bool=False, r2_PSF_fit: float =0.9):
        best_aic = 1000
        best_bic = 1000
        best_score = 1000

        #Note the current structure is prep work for when potential future iterations will attempt fits with vairable numbers of peak components

        if select_by_metric=='aic':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            if reject_great_residuals==False:
                results = fit_gmm(self.contrasts, metaparams, guesses)
                print(f"The Aikake information criterion for the GMM fit is {results['metrics']['aic']}")
            else:
                contrasts = self.contrasts[self.r2_fit>r2_PSF_fit]
                results = fit_gmm(contrasts, metaparams, guesses)
                print(f"The Aikake information criterion for the GMM fit is {results['metrics']['aic']}")
        elif select_by_metric=='bic':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            results = fit_gmm(self.contrasts, metaparams, guesses)
            if reject_great_residuals==False:
                results = fit_gmm(self.contrasts, metaparams, guesses)
                print(f"The Bayesian information criterion for the GMM fit is {results['metrics']['bic']}")
            else:
                contrasts = self.contrasts[self.r2_fit>r2_PSF_fit]
                results = fit_gmm(contrasts, metaparams, guesses)
                print(f"The Bayesian information criterion for the GMM fit is {results['metrics']['bic']}")
        elif select_by_metric=='score':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            results = fit_gmm(self.contrasts, metaparams, guesses)
            if reject_great_residuals==False:
                results = fit_gmm(self.contrasts, metaparams, guesses)
                print(f"The Log loss for the GMM fit is {results['metrics']['score']}")
            else:
                contrasts = self.contrasts[self.r2_fit>r2_PSF_fit]
                results = fit_gmm(contrasts, metaparams, guesses)
                print(f"The Log loss for the GMM fit is {results['metrics']['score']}")
        else:
            raise ValueError(f'The selected metric must be either aic, bic or score, not {select_by_metric}')
        
        #Linear Regression
        model = LinearRegression()
        contrast_means = np.array(results['means_in_contrast']).reshape(-1,1)
        model.fit(contrast_means, np.array(Event_Table.MS1000_guesses['means']))
        self.gradient_kDa_per_contrast = model.coef_[0]
        self.intercept_kDa_per_contrast = model.intercept_
        self.r2 = model.score(contrast_means, np.array(Event_Table.MS1000_guesses['means']) )
        self.calibrated_peaks = model.predict(contrast_means)
        self.calibrated_core_sigmas = np.array(results['sigmas_in_contrast'])* self.gradient_kDa_per_contrast + self.intercept_kDa_per_contrast
        #Change the stored values of masses
        self.masses_kDa = self.contrasts*self.gradient_kDa_per_contrast + self.intercept_kDa_per_contrast

        #Store the results of the calibration_results to results
        self.calibration_results = results

        #Change status to calibrated and print a message
        self.calibrated = 1

        print(f'Your event table was succesfully calibrated with an $R^{2}$ value of {self.r2} and calibration peak values of {self.calibrated_peaks} and corresponding sigmas of {self.calibrated_core_sigmas}')
        print(f'Calibration gradient: {self.gradient_kDa_per_contrast} kDa per contrast unit, intercept: {self.intercept_kDa_per_contrast} kDa')




    def save_to_csv(self, output_path: str):
        if self.calibrated !=1:
            raise ValueError('The event table has not been calibrated yet, please calibrate before saving')
        
        #Create a dataframe to save
        df_dict = {'frame':self.frame, 'y_det':self.y_det, 'x_det':self.x_det, 'contrasts_det':self.contrasts_det, 'contrasts':self.contrasts, 'x_fit':self.x_fit, 'y_fit':self.y_fit, 'contrasts_se':self.contrasts_se, 'r2_fit':self.r2_fit, 'res_fit':self.res_fit,  'x':self.x, 'y':self.y, 'masses_kDa':self.masses_kDa, 'fitting_error':self.calibration_results['metrics']['scores_per_sample'], 'r2':self.r2, 'gradient_kDa_per_contrast':self.gradient_kDa_per_contrast, 'intercept_kDa_per_contrast':self.intercept_kDa_per_contrast}
        df = pd.DataFrame(df_dict)

        #Save to csv
        df.to_csv(output_path, index=False)
        print(f'Calibrated event table saved to {output_path}')
    
    def sace_to_parquet(self, output_path: str):
        if self.calibrated !=1:
            raise ValueError('The event table has not been calibrated yet, please calibrate before saving')
        
        #Create a dataframe to save
        df_dict = {'frame':self.frame, 'y_det':self.y_det, 'x_det':self.x_det, 'contrasts_det':self.contrasts_det, 'contrasts':self.contrasts, 'x_fit':self.x_fit, 'y_fit':self.y_fit, 'contrasts_se':self.contrasts_se, 'r2_fit':self.r2_fit, 'res_fit':self.res_fit,  'x':self.x, 'y':self.y, 'masses_kDa':self.masses_kDa, 'fitting_error':self.calibration_results['metrics']['scores_per_sample'], 'r2':self.r2, 'gradient_kDa_per_contrast':self.gradient_kDa_per_contrast, 'intercept_kDa_per_contrast':self.intercept_kDa_per_contrast}
        df = pd.DataFrame(df_dict)

        #Save to parquet
        df.to_parquet(output_path, index=False)
        print(f'Calibrated event table saved to {output_path}')

    def visualise(self, x_min: float=None, x_max: float=None, b:int=200):
        if self.calibrated !=1:
            raise ValueError('The event table has not been calibrated yet, please calibrate before visualising')
        
        plt.figure(figsize=(10,6))
        plt.hist(self.masses_kDa, bins=b, density=True, alpha=0.6, color='g', label='Calibrated Mass Distribution')
        if x_min is not None and x_max is not None:
            x_min = x_min
            x_max = x_max
        else:
            x_min = min(self.masses_kDa)*1.1
            x_max = max(self.masses_kDa)*1.1
        plt.xlim([x_min, x_max])

        #Plot the fitted GMM peaks
        from scipy.stats import norm
        x = np.linspace(x_min, x_max, 1000)
        

        # Normalize/squeeze arrays to 1D
        means = np.ravel(self.calibrated_peaks)
        sigmas = np.ravel(self.calibrated_core_sigmas)
        weights = np.ravel(self.calibration_results.get('weights', np.ones_like(means)))

        # Sanity check: lengths must match
        if not (means.shape == sigmas.shape == weights.shape):
            raise ValueError(f"Mismatch shapes: means {means.shape}, sigmas {sigmas.shape}, weights {weights.shape}")

        for mean, sigma, weight in zip(means, sigmas, weights):
            # make sure they are scalars
            mean = float(np.squeeze(mean))
            sigma = float(np.squeeze(sigma))
            weight = float(np.squeeze(weight))

            if sigma <= 0 or not np.isfinite(sigma):
                # skip or set a small epsilon to avoid division by zero
                sigma = max(sigma, 1e-6)

            y_vals = weight * norm.pdf(x, loc=mean, scale=sigma)
            plt.plot(x, y_vals, linewidth=2, label=f'GMM Peak at {mean:.2f} kDa')
        plt.title('Calibrated Mass Distribution with Fitted GMM Peaks')
        plt.legend()
        plt.xlabel('Mass (kDa)')
    





    def _fit_gmm_with_stan(self,
                        contrasts: np.ndarray,
                        guesses: dict,
                        debug_Bad_Data: bool = False):
        """
        Fit a 1D 3-component Gaussian mixture to `contrasts` using Stan model
        gmm_1D_3p.stan, and return a results dict similar in spirit to fit_gmm.

        Assumptions:
        - guesses is a dict with keys:
            'means_in_contrast': 3 mean guesses in the SAME units as `contrasts`
            'sigmas_in_contrast': 3 sigma guesses in the SAME units as `contrasts`
        - Stan model expects data:
            int N;
            vector[N] y;
            vector[3] mu_guess_z;
            vector[3] sigma_guess_z;
            where *_z are in standardised (z-score) space.
        """

        # --- 0. Check guesses structure ---
        if "means_in_contrast" not in guesses:
            raise ValueError("guesses must contain key 'means_in_contrast' with 3 values.")
        if "sigmas_in_contrast" not in guesses:
            raise ValueError("guesses must contain key 'sigmas_in_contrast' with 3 values.")

        mu_guess_contrast = np.asarray(guesses["means_in_contrast"], dtype=float)
        sigma_guess_contrast = np.asarray(guesses["sigmas_in_contrast"], dtype=float)

        if mu_guess_contrast.size != 3:
            raise ValueError(
                f"'means_in_contrast' must have length 3, got {mu_guess_contrast.size}"
            )
        if sigma_guess_contrast.size != 3:
            raise ValueError(
                f"'sigmas_in_contrast' must have length 3, got {sigma_guess_contrast.size}"
            )
        if np.any(sigma_guess_contrast <= 0):
            raise ValueError("'sigmas_in_contrast' must all be > 0.")

        # --- 1. Clean and inspect data ---
        contrasts = np.asarray(contrasts, dtype=float)
        finite_mask = np.isfinite(contrasts)
        y = contrasts[finite_mask]

        if debug_Bad_Data:
            print(f"Total points: {contrasts.size}, finite: {y.size}")
            print("Any NaN?", np.isnan(contrasts).any())
            print("Any inf?", np.isinf(contrasts).any())
            if y.size > 0:
                print("Finite min, max:", np.nanmin(y), np.nanmax(y))

        if y.size == 0:
            raise ValueError("No finite data points available for Stan GMM fit.")

        # --- 2. Standardise data: z = (y - mean_y) / std_y ---
        mean_y = y.mean()
        std_y = y.std()

        if std_y == 0.0:
            raise ValueError("Standard deviation of contrasts is zero; cannot fit a GMM.")

        y_stdzd = (y - mean_y) / std_y
        N = y_stdzd.shape[0]

        # transform guesses into z-space
        mu_guess_z = (mu_guess_contrast - mean_y) / std_y
        mu_guess_z = np.sort(mu_guess_z)  # Stan has ordered[3] mu

        sigma_guess_z = sigma_guess_contrast / std_y
        if np.any(sigma_guess_z <= 0):
            raise ValueError("sigma_guess_z must all be > 0 after scaling.")

        # --- 3. Compile/load Stan model ---
        gmm_model = cmdstanpy.CmdStanModel(stan_file="gmm_1D_3p.stan")

        # Stan model should declare mu_guess_z and sigma_guess_z in data block
        stan_data = {
            "N": int(N),
            "y": y_stdzd,
            "mu_guess_z": mu_guess_z.astype(float),
            "sigma_guess_z": sigma_guess_z.astype(float),
        }

        if debug_Bad_Data:
            print("stan_data keys:", stan_data.keys())
            print("mu_guess_z:", mu_guess_z)
            print("sigma_guess_z:", sigma_guess_z)

        # --- 4. Run Stan sampling (we can let Stan pick its own inits now) ---
        fit = gmm_model.sample(
            data=stan_data,
            chains=4,
            parallel_chains=4,
            iter_warmup=5000,
            iter_sampling=1500,
            adapt_delta=0.95,
            max_treedepth=11,
            show_console=True
        )
    

        # --- 5. Extract posterior draws in standardised space ---
        mu_draws_z = fit.stan_variable("mu")       # (draws, 3) in z-space
        sigma_draws_z = fit.stan_variable("sigma") # (draws, 3) in z-space
        theta_draws = fit.stan_variable("theta")   # (draws, 3)

        mu_hat_z = mu_draws_z.mean(axis=0)
        sigma_hat_z = sigma_draws_z.mean(axis=0)
        theta_hat = theta_draws.mean(axis=0)

        # --- 6. De-standardise to original contrast units ---
        # z = (y - mean_y) / std_y  =>  y = mean_y + std_y * z
        mu_hat = mean_y + std_y * mu_hat_z
        sigma_hat = std_y * sigma_hat_z
        # theta_hat unchanged

        # --- 7. Fit quality metrics ---
        summ = fit.summary()
        if "R_hat" in summ.columns:
            rhat_max = float(summ["R_hat"].max())
        else:
            rhat_max = float("nan")

        if "N_Eff" in summ.columns:
            ess_min = float(summ["N_Eff"].min())
        else:
            ess_min = float("nan")

        # Mean log-likelihood per point in original units
        const = -0.5 * np.log(2.0 * np.pi)
        log_lik = []
        for val in y:  # finite points used in the fit
            log_norm = const - np.log(sigma_hat) - 0.5 * ((val - mu_hat) / sigma_hat) ** 2
            log_mix = np.log(np.sum(theta_hat * np.exp(log_norm)))
            log_lik.append(log_mix)
        mean_log_lik = float(np.mean(log_lik))

        metrics = {
            "mean_log_lik": mean_log_lik,
            "rhat_max": rhat_max,
            "ess_min": ess_min,
        }

        results = {
            "metrics": metrics,
            "means_in_contrast": mu_hat.tolist(),
            "sigmas_in_contrast": sigma_hat.tolist(),
            "weights": theta_hat.tolist(),
            "stan_fit": fit,
            "standardisation": {
                "mean_y": float(mean_y),
                "std_y": float(std_y),
                "finite_mask": finite_mask,
            },
        }

        return results



    def _fit_gmm_with_stan_1000(self,
                       contrasts: np.ndarray,
                       guesses: dict,
                       debug_Bad_Data: bool = False):
            """
            Fit a 1D 3-component Gaussian mixture to `contrasts` using Stan model
            gmm_1D_3p.stan, and return a results dict similar in spirit to fit_gmm.

            Assumptions:
            - guesses is a dict with keys:
                'means_in_contrast': 3 mean guesses in the SAME units as `contrasts`
                'sigmas_in_contrast': 3 sigma guesses in the SAME units as `contrasts`
            - Stan model expects data:
                int N;
                vector[N] y;
                vector[3] mu_guess_z;
                vector[3] sigma_guess_z;
            where here *_z are just 'scaled by 1000' versions of the contrast units,
            not actual z-scores.
            """

            # --- 0. Check guesses structure ---
            if "means_in_contrast" not in guesses:
                raise ValueError("guesses must contain key 'means_in_contrast' with 3 values.")
            if "sigmas_in_contrast" not in guesses:
                raise ValueError("guesses must contain key 'sigmas_in_contrast' with 3 values.")

            mu_guess_contrast = np.asarray(guesses["means_in_contrast"], dtype=float)
            sigma_guess_contrast = np.asarray(guesses["sigmas_in_contrast"], dtype=float)

            if mu_guess_contrast.size != 3:
                raise ValueError(
                    f"'means_in_contrast' must have length 3, got {mu_guess_contrast.size}"
                )
            if sigma_guess_contrast.size != 3:
                raise ValueError(
                    f"'sigmas_in_contrast' must have length 3, got {sigma_guess_contrast.size}"
                )
            if np.any(sigma_guess_contrast <= 0):
                raise ValueError("'sigmas_in_contrast' must all be > 0.")

            # --- 1. Clean and inspect data ---
            contrasts = np.asarray(contrasts, dtype=float)
            finite_mask = np.isfinite(contrasts)
            y = contrasts[finite_mask]

            if debug_Bad_Data:
                print(f"Total points: {contrasts.size}, finite: {y.size}")
                print("Any NaN?", np.isnan(contrasts).any())
                print("Any inf?", np.isinf(contrasts).any())
                if y.size > 0:
                    print("Finite min, max:", np.nanmin(y), np.nanmax(y))

            if y.size == 0:
                raise ValueError("No finite data points available for Stan GMM fit.")

            # --- 2. Simple scaling: y_scaled = y * scale_factor ---
            scale_factor = 1000.0

            y_scaled = y * scale_factor
            N = y_scaled.shape[0]

            # transform guesses into the same scaled space
            mu_guess_scaled = mu_guess_contrast * scale_factor
            sigma_guess_scaled = sigma_guess_contrast * scale_factor

            # keep ordering for Stan's ordered[3] mu if needed
            mu_guess_scaled = np.sort(mu_guess_scaled)

            # --- 3. Compile/load Stan model ---
            gmm_model = cmdstanpy.CmdStanModel(stan_file="gmm_1D_3p_1000.stan")

            # Stan model should declare mu_guess_z and sigma_guess_z in data block
            # (we're just reusing those names for the scaled versions)
            stan_data = {
                "N": int(N),
                "y": y_scaled,
                "mu_guess_z": mu_guess_scaled.astype(float),
                "sigma_guess_z": sigma_guess_scaled.astype(float),
            }

            if debug_Bad_Data:
                print("stan_data keys:", stan_data.keys())
                print("mu_guess_scaled:", mu_guess_scaled)
                print("sigma_guess_scaled:", sigma_guess_scaled)

            # --- 4. Run Stan sampling (we can let Stan pick its own inits now) ---
            fit = gmm_model.sample(
                data=stan_data,
                chains=4,
                parallel_chains=4,
                iter_warmup=5000,
                iter_sampling=1500,
                adapt_delta=0.95,
                max_treedepth=11,
                show_console=True,
            )

            # --- 5. Extract posterior draws in scaled space ---
            mu_draws_scaled = fit.stan_variable("mu")       # (draws, 3) in scaled space
            sigma_draws_scaled = fit.stan_variable("sigma") # (draws, 3) in scaled space
            theta_draws = fit.stan_variable("theta")        # (draws, 3)

            mu_hat_scaled = mu_draws_scaled.mean(axis=0)
            sigma_hat_scaled = sigma_draws_scaled.mean(axis=0)
            theta_hat = theta_draws.mean(axis=0)

            # --- 6. De-scale back to original contrast units ---
            # scaled_y = y * scale_factor  =>  y = scaled_y / scale_factor
            mu_hat = mu_hat_scaled / scale_factor
            sigma_hat = sigma_hat_scaled / scale_factor
            # theta_hat unchanged

            # --- 7. Fit quality metrics ---
            summ = fit.summary()
            if "R_hat" in summ.columns:
                rhat_max = float(summ["R_hat"].max())
            else:
                rhat_max = float("nan")

            if "N_Eff" in summ.columns:
                ess_min = float(summ["N_Eff"].min())
            else:
                ess_min = float("nan")

            # Mean log-likelihood per data point in original units
            const = -0.5 * np.log(2.0 * np.pi)
            log_lik = []
            for val in y:  # finite points used in the fit (original scale)
                log_norm = const - np.log(sigma_hat) - 0.5 * ((val - mu_hat) / sigma_hat) ** 2
                log_mix = np.log(np.sum(theta_hat * np.exp(log_norm)))
                log_lik.append(log_mix)
            mean_log_lik = float(np.mean(log_lik))

            metrics = {
                "mean_log_lik": mean_log_lik,
                "rhat_max": rhat_max,
                "ess_min": ess_min,
            }

            results = {
                "metrics": metrics,
                "means_in_contrast": mu_hat.tolist(),
                "sigmas_in_contrast": sigma_hat.tolist(),
                "weights": theta_hat.tolist(),
                "stan_fit": fit,
                "standardisation": {
                    "scale_factor": float(scale_factor),
                    "finite_mask": finite_mask,
                },
            }

            return results
    
    def _fit_gaussians_divide_and_conquer_simple(self,
                                                contrasts: np.ndarray,
                                                guesses: dict,
                                                debug_Bad_Data: bool = False):
        """
        Divide-and-conquer Gaussian fitting using scipy.stats.norm.fit
        instead of Stan.

        Uses three split points to define three peak regions and a valley:
        - Region 0 (peak 1): y < s1
        - Region 1 (peak 2): s1 <= y < s2
        - Valley:            s2 <= y < s3  (ignored)
        - Region 2 (peak 3): y >= s3

        For each peak region, fits a single 1D Gaussian with norm.fit and
        computes residual metrics.

        Assumptions:
        - guesses is a dict with keys:
            'means_in_contrast': array-like, length 3
            'sigmas_in_contrast': array-like, length 3
            'split_points': array-like, length 3, with s1 < s2 < s3 (contrast units)
        - Returns a results dict with:
            'means_in_contrast' : [mu1, mu2, mu3]
            'sigmas_in_contrast': [sigma1, sigma2, sigma3]
            'weights'           : [w1, w2, w3] (fractions of used points)
            'metrics'           : {
                'region_sizes': [N0, N1, N2],
                'valley_size': int,
                'region_residuals': [
                    {'rmse': ..., 'mae': ..., 'n_points': ...},  # region 0
                    {'rmse': ..., 'mae': ..., 'n_points': ...},  # region 1
                    {'rmse': ..., 'mae': ..., 'n_points': ...},  # region 2
                ],
                'overall_rmse': float,
            }
        """

        # --- 0. Check guesses structure ---
        required_keys = ["means_in_contrast", "sigmas_in_contrast", "split_points"]
        for k in required_keys:
            if k not in guesses:
                raise ValueError(f"guesses must contain key '{k}'.")

        mu_guess_contrast = np.asarray(guesses["means_in_contrast"], dtype=float)
        sigma_guess_contrast = np.asarray(guesses["sigmas_in_contrast"], dtype=float)
        split_points = np.asarray(guesses["split_points"], dtype=float)

        if mu_guess_contrast.size != 3:
            raise ValueError(
                f"'means_in_contrast' must have length 3, got {mu_guess_contrast.size}"
            )
        if sigma_guess_contrast.size != 3:
            raise ValueError(
                f"'sigmas_in_contrast' must have length 3, got {sigma_guess_contrast.size}"
            )
        if np.any(sigma_guess_contrast <= 0):
            raise ValueError("'sigmas_in_contrast' must all be > 0.")

        if split_points.size != 3:
            raise ValueError(
                f"'split_points' must have length 3, got {split_points.size}"
            )
        if not (split_points[0] < split_points[1] < split_points[2]):
            raise ValueError(
                f"'split_points' must satisfy s1 < s2 < s3, got {split_points}"
            )

        s1, s2, s3 = split_points

        # --- 1. Clean and inspect data ---
        contrasts = np.asarray(contrasts, dtype=float)
        finite_mask = np.isfinite(contrasts)
        y = contrasts[finite_mask]

        if debug_Bad_Data:
            print(f"Total points: {contrasts.size}, finite: {y.size}")
            print("Any NaN?", np.isnan(contrasts).any())
            print("Any inf?", np.isinf(contrasts).any())
            if y.size > 0:
                print("Finite min, max:", np.nanmin(y), np.nanmax(y))

        if y.size == 0:
            raise ValueError("No finite data points available for Gaussian fits.")

        # --- 2. Build region masks based on split_points ---
        # Region 0: peak 1
        mask0 = (y < s1)
        # Region 1: peak 2
        mask1 = (y >= s1) & (y < s2)
        # Valley [s2, s3): ignored
        valley_mask = (y >= s2) & (y < s3)
        # Region 2: peak 3
        mask2 = (y >= s3)

        region_masks = [mask0, mask1, mask2]
        region_sizes = [int(m.sum()) for m in region_masks]

        if debug_Bad_Data:
            print("Region sizes (peak1, peak2, peak3):", region_sizes)
            print("Valley size:", int(valley_mask.sum()))

        if any(sz == 0 for sz in region_sizes):
            raise ValueError(f"At least one peak region has zero points: sizes={region_sizes}")

        used_mask = mask0 | mask1 | mask2
        total_N_used = float(used_mask.sum())

        # --- 3. Fit each region: norm.fit + residuals ---
        mu_hats = []
        sigma_hats = []
        weights = []
        region_residuals = []

        overall_residuals = []

        for k, mask_k in enumerate(region_masks):
            y_k = y[mask_k]
            N_k = y_k.size

            if debug_Bad_Data:
                print(f"\n--- Peak region {k} ---")
                print("N_k:", N_k)
                print("y_k min, max:", y_k.min(), y_k.max())
                print("guess mu, sigma:", mu_guess_contrast[k], sigma_guess_contrast[k])

            # MLE fit of Normal
            if N_k >= 2:
                loc_hat, scale_hat = norm.fit(y_k)
            else:
                # too few points: fall back completely to guesses
                loc_hat = float(mu_guess_contrast[k])
                scale_hat = float(sigma_guess_contrast[k])

            # guard against degenerate / nonsense scale
            if scale_hat <= 0 or not np.isfinite(scale_hat):
                scale_hat = float(sigma_guess_contrast[k])

            mu_hats.append(float(loc_hat))
            sigma_hats.append(float(scale_hat))
            weights.append(N_k / total_N_used)

            # residuals for this region
            residuals_k = y_k - loc_hat
            rmse_k = float(np.sqrt(np.mean(residuals_k**2)))
            mae_k = float(np.mean(np.abs(residuals_k)))

            region_residuals.append(
                {
                    "rmse": rmse_k,
                    "mae": mae_k,
                    "n_points": int(N_k),
                }
            )

            # collect for overall residual
            overall_residuals.append(residuals_k)

        # --- 4. Overall residual metric (over all peak regions) ---
        if overall_residuals:
            overall_residuals = np.concatenate(overall_residuals)
            overall_rmse = float(np.sqrt(np.mean(overall_residuals**2)))
        else:
            overall_rmse = float("nan")

        # --- 5. Package metrics and results ---
        metrics = {
            "region_sizes": region_sizes,
            "valley_size": int(valley_mask.sum()),
            "region_residuals": region_residuals,
            "overall_rmse": overall_rmse,
        }

        results = {
            "metrics": metrics,
            "means_in_contrast": mu_hats,      # list of 3 floats
            "sigmas_in_contrast": sigma_hats,  # list of 3 floats
            "weights": weights,                # list of 3 floats (fractions of used points)
            "stan_fit": None,                  # kept for API compatibility
            "standardisation": {
                "finite_mask": finite_mask.tolist(),
                "split_points": split_points.tolist(),
            },
        }

        return results






    #new calibration
    def calibrate_with_stan(self, select_by_metric: str,
                  reject_great_residuals: bool = False,
                  r2_PSF_fit: float = 0.9,
                  debug_Bad_Data=False):
        # Sanity check on metric choice, even though we no longer use AIC/BIC/score literally
        if select_by_metric not in ('aic', 'bic', 'score'):
            raise ValueError(f'The selected metric must be either aic, bic or score, not {select_by_metric}')

        # Prep for possible future extension with different numbers of components
        metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
        guesses = Event_Table.MS1000_guesses
        # (Currently not used by Stan, but kept for logical continuity / future use.)

        # Optionally reject events with poor PSF fit
        contrasts = self.contrasts
        if reject_great_residuals:
            contrasts = contrasts[self.r2_fit > r2_PSF_fit]
        else:
            pass

        #--------Clipping the data--------#
        metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
        contrasts = contrasts[(contrasts > ((-30-np.array(metaparams)[1])/(np.array(metaparams)[0]))) & (contrasts < ((500-np.array(metaparams)[1])/(np.array(metaparams)[0])))] 
        print(f'splits utilised {(np.array([110, 200, 280])+ np.array(metaparams)[1])*(np.array(metaparams)[0]).tolist()}')

        #-------guesses
        guesses = {'means_in_contrast':((np.array(Event_Table.MS1000_guesses['means']) - np.array(metaparams)[1])*(np.array(metaparams)[0])).tolist(),
                    'sigmas_in_contrast':((np.array(Event_Table.MS1000_guesses['sigmas']))*(np.array(metaparams)[0])).tolist(),
                    'split_points':  ((np.array([140, 220, 280])+ np.array(metaparams)[1])*(np.array(metaparams)[0])).tolist()}
        

        print(f'One unit on z-axis is  {(np.std(1*np.std(contrasts))+np.mean(contrasts))/(np.array(metaparams)[0])+np.array(metaparams)[1]} kDa roughly')
        results = self._fit_gmm_with_stan_1000(contrasts, guesses, debug_Bad_Data=debug_Bad_Data)

        metric_val = results['metrics']['mean_log_lik']
        print(
            f"Stan GMM fit completed. Mean log-likelihood per point = {metric_val:.3f}, "
            f"max R-hat = {results['metrics']['rhat_max']:.3f}, "
            f"min ESS = {results['metrics']['ess_min']:.1f}"
        )

        # ----- Linear Regression (unchanged) -----
        model = LinearRegression()
        contrast_means = np.array(results['means_in_contrast']).reshape(-1, 1)
        model.fit(contrast_means, np.array(Event_Table.MS1000_guesses['means']))

        self.gradient_kDa_per_contrast = model.coef_[0]
        self.intercept_kDa_per_contrast = model.intercept_
        self.r2 = model.score(
            contrast_means,
            np.array(Event_Table.MS1000_guesses['means'])
        )

        self.calibrated_peaks = model.predict(contrast_means)
        self.calibrated_core_sigmas = (
            np.array(results['sigmas_in_contrast']) * self.gradient_kDa_per_contrast
        )

        # Change the stored values of masses
        self.masses_kDa = (
            self.contrasts * self.gradient_kDa_per_contrast
            + self.intercept_kDa_per_contrast
        )

        # Store the results of the calibration to results
        self.calibration_results = results

        # Change status to calibrated and print a message
        self.calibrated = 1

        print(
            f'Your event table was successfully calibrated with an $R^{2}$ value of {self.r2:.4f} '
            f'and calibration peak values of {self.calibrated_peaks} '
            f'and corresponding sigmas of {self.calibrated_core_sigmas}'
        )
        print(
            f'Calibration gradient: {self.gradient_kDa_per_contrast} kDa per contrast unit, '
            f'intercept: {self.intercept_kDa_per_contrast} kDa'
        )   






            

    

        




