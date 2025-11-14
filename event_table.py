"""This is a script that defines the core Event Table class"""


from load_event_table import *
from scipy.stats import wasserstein_distance
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fit_gmm import fit_gmm

class Event_Table():
    meta_calibration_parameters = {'Refeyn OneMP':{'Medium':[2.8E-5, 0 ], 'Full':[2.8E-5, 0 ], 'Small':[2.8E-5, 0 ], 'Regular':[2.8E-5, 0 ]}, 'Refeyn TwoMP':{'Medium':[4.2E-5, 7E-4 ], 'Full':[4.2E-5,8], 'Small':[4.2E-5, 8 ], 'Regular':[4.2E-5, 8 ]}}
    MS1000_guesses = {'means':[90, 180, 360], 'sigmas':[2, 9, 20]}
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
       

    def calibrate(self, select_by_metric:str):
        best_aic = 1000
        best_bic = 1000
        best_score = 1000

        #Note the current structure is prep work for when potential future iterations will attempt fits with vairable numbers of peak components

        if select_by_metric=='aic':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            results = fit_gmm(self.contrasts, metaparams, guesses)
            print(f"The Aikake information criterion for the GMM fit is {results['metrics']['aic']}'=")
        elif select_by_metric=='bic':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            results = fit_gmm(self.contrasts, metaparams, guesses)
            print(f"The Bayesian information criterion for the GMM fit is {results['metrics']['bic']}")
        elif select_by_metric=='score':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            results = fit_gmm(self.contrasts, metaparams, guesses)
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
        self.calibrated_core_peaks = model.predict(contrast_means )
        self.calibrated_core_sigmas = np.array(results['sigmas_in_contrast'])* self.gradient_kDa_per_contrast + self.intercept_kDa_per_contrast
        #Change the stored values of masses
        self.masses_kDa = self.contrasts*self.gradient_kDa_per_contrast + self.intercept_kDa_per_contrast

        #Store the results of the calibration_results to results
        self.calibration_results = results

        #Change status to calibrated and print a message
        self.calibrated = 1

        print(f'Your event table was succesfully calibrated with an $R^{2}$ value of {self.r2} and calibration peak values of {self.calibrated_core_peaks} and corresponding sigmas of {self.calibrated_core_sigmas}')
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
            plt.xlim([x_min, x_max])
        else:
            x_min = min(self.masses_kDa)*1.1
            x_max = max(self.masses_kDa)*1.1
        plt.xlim([x_min, x_max])

        #Plot the fitted GMM peaks
        from scipy.stats import norm
        x = np.linspace(x_min, x_max, 1000)
        

        # Normalize/squeeze arrays to 1D
        means = np.ravel(self.calibrated_core_peaks)
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
        plt.xlabel('Mass (kDa)')







            

    

        




