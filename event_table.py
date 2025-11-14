"""This is a script that defines the core Event Table class"""


from load_event_table import *
from scipy.stats import wasserstein_distance
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from fit_gmm import fit_gmm

class Event_Table():
    meta_calibration_parameters = {'Refeyn OneMP':{'Medium':[], 'Full':[], 'Small':[], 'Regular':[]}, 'Refeyn TwoMP':{'Medium':[], 'Full':[], 'Small':[], 'Regular':[]}}
    MS1000_guesses = {'means':[90, 180, 360], 'sigmas':[9, 10, 20]}
    def __init__(self, frame,	y_det,	x_det,	contrasts_det,	contrasts,	x_fit,	y_fit,	contrasts_se,	r2_fit,	res_fit,	x,	y,	masses_kDa, calibration_required, instrument_name, ROI_mode, calibration_results, calibrated):
        self.frame = frame
        self.y_det = y_det
        self.x_det = x_det
        self.constrasts_det = contrasts_det
        self.constrasts = contrasts
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.constrasts_se = contrasts_se
        self.r2_fit = r2_fit
        self.res_fit = res_fit
        self.x = x
        self.y = y
        self.masses_kDa = masses_kDa
        self.fit_required = calibration_required 
        self.instrument_name = instrument_name
        self.ROI_mode = ROI_mode

        self.calibration_results = None
        calibrated = 0
        self.gradient_kDa_per_contrast = None
        self.intercept_kDa_per_contrast = None
        self.r2 = None
        self.calibrated_peaks = None
        self.calibrated_core_sigmas = None
    
    def from_dataframe(self, file_path: str, ROI: str, Instrument:str, needs_calibration: bool):

        df = load_event_table(Path(file_path))

        if table_header_format_is_correct(df):
            pass
        else:
            raise ValueError('Review Header')
        
        self.frame = df['frame'].values()
        self.y_det = df['y'].values()
        self.x_det = df['y'].values()
        self.constrasts_det = df['y'].values()
        self.constrasts = df['y'].values()
        self.x_fit = df['y'].values()
        self.y_fit = df['y'].values()
        self.constrasts_se = df['y'].values()
        self.r2_fit = df['y'].values()
        self.res_fit = df['y'].values()
        self.x = df['y'].values()
        self.y = ydf['y'].values()
        self.masses_kDa = df['y'].values()
        self.calibration_required = needs_calibration
        if Instrument == 'Refeyn OneMP' or Instrument=='Refeyn TwoMP':
            self.instrument_name = Instrument
        else:
            raise ValueError(f'Intrument name cannot be {Instrument}, only Refeyn OneMP or Refeyn TwoMP')
        if ROI=='Regular' or ROI=='Full' or ROI=='Small' or ROI=='Medium':
            self.ROI_mode = ROI
        else:
            raise ValueError(f'ROI name cannot be {ROI}, only Medium, Small, Full or Regular')

    def calibrate(self, select_by_metric:str):
        best_aic = 1000
        best_bic = 1000
        best_score = 1000

        #Note the current structure is prep work for when potential future iterations will attempt fits with vairable numbers of peak components
        if select_by_metric=='aic':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            results = fit_gmm(self.contrasts, metaparams, guesses)
            print(f'The Aikake information criterion for the GMM fit is {results['metrics']['aic']}')
        elif select_by_metric=='bic':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            results = fit_gmm(self.contrasts, metaparams, guesses)
            print(f'The Bayesian information criterion for the GMM fit is {results['metrics']['bic']}')
        elif select_by_metric=='score':
            metaparams = Event_Table.meta_calibration_parameters[self.instrument_name][self.ROI_mode]
            guesses = Event_Table.MS1000_guesses
            results = fit_gmm(self.contrasts, metaparams, guesses)
            print(f'The Bayesian information criterion for the GMM fit is {results['metrics']['']}')
        else:
            raise ValueError(f'The selected metric must be either aic, bic or score, not {select_by_metric}')
        
        #Linear Regression
        model = LinearRegression()
        contrast_means = np.array(results['means_in_contrast']).reshape(-1,1)
        model.fit(contrast_means, np.array(Event_Table.MS1000_guesses['means']))
        self.gradient_kDa_per_contrast = model.coef_[0]
        self.intercept_kDa_per_contrast = model.coef_[1] 
        self.r2 = model.score(contrast_means, np.array(Event_Table.MS1000_guesses['means']) )
        self.calibrated_core_peaks = model.predict(contrast_means, np.array(Event_Table.MS1000_guesses['means']) )
        self.calibrated_core_sigmas = np.array(results['sigmas_in_contrast'])* self.gradient_kDa_per_contrast + self.intercept_kDa_per_contrast
        #Change the stored values of masses
        self.masses_kDa = self.constrasts*self.gradient_kDa_per_contrast + self.intercept_kDa_per_contrast

        #Store the results of the calibration_results to results
        self.calibration_results = results

        #Change status to calibrated and print a message
        self.calibrated = 1

        print(f'Your event table was succesfully calibrated with an $R^{2}$ value of {self.r2} and calibration peak values of {self.calibrated_core_peaks} and corresponding sigmas of {self.calibrated_core_sigmas}')









            

    

        




