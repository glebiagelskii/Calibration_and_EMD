"""This is a script that computes EMD metrics from a calibrated event table"""


from load_event_table import *
from scipy.stats import wasserstein_distance
import numpy as np
from pathlib import Path

class Event_Table():
    meta_calbibration_parameters = {'Refeyn OneMP':{'Medium':[], 'Full':[], 'Small':[], 'Regular':[]}, 'Refeyn TwoMP':{'Medium':[], 'Full':[], 'Small':[], 'Regular':[]}}
    def __init__(self, frame,	y_det,	x_det,	contrasts_det,	contrasts,	x_fit,	y_fit,	contrasts_se,	r2_fit,	res_fit,	x,	y,	masses_kDa, calibration_required, instrument_name, ROI_mode):
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
        self.instrument_name = Instrument
        self.ROI_mode = ROI

    def calibrate(self):
        

        




