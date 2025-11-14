from sklearn.mixture import GaussianMixture
import numpy as np

from typing import Optional
def fit_gmm(contrasts: np.ndarray, metaparams: list, guesses: dict):
    """This function fits a GMM to Event Table's contrasts and returns a dictionary with parameters and metrics of fit back"""        
    gradient=metaparams[0]
    intercept = metaparams[1]
    guess_mean_masses = np.array(guesses['mean'])
    guess_sigma_masses = np.array(guesses['sigma'])
    K = guess_mean_masses.shape[0]

    guess_mean_contrasts = (guess_mean_masses - intercept)/gradient
    guess_sigma_contrasts = (guess_sigma_masses - intercept)/gradient
    guess_precisions_contrasts = 1/(guess_sigma_contrasts**2)

    #Clip the contrast window to [0, 5 * expected monomer contrast]
    contrasts = contrasts[contrasts>0 and contrasts<(5*guess_mean_contrasts[0])]
    contrasts = contrasts.reshape(1, -1)

    gmm = GaussianMixture(n_components=3,
                          covariance_type='full',
                          tol=0.00001,
                          max_iter = 500,
                          means_init=guess_mean_contrasts,
                          precisions_init=guess_precisions_contrasts,
                          n_init=1,
                          random_state=0
                        )                      
    gmm.fit(contrasts)
    
    return {'means_in_contrast':[gmm.means_.ravel()],'sigmas_in_contrast' : [gmm.covariances_.ravel()], 'weights' :[gmm.weights_.ravel()],'metrics_of_fit':{'score':gmm.score(contrasts), 'aic':gmm.aic(contrasts), 'bic': gmm.bic(contrasts), 'scores_per_sample':gmm.score(contrasts)}}



