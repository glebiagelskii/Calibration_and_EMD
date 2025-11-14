from sklearn.mixture import GaussianMixture
import numpy as np

from typing import Optional
def fit_gmm(contrasts: np.ndarray, metaparams: list, guesses: dict):
    """This function fits a GMM to Event Table's contrasts and returns a dictionary with parameters and metrics of fit back"""        
    gradient=metaparams[0]
    intercept = metaparams[1]
    guess_mean_masses = np.array(guesses['means'])
    guess_sigma_masses = np.array(guesses['sigmas'])
    K = guess_mean_masses.shape[0]

    guess_mean_contrasts = (guess_mean_masses - intercept)*gradient
    guess_sigma_contrasts = (guess_sigma_masses - intercept)*gradient
    guess_precisions_contrasts = 1/(guess_sigma_contrasts**2)
    print(f'Initial mean contrasts guesses: {guess_mean_contrasts}')
    print(f'Initial contrast precisions guesses: {guess_precisions_contrasts}')
    #Note no utilisation of the intercept above

    #Clip the contrast window to [0, 6 * expected monomer contrast]
    print(f'min contrast {min(contrasts)} max contrast {max(contrasts)} number of samples {contrasts.shape[0]}')
    contrasts = contrasts[(contrasts > 0) & (contrasts < (6 * guess_mean_contrasts[0]))]
    contrasts = contrasts.reshape(-1, 1)
    print(f'min contrast {min(contrasts)} max contrast {max(contrasts)} number of samples {contrasts.shape[0]}')

    gmm = GaussianMixture(n_components=3,
                          covariance_type='full',
                          tol=1e-12,
                          max_iter = 5000,
                          means_init=guess_mean_contrasts.reshape(-1,1),
                          precisions_init=guess_precisions_contrasts.reshape(-1,1,1),
                          n_init=1
                        )                      
    gmm.fit(contrasts)
    
    return {'means_in_contrast':[gmm.means_.ravel()],'sigmas_in_contrast' : [gmm.covariances_.ravel()], 'weights' :[gmm.weights_.ravel()],'metrics':{'score':gmm.score(contrasts), 'aic':gmm.aic(contrasts), 'bic': gmm.bic(contrasts), 'scores_per_sample':gmm.score_samples(contrasts)}}



