data {
  int<lower=1> N;              // number of data points
  vector[N] y;                 // observations (already standardised in Python)

  // Informative guesses, also in standardised space
  vector[3] mu_guess_z;        // guessed means (z-scored & sorted low->high)
  vector<lower=0>[3] sigma_guess_z; // guessed sigmas (z-scored)
}

parameters {
  simplex[3] theta;            // mixture weights
  ordered[3] mu;               // ordered means (in z-space)
  vector[3] log_sigma;         // log standard deviations (in z-space)
}

transformed parameters {
  vector<lower=0>[3] sigma;    // standard deviations (in z-space)
  sigma = exp(log_sigma);
}

model {
  // -------- Priors --------

  // Weights: no tiny components expected; favour roughly balanced weights.
  // Dirichlet(3,3,3) 
  theta ~ dirichlet(rep_vector(4.0, 3));

  // Means: tight priors around guessed locations (in z-space).
  // sd = 0.7 in z-space 
  for (k in 1:3) {
    mu[k] ~ normal(mu_guess_z[k], 0.005);
  }

  // Repulsion between means
  for (k in 2:3) {
    real gap = mu[k] - mu[k-1];
    real gap_guess = mu_guess_z[k] - mu_guess_z[k-1];
    gap ~ normal(gap_guess, 0.02);   
  }

  // Sigmas: log-normal around guessed sigmas, but wide.
  // sd(log_sigma) ~ 0.1 
  for (k in 1:3) {
    log_sigma[k] ~ normal(log(sigma_guess_z[k]), 0.0002);
  }

  // -------- Marginalised mixture likelihood --------
  for (n in 1:N) {
    vector[3] lps;
    for (k in 1:3) {
      lps[k] = log(theta[k]) + normal_lpdf(y[n] | mu[k], sigma[k]);
    }
    target += log_sum_exp(lps);
  }
}