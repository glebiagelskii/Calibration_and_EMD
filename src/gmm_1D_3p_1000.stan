data {
  int<lower=1> N;                   // number of data points
  vector[N] y;                      // observations (already standardised in Python)

  // Informative guesses, also in the same space as y
  vector[3] mu_guess_z;             // guessed means (sorted low->high)
  vector<lower=0>[3] sigma_guess_z; // guessed sigmas
}

parameters {
  simplex[3] theta;                 // mixture weights
  ordered[3] mu;                    // ordered means
  vector[3] log_sigma;              // log standard deviations
}

transformed parameters {
  vector<lower=0>[3] sigma;         // standard deviations
  sigma = exp(log_sigma);
}

model {
  // -------- Weights --------
  // No tiny components expected; favour roughly balanced weights.
  theta ~ dirichlet(rep_vector(4.0, 3));

  // -------- Means: peaks 1 & 2 tight, peak 3 softer --------
  // Peaks 1 and 2: same as before (sd = 0.05)
  mu[1] ~ normal(mu_guess_z[1], 0.05);
  mu[2] ~ normal(mu_guess_z[2], 0.05);

  // Peak 3: softer prior (larger sd, e.g. 0.20)
  mu[3] ~ normal(mu_guess_z[3]-1.5, 1);

  // -------- Repulsion between means --------
  // 1–2 gap: same as before
  {
    real gap12 = mu[2] - mu[1];
    real gap12_guess = mu_guess_z[2] - mu_guess_z[1];
    gap12 ~ normal(gap12_guess, 1);     // unchanged
  }

  // 2–3 gap: softer repulsion (larger sd than 1.0)
  {
    real gap23 = mu[3] - mu[2];
    real gap23_guess = mu_guess_z[3] - mu_guess_z[2];
    gap23 ~ normal(gap23_guess, 4);     // e.g. 2.0 instead of 1.0
  }

  // -------- Sigmas: peaks 1 & 2 tight, peak 3 softer --------
  // Peaks 1 & 2: as before (very tight around guesses)
  log_sigma[1] ~ normal(log(sigma_guess_z[1]), 0.005);
  log_sigma[2] ~ normal(log(sigma_guess_z[2]), 0.005);

  // Peak 3: softer prior on log-sigma (e.g. sd = 0.05)
  log_sigma[3] ~ normal(log(sigma_guess_z[3]), 0.0001);

  // -------- Marginalised mixture likelihood --------
  for (n in 1:N) {
    vector[3] lps;
    for (k in 1:3) {
      lps[k] = log(theta[k]) + normal_lpdf(y[n] | mu[k], sigma[k]);
    }
    target += log_sum_exp(lps);
  }
}