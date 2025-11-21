data {
  int<lower=1> N;                // number of data points
  vector[N] y;                   // observations (scaled by 1000)
  real mu_guess;                 // guessed mean (scaled)
  real<lower=0> sigma_guess;     // guessed sigma (scaled)
}

parameters {
  real mu;                       // mean (scaled)
  real<lower=0> sigma;           // sigma (scaled)
}

model {
  // ---- Priors ----
  // Mean: fairly informative around guess, but not insanely tight.
  // You can tune 50.0 depending on typical spacing in the scaled space.
  mu ~ normal(mu_guess, 20.0);

  // Sigma: log-normal around guess, moderately wide.
  sigma ~ lognormal(log(sigma_guess), 0.0001);

  // ---- Likelihood ----
  y ~ normal(mu, sigma);
}