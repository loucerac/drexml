library(rstan)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#stan_dat <- read_rdump('gp-fit-multi-output.data.R')

stan_dat <- list(
    x = as.matrix(X_train),
    y = as.matrix(Y_train),
    N = nrow(X_train),
    M = ncol(X_train),
    D = ncol(Y_train))

fit_mult_out <- stan(file="mogp.stan",
                     data=stan_dat,
                     iter=200, chains=8,
                     control = list(max_treedepth = 15))
print(fit_mult_out, c('rho','alpha','sigma','Omega'))

# print(apply(rstan::extract(fit_mult_out)$Omega,c(2,3), mean))
# print(stan_dat$Omega)
