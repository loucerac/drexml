rm(list = ls())

library(here)
library(hipathia)
library(dotenv)
dotenv::load_dot_env(file = here(".env"))
data_path <- Sys.getenv("DATA_PATH")

library(randomForestSRC)
library(parallel)

source(here("notebooks", "metrics.R"))

n_cores <- detectCores() - 1
options(rf.cores = n_cores)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

pathways <- load_pathways("hsa")

## ------------------------------------------------------------
## interactions for multivariate mixed forests
## ------------------------------------------------------------


X_train <- feather::read_feather(file.path(data_path, "X_train.feather"))
X_train <- as.data.frame(X_train)
rownames(X_train) <- X_train$index
X_train$index <- NULL
Y_train <- feather::read_feather(file.path(data_path, "Y_train.feather"))
Y_train <- as.data.frame(Y_train)
rownames(Y_train) <- Y_train$index
Y_train$index <- NULL
X_test <- feather::read_feather(file.path(data_path, "X_test.feather"))
X_test <- as.data.frame(X_test)
rownames(X_test) <- X_test$index
X_test$index <- NULL
Y_test <- feather::read_feather(file.path(data_path, "Y_test.feather"))
Y_test <- as.data.frame(Y_test)
rownames(Y_test) <- Y_test$index
Y_test$index <- NULL

circuit_names <- names(Y_train)
gene_names <- names(X_train)

data_train <- cbind(X_train, Y_train)
rm(X_train, Y_train)

f <- as.formula(paste("cbind(",paste(circuit_names,collapse=","),")~.",sep=""))

model <- rfsrcFast(f, data_train)

data_test <- cbind(X_test, Y_test)
rm(X_test, Y_test)

res <- predict(model, data_test)

rm(res, model)

morf_cv <- tune(f, data_train, trace=TRUE, ntreeTry = 500)

res <- predict(morf_cv$rf, data_test)

Y_hat <- get_preds_rfsrc(res)

r2_raw <- r2_score(Y_test, Y_hat)


library(glmnet)
library(doParallel)

registerDoParallel(4)


cvmfit = cv.glmnet(as.matrix(X_train), as.matrix(Y_train), family = "mgaussian", parallel=TRUE)


err <- get.mv.error(rf)
vmp <- get.mv.vimp(rf)
pred <- get.mv.predicted(rf)

err.std <- get.mv.error(rf, standardize = TRUE)
vmp.std <- get.mv.vimp(rf, standardize = TRUE)

vmp <- vimp(rf)

