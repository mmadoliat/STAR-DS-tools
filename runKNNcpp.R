library(microbenchmark)
source("R/knn_functions.R")
Rcpp::sourceCpp("src/knn_predDC.cpp")

set.seed(42)
n <- 1000; p <- 5; m <- 200; k <- 10
Xtrain <- matrix(rnorm(n*p), n, p)
Ytrain <- rnorm(n)
Xtest  <- matrix(rnorm(m*p), m, p)

# warm up & check correctness
pred_R <- knn_pred(Xtrain, Ytrain, Xtest, k, method="R")
pred_C <- knn_pred(Xtrain, Ytrain, Xtest, k, method="cpp")
stopifnot(max(abs(pred_R$preds - pred_C$preds)) < 1e-12)

# benchmark
mb <- microbenchmark(
  pure_R = knn_pred(Xtrain, Ytrain, Xtest, k, method="R"),
  Rcpp   = knn_pred(Xtrain, Ytrain, Xtest, k, method="cpp"),
  times  = 20
)
print(mb)
