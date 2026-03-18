library(microbenchmark)
source("R/knn_s3.R")
Rcpp::sourceCpp("src/knn_pred.cpp")

set.seed(42)
n <- 1000; p <- 5; m <- 200; k <- 10
Xtrain <- matrix(rnorm(n*p), n, p)
Ytrain <- rnorm(n)
Xtest  <- matrix(rnorm(m*p), m, p)

# build S3 model
mod <- knn_s3(Xtrain, Ytrain, k)
print(mod)

# warm up & check correctness
pred_R <- predict(mod, Xtest, method="R")
pred_C <- predict(mod, Xtest, method="cpp")
stopifnot(max(abs(pred_R - pred_C)) < 1e-12)

# benchmark
mb <- microbenchmark(
  pure_R = predict(mod, Xtest, method="R"),
  Rcpp    = predict(mod, Xtest, method="cpp"),
  times = 20
)
print(mb)




# install Rcpp & compile C++
Rcpp::sourceCpp("src/knn_pred.cpp")

# load model code
source("R/knn_s3_formula.R")

# Fit two k-NN models on mtcars
mod5  <- knn_s3(mpg ~ disp + hp + wt, mtcars, k = 5)
mod10 <- knn_s3(mpg ~ disp + hp + wt, mtcars, k = 10)

print(mod5)
#> knn_s3 model:
#>   Call: knn_s3(formula = mpg ~ disp + hp + wt, data = mtcars, k = 5)
#>   Training: n = 32  p = 4  k = 5 

summary(mod5)
#> Training performance:
#>   MSE: 9.437   R²: 0.72 

fitted(mod5)[1:5]
#> 20.01 21.44 23.62 25.12 21.78

predict(mod5, newdata = head(mtcars), method = "cpp")

anova(mod5, mod10)
#>                             Model  k       MSE   DeltaMSE
#>  knn_s3(mpg ~ disp + hp + wt, ...  5  9.437070         NA
#> knn_s3(mpg ~ disp + hp + wt, ...   10 10.582341   1.145271
