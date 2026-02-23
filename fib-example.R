# Iterative Fibonacci in R (more efficient than recursive)
fib_r <- function(n) {
  if (n <= 1) return(n)
  a <- 0
  b <- 1
  for (i in 2:n) {
    temp <- b
    b <- a + b
    a <- temp
  }
  return(b)
}

library(Rcpp)

cppFunction('
int fib_cpp(int n) {
  if (n <= 1) return n;
  int a = 0, b = 1, temp;
  for (int i = 2; i <= n; ++i) {
    temp = b;
    b = a + b;
    a = temp;
  }
  return b;
}
')

library(microbenchmark)

microbenchmark(
  R = fib_r(30),
  Rcpp = fib_cpp(30),
  times = 100L
)

microbenchmark(
  R = fib_r(30000),
  Rcpp = fib_cpp(30000),
  times = 1000L
)