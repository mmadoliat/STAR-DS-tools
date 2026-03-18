// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector knn_pred_cpp(NumericMatrix train_x,
                           NumericVector train_y,
                           NumericMatrix test_x,
                           int k) {
  int n = train_x.nrow(), 
    p = train_x.ncol(), 
    m = test_x.nrow();
  NumericVector preds(m);
  
  for (int j = 0; j < m; ++j) {
    // compute and store (distance,index)
    std::vector<std::pair<double,int>> dists(n);
    for (int i = 0; i < n; ++i) {
      double sumsq = 0;
      for (int d = 0; d < p; ++d) {
        double diff = train_x(i,d) - test_x(j,d);
        sumsq += diff * diff;
      }
      dists[i] = std::make_pair(sumsq, i);
    }
    // partial sort to get k nearest
    std::nth_element(dists.begin(), dists.begin() + k, dists.end(),
                     [](auto &a, auto &b){ return a.first < b.first; });
    double s = 0;
    for (int r = 0; r < k; ++r) {
      s += train_y[dists[r].second];
    }
    preds[j] = s / k;
  }
  
  return preds;
}
