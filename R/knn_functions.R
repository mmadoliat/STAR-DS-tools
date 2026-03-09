knn_pred <- function(train_x, train_y, test_x, k, method = "cpp") {
  # Coerce types
  train_x <- as.matrix(train_x)
  test_x  <- as.matrix(test_x)
  train_y <- as.numeric(train_y)
  
  # Scale train and test appropriately
  train_x <- scale(train_x)
  # Extract the mean and sd used
  train_mean <- attr(train_x, "scaled:center")
  train_sd <- attr(train_x, "scaled:scale")

  # Apply the same transformation to test data
  test_x <- scale(test_x, center = train_mean, scale = train_sd)
  
  n <- nrow(train_x) 
  p <- ncol(train_x)
  m <- nrow(test_x)
  
  if (method != "cpp") {
    
    # Store predictions
    preds <- numeric(m)
    # Store ids of k nearest neighbors
    ids <- numeric(m * k)
    
    for (j in seq_len(m)) {
      
      # Squared Euclidean distances using matrix ops (fast):
      # t(train_x) is p x n; subtract p-vector test_x[j,]; colSums -> length n
      dists <- colSums((t(train_x) - test_x[j, ])^2)
      
      # Indices of k smallest distances (full sort, simple & reliable)
      idx_k <- order(dists)[seq_len(k)]
      
      # Mean of neighbor labels
      preds[j] <- mean(train_y[idx_k])
      
      # Store ids of nearest neighbors
      # ids of vars
      ids[((j-1) * k + 1):(j * k)] <- idx_k
    }
    
    out = list("preds" = preds, "ids" = ids)
    
  } else {
    
    out = knn_pred_cpp(train_x, train_y, test_x, k)
  
  }
  
  out
}




