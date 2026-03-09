# app.R

library(shiny)
library(Rcpp)
library(tidyverse)
# compile Rcpp code once at startup
Rcpp::sourceCpp("src/knn_predDC.cpp")

# load our formula‐interface S3 model
source("R/knn_functions.R")

# fixed split function
make_split <- function(data, prop = 0.7) {
  n <- nrow(data)
  train_idx <- sample(n, size = floor(prop * n))
  list(train = data[train_idx, ], test = data[-train_idx, ])
}

# Used to generate plot
segment_df <- function(train_x, test_x, ids, k){
  train_x <- as.matrix(train_x)
  test_x  <- as.matrix(test_x)
  
  n <- nrow(train_x); p <- ncol(train_x); m <- nrow(test_x)
  
  # Create df with each test point alongside k nearest neighbors
  test_x_out = test_x[rep(1:m, each = k), ]
  train_x_out = train_x[ids, ]
  out = as.data.frame(cbind(test_x_out, train_x_out))
  colnames(out) <- c(paste0("test_", colnames(train_x)), 
                     paste0("train_", colnames(train_x)))
  out
}

ui <- fluidPage(
  titlePanel("k-NN Regression Explorer"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("k", "Number of neighbors (k):",
                  min = 1, max = 20, value = 5, step = 1),
      radioButtons("backend", "Compute backend:",
                   choices = c("R","cpp"), selected = "cpp"),
      actionButton("resample", "New train/test split"),
      hr(),
      h4("Performance"),
      verbatimTextOutput("trainPerf"),
      verbatimTextOutput("testPerf")
    ),
    mainPanel(
      plotOutput("predPlot", height = "400px"),
      tableOutput("predTable")
    )
  )
)

server <- function(input, output, session) {
  # reactive train/test split, re-draw when resample pressed
  split <- eventReactive(input$resample, {
    make_split(mtcars, prop = 0.92)
  }, ignoreNULL = FALSE)
  
  # compute fitted (train) and predicted (test)
  fitted_vals <- reactive({
    d_train <- split()$train
    knn_pred(d_train[, c("wt", "qsec")], d_train[, c("mpg")], 
             d_train[, c("wt", "qsec")], input$k, input$backend)
  })
  predicted <- reactive({
    d_train <- split()$train
    d_test <- split()$test
    knn_pred(d_train[, c("wt", "qsec")], d_train[, c("mpg")], 
             d_test[, c("wt", "qsec")], input$k, input$backend)
  })
  
  # compute metrics
  mse <- function(obs, pred) mean((obs - pred)^2)
  r2  <- function(obs, pred) 1 - (sum((obs - pred)^2) / sum((obs - mean(obs))^2))
  
  output$trainPerf <- renderPrint({
    obs <- split()$train$mpg
    cat("Training set:\n")
    cat("  MSE =", round(mse(obs, fitted_vals()$preds), 3),
        "  R² =", round(r2(obs, fitted_vals()$preds), 3), "\n")
  })
  output$testPerf <- renderPrint({
    obs <- split()$test$mpg
    cat("Test set:\n")
    cat("  MSE =", round(mse(obs, predicted()$preds), 3),
        "  R² =", round(r2(obs, predicted()$preds), 3), "\n")
  })
  
  # Plot showing the k nearest neighbors for each test point
  output$predPlot <- renderPlot({
    # Store train/test
    d_train <- split()$train
    d_test <- split()$test
    d_test$id = 1:nrow(d_test)
    
    # Create df for geom_segment 
    nn <- segment_df(d_train[, c("wt", "qsec")], 
                     d_test[, c("wt", "qsec")], 
                     predicted()$ids, input$k)
    
    # Add predictions for test set
    d_test$pred <- predicted()$preds
    
    # Create the plot
    ggplot() +
      geom_segment(aes(x = test_wt, xend = train_wt, 
                       y = test_qsec, yend = train_qsec), 
                   color = "grey50", linetype = "dotted",
                   data = nn) +
      geom_point(aes(x = wt, y = qsec, color = mpg), 
                 size = 2, data = d_train) +
      geom_text(aes(x = wt, y = qsec, color = pred, label = id), 
                size = 5, fontface = "bold", data = d_test) +
      scale_color_gradient(low = "#6A00A8FF", high = "#FCA636FF") +
      theme_classic() +
      labs(x = "wt", y = "qsec")
  })
  
  # show first few test observations
  output$predTable <- renderTable({
    d_test <- split()$test
    data.frame(
      id = 1:nrow(d_test),
      car = rownames(d_test),
      actual = round(d_test$mpg,2),
      predicted = round(predicted()$preds,2)
    )
  }, rownames = FALSE)
}

shinyApp(ui, server)
