# app.R

library(shiny)
library(Rcpp)
library(tidyverse)
# compile Rcpp code once at startup
Rcpp::sourceCpp("src/knn_predDC.cpp")

# load our formula‐interface S3 model
source("R/knn_s3_formulaDC.R")

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
  test_x_out = test_x[rep(1:m, each = k), ]
  train_x_out = train_x[ids, ]
  out = as.data.frame(cbind(test_x_out, train_x_out))
  colnames(out) <- c(paste0("test_", colnames(train_x)), paste0("train_", colnames(train_x)))
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
  
  # fit model on training
  model <- reactive({
    d <- split()$train
    knn_s3(mpg ~ wt + qsec, data = d, k = input$k)
  })
  
  # compute fitted (train) and predicted (test)
  fitted_vals <- reactive({
    predict(model(), newdata = split()$train, method = input$backend)
  })
  predicted    <- reactive({
    predict(model(), newdata = split()$test,  method = input$backend)
  })
  
  # compute metrics
  mse <- function(obs, pred) mean((obs - pred)^2)
  r2  <- function(obs, pred) 1 - sum((obs - pred)^2) / sum((obs - mean(obs))^2)
  
  output$trainPerf <- renderPrint({
    obs <- split()$train$mpg
    cat("Training set:\n")
    cat("  MSE =", round(mse(obs, fitted_vals()[[1]]), 3),
        "  R² =", round(r2(obs, fitted_vals()[[1]]), 3), "\n")
  })
  output$testPerf <- renderPrint({
    obs <- split()$test$mpg
    cat("Test set:\n")
    cat("  MSE =", round(mse(obs, predicted()[[1]]), 3),
        "  R² =", round(r2(obs, predicted()[[1]]), 3), "\n")
  })
  
  # scatter plot of actual vs predicted on test set
  output$predPlot <- renderPlot({
    df_test <- split()$test
    df_train <- split()$train
    df_test$pred <- predicted()[[1]]
    df_test$id = 1:nrow(df_test)
    
    # Create plot for geom_segment 
    plot_df <- segment_df(df_train[, c("wt", "qsec")], 
                          df_test[, c("wt", "qsec")], 
                          predicted()[[2]], input$k)
    
    ggplot() +
      geom_segment(aes(x = test_wt, y = test_qsec, xend = train_wt, yend = train_qsec), 
                   color = "grey80", data = plot_df, linetype = "dotted") +
      geom_point(aes(x = wt, y = qsec, color = mpg), data = df_train) +
      geom_text(aes(x = wt, y = qsec, color = pred, label = id), 
                fontface = "bold", data = df_test) +
      scale_color_gradient(low = "yellow", high = "red") +
      theme_classic() +
      labs(x = "wt", y = "qsec")
  })
  
  # show first few test observations
  output$predTable <- renderTable({
    df <- split()$test
    data.frame(
      id = 1:nrow(df),
      car = rownames(df),
      actual = round(df$mpg,2),
      predicted = round(predicted()[[1]],2)
    )
  }, rownames = FALSE)
}

shinyApp(ui, server)
