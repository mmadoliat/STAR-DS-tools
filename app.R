# app.R

library(shiny)
library(Rcpp)
# compile Rcpp code once at startup
Rcpp::sourceCpp("src/knn_pred.cpp")

# load our formula‐interface S3 model
source("R/knn_s3_formula.R")

# fixed split function
make_split <- function(data, prop = 0.7) {
  n <- nrow(data)
  train_idx <- sample(n, size = floor(prop * n))
  list(train = data[train_idx, ], test = data[-train_idx, ])
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
    make_split(mtcars, prop = 0.7)
  }, ignoreNULL = FALSE)
  
  # fit model on training
  model <- reactive({
    d <- split()$train
    knn_s3(mpg ~ disp + hp + wt, data = d, k = input$k)
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
    cat("  MSE =", round(mse(obs, fitted_vals()), 3),
        "  R² =", round(r2(obs, fitted_vals()), 3), "\n")
  })
  output$testPerf <- renderPrint({
    obs <- split()$test$mpg
    cat("Test set:\n")
    cat("  MSE =", round(mse(obs, predicted()), 3),
        "  R² =", round(r2(obs, predicted()), 3), "\n")
  })
  
  # scatter plot of actual vs predicted on test set
  output$predPlot <- renderPlot({
    df <- split()$test
    df$pred <- predicted()
    plot(df$mpg, df$pred,
         xlab = "Actual mpg", ylab = "Predicted mpg",
         pch = 16, col = "#2C3E50AA",
         main = paste("k-NN (k =", input$k, ", backend =", input$backend, ")"))
    abline(0,1, col = "firebrick", lwd = 2)
  })
  
  # show first few test observations
  output$predTable <- renderTable({
    df <- split()$test
    data.frame(
      car = rownames(df),
      actual = round(df$mpg,2),
      predicted = round(predicted(),2)
    )[1:6, ]
  }, rownames = FALSE)
}

shinyApp(ui, server)
