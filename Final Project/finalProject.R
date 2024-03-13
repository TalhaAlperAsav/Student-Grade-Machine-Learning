#
# Made by Talha Alper Asav (201805072) and Ayşe Zehra Marangoz (201805005)
# This project's aim is to predict "G3" values of the Portuguese Lecture.
# Lecture: 2023-2024 FALL- CSE 418 DATA MINING
# Lecturer: Denizhan Demirkol 

# Installing necessary libraries (packages)
install.packages("ISLR")
install.packages("ggplot2")
install.packages("lattice")
install.packages("caret")
install.packages("dplyr")
install.packages("e1071")
install.packages("class")
install.packages("tidyverse")
install.packages("reshape2")
install.packages("randomForest")
install.packages("gbm")
install.packages("ada")
install.packages("fastAdaboost")
install.packages("gridExtra")
install.packages("rpart")
install.packages("corrplot")

# Loading the installed libraries
library(ISLR)
library(ggplot2)
library(lattice)
library(caret)
library(dplyr)
library(e1071)
library(class)
library(tidyverse)
library(reshape2)
library(randomForest)
library(gbm)
library(rpart)
library(gridExtra)
library(ada)
library(corrplot)

# Reading the .csv for Math lecture
studentMathData = read.table("C:/Users/ALPER ASAV/Desktop/Veri Madenciliği/student+performance/student/student-mat.csv",sep=";",header=TRUE)
column_names <- names(studentMathData)

# Reading the .csv for Portuguese lecture
studentPorData = read.table("C:/Users/ALPER ASAV/Desktop/Veri Madenciliği/student+performance/student/student-por.csv",sep=";",header=TRUE)
column_names <- names(studentPorData)

head(studentPorData, 1)

# Here we are merging the two lectures
mergedData=merge(studentMathData,studentPorData,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(mergedData))
write.csv(mergedData, "merged_data.csv", row.names = FALSE)

# Reading our merged .csv file
studentData <- read.csv("C:/Users/ALPER ASAV/Desktop/Veri Madenciliği/student+performance/student/merged_data.csv")

# Print the column names
column_names <- names(studentData)
print(column_names)

summary(studentData)

colSums(is.na(studentData))

# We decided to use only data of the Portuguese Lecture. 
# Because with the merging the number of instances got really decreased.
# Because of the number of instances of the Math Lecture.

# ------------------- Visualization for Data ------------------

# Histogram for the 'age' variable
ggplot(studentPorData, aes(x = age)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency")

# Histogram of G3 scores
ggplot(studentPorData, aes(x = G3)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Distribution of G3 Scores", x = "G3", y = "Frequency")

# Barplot of school frequency
ggplot(studentPorData, aes(x = school)) +
  geom_bar(fill = "orange") +
  labs(title = "Frequency of Schools", x = "School", y = "Frequency")

# Scatterplot of G3 vs. absences
ggplot(studentPorData, aes(x = absences, y = G3)) +
  geom_point(color = "red") +
  labs(title = "Scatterplot of Absences vs. G3", x = "Absences", y = "G3")

# Label Encoding for binary categorical variables
studentPorData$sex <- as.numeric(factor(studentPorData$sex))
studentPorData$address <- as.numeric(factor(studentPorData$address))
studentPorData$famsize <- as.numeric(factor(studentPorData$famsize))
studentPorData$Pstatus <- as.numeric(factor(studentPorData$Pstatus))
studentPorData$schoolsup <- as.numeric(factor(studentPorData$schoolsup))
studentPorData$famsup <- as.numeric(factor(studentPorData$famsup))
studentPorData$paid <- as.numeric(factor(studentPorData$paid))
studentPorData$activities <- as.numeric(factor(studentPorData$activities))
studentPorData$nursery <- as.numeric(factor(studentPorData$nursery))
studentPorData$higher <- as.numeric(factor(studentPorData$higher))
studentPorData$internet <- as.numeric(factor(studentPorData$internet))
studentPorData$romantic <- as.numeric(factor(studentPorData$romantic))


# One-Hot Encoding using the 'model.matrix' function
encodedData <- model.matrix(~ school + Mjob + Fjob + reason + guardian, data = studentPorData)
studentPorData <- cbind(studentPorData[, -which(names(studentPorData) %in% c("school", "Mjob", "Fjob", "reason", "guardian"))], encodedData)

# Convert numeric variables to numeric format
studentPorData$age <- as.numeric(studentPorData$age)
studentPorData$Medu <- as.numeric(studentPorData$Medu)
studentPorData$Fedu <- as.numeric(studentPorData$Fedu)
studentPorData$traveltime <- as.numeric(studentPorData$traveltime)
studentPorData$studytime <- as.numeric(studentPorData$studytime)
studentPorData$failures <- as.numeric(studentPorData$failures)
studentPorData$famrel <- as.numeric(studentPorData$famrel)
studentPorData$freetime <- as.numeric(studentPorData$freetime)
studentPorData$goout <- as.numeric(studentPorData$goout)
studentPorData$Dalc <- as.numeric(studentPorData$Dalc)
studentPorData$Walc <- as.numeric(studentPorData$Walc)
studentPorData$health <- as.numeric(studentPorData$health)
studentPorData$absences <- as.numeric(studentPorData$absences)
studentPorData$G1 <- as.numeric(studentPorData$G1)
studentPorData$G2 <- as.numeric(studentPorData$G2)
studentPorData$G3 <- as.numeric(studentPorData$G3)

# Select relevant features and the target variable
selectedFeatures <- c("sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2")
subsetPorData <- studentPorData[, c(selectedFeatures, "G3")]

# -------- Correlation Matrix -----------

# Selecting a subset of columns
subset_data <- studentPorData[, c("age", "Medu", "Fedu", "traveltime", "studytime", "failures", "G1", "G2", "G3")]

# Creating the correlation matrix
cor_matrix <- cor(subset_data, use = "complete.obs")

# Visualizing the correlation matrix 
corrplot(cor_matrix, method = "circle", type = "upper", tl.col = "black", tl.srt = 45)

# Compute correlation matrix
correlationMatrix <- cor(studentPorData[, c("G3", "G1", "G2", "studytime", "failures", "absences")])

# Convert correlation matrix to long format
correlationLong <- melt(correlationMatrix)

# Create a correlation heatmap
ggplot(data = correlationLong, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1))

# ---------------- Splitting the Data -----------

# Set the seed for reproducibility
set.seed(123)

# Splitting the data into training and test sets as 0.9 and 0.1
trainIndex <- createDataPartition(subsetPorData$G3, p = 0.9, list = FALSE)
trainData <- subsetPorData[trainIndex, ]
testData <- subsetPorData[-trainIndex, ]

# ---------------- Feature Selection with Random Forest -----------

# Feature selection using Random Forest
set.seed(123)

# Train a Random Forest model
rf_model <- randomForest(G3 ~ ., data = trainData, importance = TRUE)

# Extract variable importance scores
importance_scores <- importance(rf_model)

# Order features by importance
sorted_features <- importance_scores[order(importance_scores[, 1], decreasing = TRUE), ]

# Select the top N important features
top_n_features <- 10
selected_features <- rownames(sorted_features)[1:top_n_features]

# Print the selected features
print("Selected Features:")
print(selected_features)

# Subset the data with selected features
trainData_selected <- trainData[, c("G3", selected_features)]
testData_selected <- testData[, c("G3", selected_features)]

# Retrain models with selected features
linearModel_selected <- train(G3 ~ ., data = trainData_selected, method = "lm")
rfModel_selected <- train(G3 ~ ., data = trainData_selected, method = "rf")
gbmModel_selected <- train(G3 ~ ., data = trainData_selected, method = "gbm", verbose = FALSE)
knnModel_selected <- train(G3 ~ ., data = trainData_selected, method = "knn")
treeModel_selected <- train(G3 ~ ., data = trainData_selected, method = "rpart")

# Model Evaluation Function
evaluateModel <- function(model, testData) {
  predictions <- predict(model, testData)
  rmse <- sqrt(mean((testData$G3 - predictions)^2))
  return(rmse)
}

# Evaluate models with selected features
linearModelRMSE_selected <- evaluateModel(linearModel_selected, testData_selected)
rfModelRMSE_selected <- evaluateModel(rfModel_selected, testData_selected)
gbmModelRMSE_selected <- evaluateModel(gbmModel_selected, testData_selected)
knnModelRMSE_selected <- evaluateModel(knnModel_selected, testData_selected)
treeModelRMSE_selected <- evaluateModel(treeModel_selected, testData_selected)

# Print RMSEs for models with selected features
print(paste("Linear Model RMSE (Selected Features):", linearModelRMSE_selected))
print(paste("Random Forest Model RMSE (Selected Features):", rfModelRMSE_selected))
print(paste("Gradient Boosting Model RMSE (Selected Features):", gbmModelRMSE_selected))
print(paste("KNN Model RMSE (Selected Features):", knnModelRMSE_selected))
print(paste("Decision Tree Model RMSE (Selected Features):", treeModelRMSE_selected))

# Create a dataframe for results
results_selected <- data.frame(
  Model = c("Linear", "Random Forest", "Gradient Boosting", "KNN", "Decision Tree"),
  RMSE = c(linearModelRMSE_selected, rfModelRMSE_selected, gbmModelRMSE_selected, knnModelRMSE_selected, treeModelRMSE_selected)
)

# Plotting RMSE values
library(ggplot2)

ggplot(results_selected, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model RMSE Comparison with Selected Features", x = "Model", y = "RMSE") +
  theme_minimal()


# --------------- RFE Feature Selection ------------------------
# Load required libraries
library(caret)
library(randomForest)
library(gbm)

# Specify the model
knn_model <- train(G3 ~ ., data = trainData, method = "knn", trControl = trainControl(method = "cv", number = 5))

# Feature selection using Recursive Feature Elimination (RFE)
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
knn_rfe <- rfe(trainData[, -which(names(trainData) == "G3")], trainData$G3, sizes = c(1:ncol(trainData) - 1), rfeControl = ctrl, model = knn_model)

# Print the results
print("Results for KNN:")
print(knn_rfe)

# Plot feature ranking
plot(knn_rfe)

# Specify the model
rf_model <- train(G3 ~ ., data = trainData, method = "rf", trControl = trainControl(method = "cv", number = 5))

# Feature selection using Recursive Feature Elimination (RFE)
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
rf_rfe <- rfe(trainData[, -which(names(trainData) == "G3")], trainData$G3, sizes = c(1:ncol(trainData) - 1), rfeControl = ctrl, model = rf_model)

# Print the results
print("Results for Random Forest:")
print(rf_rfe)

# Plot feature ranking
plot(rf_rfe)

# Specify the model
lm_model <- train(G3 ~ ., data = trainData, method = "lm")

# Feature selection using Recursive Feature Elimination (RFE)
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
lm_rfe <- rfe(trainData[, -which(names(trainData) == "G3")], trainData$G3, sizes = c(1:ncol(trainData) - 1), rfeControl = ctrl, model = lm_model)

# Print the results
print("Results for Linear Regression:")
print(lm_rfe)

# Plot feature ranking
plot(lm_rfe)

# Specify the model
tree_model <- train(G3 ~ ., data = trainData, method = "rpart")

# Feature selection using Recursive Feature Elimination (RFE)
ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
tree_rfe <- rfe(trainData[, -which(names(trainData) == "G3")], trainData$G3, sizes = c(1:ncol(trainData) - 1), rfeControl = ctrl, model = tree_model)

# Print the results
print("Results for Decision Tree:")
print(tree_rfe)

# Plot feature ranking
plot(tree_rfe)

# Specify the models
models <- c("knn", "rf", "lm", "rpart", "gbm")

# Store the results for each model
model_results <- list()

for (model_name in models) {
  # Specify the model
  model_formula <- as.formula(paste("G3 ~ ."))
  if (model_name != "lm") {
    model_formula <- as.formula(paste("G3 ~ ."))
  }
  model <- train(model_formula, data = trainData, method = model_name, trControl = trainControl(method = "cv", number = 5))
  
  # Evaluate the model
  predictions <- predict(model, testData)
  rmse <- sqrt(mean((testData$G3 - predictions)^2))
  
  # Store the results
  model_results[[model_name]] <- list(model = model, rmse = rmse)
}

# Find the best model based on the lowest RMSE
best_model_name <- models[which.min(sapply(model_results, function(result) result$rmse))]
best_model <- model_results[[best_model_name]]$model

# Obtain the RMSE from the best model
best_model_rmse <- ifelse("trainRMSE" %in% names(best_model$finalModel),
                          best_model$finalModel$trainRMSE,
                          best_model$results$RMSE)

# Print the best model and its RMSE
print(paste("Best Model:", toupper(best_model_name)))
print(paste("Best Model RMSE:", best_model_rmse))

# Create a dataframe for results
models <- c("knn", "rf", "lm", "rpart", "gbm")
rmse_values <- sapply(models, function(model_name) model_results[[model_name]]$rmse)

results <- data.frame(Model = models, RMSE = rmse_values)

# Plotting RMSE values
library(ggplot2)

ggplot(results, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Model RMSE Comparison", x = "Model", y = "RMSE") +
  theme_minimal()



# ---------------- Feature Scaling ----------------

# Feature Scaling
preProcValues <- preProcess(trainData[, -which(names(trainData) == "G3")], method = c("center", "scale"))
trainDataScaled <- predict(preProcValues, trainData)
testDataScaled <- predict(preProcValues, testData)

# Build your models using the scaled data
linearModel <- train(G3 ~ ., data = trainDataScaled, method = "lm")
rfModel <- train(G3 ~ ., data = trainDataScaled, method = "rf")
gbmModel <- train(G3 ~ ., data = trainDataScaled, method = "gbm", verbose = FALSE)
knnModel <- train(G3 ~ ., data = trainDataScaled, method = "knn")
treeModel <- train(G3 ~ ., data = trainDataScaled, method = "rpart")

# Model Evaluation Function
evaluateModel <- function(model, testData) {
  predictions <- predict(model, testData)
  rmse <- sqrt(mean((testData$G3 - predictions)^2))
  return(rmse)
}


# Evaluate Models
linearModelRMSE <- evaluateModel(linearModel, testDataScaled)
rfModelRMSE <- evaluateModel(rfModel, testDataScaled)
gbmModelRMSE <- evaluateModel(gbmModel, testDataScaled)
knnModelRMSE <- evaluateModel(knnModel, testDataScaled)
treeModelRMSE <- evaluateModel(treeModel, testDataScaled)

# Print RMSEs
print(paste("Linear Model RMSE:", linearModelRMSE))
print(paste("Random Forest Model RMSE:", rfModelRMSE))
print(paste("Gradient Boosting Model RMSE:", gbmModelRMSE))
print(paste("KNN Model RMSE:", knnModelRMSE))
print(paste("Decision Tree Model RMSE:", treeModelRMSE))

# Create a dataframe to store RMSE values for each model
results <- data.frame(
  Model = c("Linear Model", "Random Forest Model", "Gradient Boosting Model", "KNN Model", "Decision Tree Model"),
  RMSE = c(linearModelRMSE, rfModelRMSE, gbmModelRMSE, knnModelRMSE, treeModelRMSE)
)

# Print the RMSE values
print(results)

# Find the model with the minimum RMSE
best_model <- results[which.min(results$RMSE), ]
print(paste("Best Model:", best_model$Model))
print(paste("Best Model RMSE:", best_model$RMSE))

# Load required libraries
library(ggplot2)

# Create a dataframe for results
models <- c("Linear Model", "Random Forest Model", "Gradient Boosting Model", "KNN Model", "Decision Tree Model")
rmse_values <- c(linearModelRMSE, rfModelRMSE, gbmModelRMSE, knnModelRMSE, treeModelRMSE)

results <- data.frame(Model = models, RMSE = rmse_values)

# Plotting RMSE values
ggplot(results, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "lightcoral") +
  labs(title = "Model RMSE Comparison (After Feature Scaling)", x = "Model", y = "RMSE") +
  theme_minimal()


# --------------Building the models so we can predict G3 values ---------------

# Building Linear Regression model
linearModel <- train(G3 ~ ., data = trainData, method = "lm")
# Print model summary
print(linearModel)

# Building Random Forest model
rfModel <- train(G3 ~ ., data = trainData, method = "rf")
# Print model summary
print(rfModel)

# Building Gradient Boosting model
gbmModel <- train(G3 ~ ., data = trainData, method = "gbm", verbose = FALSE)
# Print model summary
print(gbmModel)

# Building KNN model
knnModel <- train(G3 ~ ., data = trainData, method = "knn")
# Print model summary
print(knnModel)

# Building Decision Tree model
treeModel <- train(G3 ~ ., data = trainData, method = "rpart")
# Print model summary
print(treeModel)


# ------- Evaluating the models we built -----------------

# Feature Scaling
preProcValues <- preProcess(trainData[, -which(names(trainData) == "G3")], method = c("center", "scale"))
trainDataScaled <- predict(preProcValues, trainData)
testDataScaled <- predict(preProcValues, testData)

# Model Evaluation Function
evaluateModel <- function(model, testData) {
  predictions <- predict(model, testData)
  rmse <- sqrt(mean((testData$G3 - predictions)^2))
  return(rmse)
}

# Evaluate Models
linearModelRMSE <- evaluateModel(linearModel, testDataScaled)
rfModelRMSE <- evaluateModel(rfModel, testDataScaled)
gbmModelRMSE <- evaluateModel(gbmModel, testDataScaled)
knnModelRMSE <- evaluateModel(knnModel, testDataScaled)
#adaboostModelRMSE <- evaluateModel(adaboostModel, testDataScaled)
treeModelRMSE <- evaluateModel(treeModel, testDataScaled)

# Print RMSEs
print(paste("Linear Model RMSE:", linearModelRMSE))
print(paste("Random Forest Model RMSE:", rfModelRMSE))
print(paste("Gradient Boosting Model RMSE:", gbmModelRMSE))
print(paste("KNN Model RMSE:", knnModelRMSE))
#print(paste("AdaBoost Model RMSE:", adaboostModelRMSE))
print(paste("Decision Tree Model RMSE:", treeModelRMSE))


# ---------- The RMSE values are too high we will try to decrease it with tuning --------------

# ----------- Random Forest -----------------
tuneGrid_rf <- expand.grid(mtry = c(2, 4, 6, 8, 10))
# Perform tuning for Random Forest
set.seed(123)
rf_tuned <- train(
  G3 ~ .,
  data = trainData,
  method = "rf",
  tuneGrid = tuneGrid_rf,
  trControl = trainControl(method = "cv", number = 5)
)
# Print the best parameters for Random Forest
print(rf_tuned)
# Build the final Random Forest model with the best mtry value
final_rf_model <- randomForest(
  G3 ~ .,
  data = trainData,
  mtry = 10,
  importance = TRUE
)
# Predict on the test data
predictions <- predict(final_rf_model, testData)
# Calculate RMSE on the test data
rmse <- sqrt(mean((predictions - testData$G3)^2))
print(paste("Final Random Forest Model RMSE on Test Data:", rmse))

# ---------- Linear Regression -------
# Using caret here
control <- trainControl(method="cv", number=5)
tuned_lm <- train(G3 ~ ., data=trainData, method="lm", trControl=control)
print(tuned_lm)

# -------- Gradient Boosting -------
# Setting up grid for tuning
grid_gbm <- expand.grid(
  n.trees = c(50, 100, 150),
  interaction.depth = c(1, 3, 5),
  shrinkage = c(0.01, 0.1, 0.2),
  n.minobsinnode = 10
)

control <- trainControl(method="cv", number=5)
tuned_gbm <- train(G3 ~ ., data=trainData, method="gbm", trControl=control, tuneGrid=grid_gbm)
print(tuned_gbm)

# ----------- KNN ----------
# Setting up grid for tuning
grid_knn <- expand.grid(k = c(3, 5, 7, 9, 11))

control <- trainControl(method="cv", number=5)
tuned_knn <- train(G3 ~ ., data=trainData, method="knn", trControl=control, tuneGrid=grid_knn)
print(tuned_knn)

# ---------- Decision Tree --------
# Setting up grid for tuning
grid_tree <- expand.grid(
  cp = seq(0.01, 0.05, by = 0.01)
)

control <- trainControl(method="cv", number=5)
tuned_tree <- train(G3 ~ ., data=trainData, method="rpart", trControl=control, tuneGrid=grid_tree)
print(tuned_tree)

residual_plots <- lapply(list(linearModel, rfModel, gbmModel, knnModel, treeModel), function(model) {
  predictions <- predict(model, testDataScaled)
  residuals <- testData$G3 - predictions
  ggplot(data.frame(Fitted = predictions, Residuals = residuals), aes(x = Fitted, y = Residuals)) +
    geom_point() +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    ggtitle(paste("Residuals vs Fitted for", model$method))
})

do.call(grid.arrange, c(residual_plots, ncol = 3))

# Barplot for Before and After Tuning
tuning_results <- data.frame(
  Model = c("Random Forest", "Linear Regression", "Gradient Boosting", "KNN", "Decision Tree"),
  Before = c(rfModelRMSE, linearModelRMSE, gbmModelRMSE, knnModelRMSE, treeModelRMSE),
  After = c(rf_rmse, lm_rmse, gbm_rmse, knn_rmse, tree_rmse)
)

ggplot(tuning_results, aes(x = Model)) +
  geom_bar(aes(y = Before, fill = "Before Tuning"), stat = "identity", position = "dodge") +
  geom_bar(aes(y = After, fill = "After Tuning"), stat = "identity", position = "dodge") +
  labs(title = "RMSE Comparison Before and After Tuning", y = "RMSE") +
  theme_minimal() +
  scale_fill_manual(values = c("Before Tuning" = "skyblue", "After Tuning" = "orange"))


scatter_plots <- lapply(list(linearModel, rfModel, gbmModel, knnModel, treeModel), function(model) {
  predictions <- predict(model, testDataScaled)
  ggplot(data.frame(Actual = testData$G3, Predicted = predictions), aes(x = Actual, y = Predicted)) +
    geom_point() +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") + # Diagonal line for reference
    ggtitle(paste("Actual vs Predicted G3 for", model$method)) +
    xlab("Actual G3") + ylab("Predicted G3")
})

# Displaying the scatter plots
do.call(grid.arrange, c(scatter_plots, ncol = 3))

# ----------- BARPLOT ---------------

# Calculate residuals for each model
linear_residuals <- testData$G3 - predict(linearModel, testDataScaled)
rf_residuals <- testData$G3 - predict(rfModel, testDataScaled)
gbm_residuals <- testData$G3 - predict(gbmModel, testDataScaled)
knn_residuals <- testData$G3 - predict(knnModel, testDataScaled)
tree_residuals <- testData$G3 - predict(treeModel, testDataScaled)

# Create a data frame for residuals
residual_data <- data.frame(
  Model = rep(c("Linear", "Random Forest", "Gradient Boosting", "KNN", "Decision Tree"), each = nrow(testData)),
  Residuals = c(linear_residuals, rf_residuals, gbm_residuals, knn_residuals, tree_residuals)
)

# Create a bar plot for residuals

ggplot(residual_data, aes(x = Model, y = Residuals)) +
  geom_boxplot(fill = "skyblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +  # Reference line at 0
  ggtitle("Distribution of Residuals by Model") +
  xlab("Model") + ylab("Residuals") +
  theme_minimal()

# -------------- Selecting the Best Model -------------------

# Computing RMSE for Random Forest
rf_predictions <- predict(rf_tuned, testData)
rf_rmse <- sqrt(mean((rf_predictions - testData$G3)^2))
print(paste("RMSE for Random Forest:", rf_rmse))

# For Linear Regression
lm_predictions <- predict(tuned_lm, testData)
lm_rmse <- sqrt(mean((lm_predictions - testData$G3)^2))
print(paste("RMSE for Linear Regression:", lm_rmse))

# For Gradient Boosting
gbm_predictions <- predict(tuned_gbm, testData)
gbm_rmse <- sqrt(mean((gbm_predictions - testData$G3)^2))
print(paste("RMSE for Gradient Boosting:", gbm_rmse))

# For KNN
knn_predictions <- predict(tuned_knn, testData)
knn_rmse <- sqrt(mean((knn_predictions - testData$G3)^2))
print(paste("RMSE for KNN:", knn_rmse))

# For Decision Tree
tree_predictions <- predict(tuned_tree, testData)
tree_rmse <- sqrt(mean((tree_predictions - testData$G3)^2))
print(paste("RMSE for Decision Tree:", tree_rmse))


# Create a dataframe to display the RMSE values for each model we built
results <- data.frame(
  Model = c("Random Forest", "Linear Regression", "Gradient Boosting", "KNN", "Decision Tree"),
  RMSE = c(rf_rmse, lm_rmse, gbm_rmse, knn_rmse, tree_rmse)
)

# Printing the results
print(results)

# Selecting the best model
best_model <- results[which.min(results$RMSE), ]
print(paste("The best model is:", best_model$Model))

# Creating a barplot from RMSE values of each model
ggplot(results, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.6) +
  labs(title = "Model RMSE Comparison", x = "Model", y = "RMSE") +
  theme_minimal()


