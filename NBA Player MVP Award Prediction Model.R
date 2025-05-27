install.packages(c("vscDebugger", "R6", "jsonlite"))

# Install the necessary packages 
install.packages("ggplot2")
install.packages("reshape2")
install.packages("glmnet")
install.packages("pROC")
install.packages("rmarkdown")

#read the data set from the folder
nba_data <- read.csv("C:/Users/hengl/OneDrive/Computer Science/Data Mining/nba player stats.csv")


#View the first 5 rows of the dataset
head(nba_data)

# View the structure of the dataset
str(nba_data)

# View the summary statistics of the dataset
summary(nba_data)



# Pre-processing techniques.

# Check for missing values. In this case there are no missing values
any(is.na(nba_data))


#Spliting data set into train test set

#Spliting the data set into train test split where 70% of dataset is the training set and 30% is the test set
set.seed(1)

sample <- sample(c(TRUE, FALSE), nrow(nba_data), replace = TRUE, prob = c(0.7, 0.3))
train <- nba_data[sample, ]
test <- nba_data[!sample, ]

# Split the training data into features (X) and target variable (y)
X_train <- train[, selected_features]
y_train <- train$MVP

# Print the dimensions of the train dataset
cat("Train dataset dimensions:", dim(train), "\n")

# Print the dimensions of the test dataset
cat("Test dataset dimensions:", dim(test), "\n")




#Normalization

# Normalize features using min-max scaling
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

X_train_normalized <- as.data.frame(lapply(X_train, normalize))

summary(X_train_normalized)



#Explanatory Data Analysis

#Remove League variable from the data set as it does not indicate anything to do with a player's performance
nba_data <- nba_data[, -4]
dim(nba_data)


# Select the necessary features for analysis which contributes to a player's chances of winning MVP 
selected_features <- c("Points", "Rebounds", "Assists", "Steals", "Blocks", "Minutes.Played", "Games.Played", "MVP")
nba_data <- nba_data[, selected_features]



#Pie Chart

# Calculate the count of MVP and non-MVP seasons
mvp_counts <- sum(nba_data$MVP == 1)
non_mvp_counts <- sum(nba_data$MVP == 0)

# Create a vector of counts
counts <- c(MVP = mvp_counts, Non_MVP = non_mvp_counts)

# Create a pie chart which illustrate the count of MVP and non-MVP seasons
pie(counts, labels = paste(names(counts), ": ", counts), col = c("red", "blue"), main = "Distribution of MVP and Non-MVP Seasons")



#Histogram

# Plot histogram for Points statistic for MVP seasons
hist(mvp_data$Points, main = "Distribution of Points Per Game (PPG) for MVP and Non-MVP Seasons", 
     xlab = "Points Per Game (PPG)", ylab = "Frequency", col = rgb(1, 0, 0, alpha = 0.5), 
     ylim = c(0, 20), xlim = c(5, 40))

# Add histogram for Points statistic for non-MVP seasons
hist(non_mvp_data$Points, col = rgb(0, 0, 1, alpha = 0.5), add = TRUE)

# Add legend
legend("topright", legend = c("MVP", "Non-MVP"), fill = c(rgb(1, 0, 0, alpha = 0.5), 
                                                          rgb(0, 0, 1, alpha = 0.5)))



#Box Plot

# Create boxplots for each statistic grouped by MVP status
par(mfrow=c(2,3)) # Arrange plots in a grid of 2 rows and 3 columns
boxplot(Points ~ MVP, data = nba_data, main = "Points Per Game (PPG) by MVP Status", xlab = "MVP Status", ylab = "Points Per Game (PPG)", col = c("blue", "red"))
boxplot(Rebounds ~ MVP, data = nba_data, main = "Rebounds by MVP Status", xlab = "MVP Status", ylab = "Rebounds", col = c("blue", "red"))
boxplot(Assists ~ MVP, data = nba_data, main = "Assists by MVP Status", xlab = "MVP Status", ylab = "Assists", col = c("blue", "red"))
boxplot(Steals ~ MVP, data = nba_data, main = "Steals by MVP Status", xlab = "MVP Status", ylab = "Steals", col = c("blue", "red"))
boxplot(Blocks ~ MVP, data = nba_data, main = "Blocks by MVP Status", xlab = "MVP Status", ylab = "Blocks", col = c("blue", "red"))
boxplot(Minutes.Played ~ MVP, data = nba_data, main = "Minutes Played by MVP Status", xlab = "MVP Status", ylab = "Minutes Played", col = c("blue", "red"))
boxplot(Games.Played ~ MVP, data = nba_data, main = "Games Played by MVP Status", xlab = "MVP Status", ylab = "Games Played", col = c("blue", "red"))




#Scatter Plot

# Create a jittered scatter plot of Points per Game (PPG) vs MVP Status
plot(jitter(nba_data$Points), nba_data$MVP, 
     main = "Points Per Game (PPG) vs. MVP Status", 
     xlab = "Points Per Game (PPG)", 
     ylab = "MVP Status (0 = Not MVP, 1 = MVP)", 
     col = ifelse(nba_data$MVP == 1, "red", "blue"), 
     pch = 19)

# Create a jittered scatter plot of Rebounds Per Game (PPG) vs. MVP Status
plot(jitter(nba_data$Rebounds), nba_data$MVP, 
     main = "Rebounds Per Game (PPG) vs. MVP Status", 
     xlab = "Rebounds Per Game (PPG)", 
     ylab = "MVP Status (0 = Not MVP, 1 = MVP)", 
     col = ifelse(nba_data$MVP == 1, "red", "blue"), 
     pch = 19)

# Create a jittered scatter plot of Assists Per Game (PPG) vs. MVP Status
plot(jitter(nba_data$Assists), nba_data$MVP, 
     main = "Assists Per Game (PPG) vs. MVP Status", 
     xlab = "Assists Per Game (PPG)", 
     ylab = "MVP Status (0 = Not MVP, 1 = MVP)", 
     col = ifelse(nba_data$MVP == 1, "red", "blue"), 
     pch = 19)

# Create a jittered scatter plot of Blocks Per Game (PPG) vs. MVP Status
plot(jitter(nba_data$Blocks), nba_data$MVP, 
     main = "Blocks Per Game (PPG) vs. MVP Status", 
     xlab = "Blocks Per Game (PPG)", 
     ylab = "MVP Status (0 = Not MVP, 1 = MVP)", 
     col = ifelse(nba_data$MVP == 1, "red", "blue"), 
     pch = 19)


# Create a jittered scatter plot of Steals Per Game (PPG) vs. MVP Status
plot(jitter(nba_data$Steals), nba_data$MVP, 
     main = "Steals Per Game (PPG) vs. MVP Status", 
     xlab = "Steals Per Game (PPG)", 
     ylab = "MVP Status (0 = Not MVP, 1 = MVP)", 
     col = ifelse(nba_data$MVP == 1, "red", "blue"), 
     pch = 19)

# Create a jittered scatter plot of Minutes Played Per Game (PPG) vs. MVP Status
plot(jitter(nba_data$Minutes.Played), nba_data$MVP, 
     main = "Minutes Played Per Game vs. MVP Status", 
     xlab = "Minutes Played Per Game", 
     ylab = "MVP Status (0 = Not MVP, 1 = MVP)", 
     col = ifelse(nba_data$MVP == 1, "red", "blue"), 
     pch = 19)

# Create a jittered scatter plot of Minutes Played Per Game (PPG) vs. MVP Status
plot(jitter(nba_data$Games.Played), nba_data$MVP, 
     main = "Games Played Per Game  vs. MVP Status", 
     xlab = "Games Played Per Game", 
     ylab = "MVP Status (0 = Not MVP, 1 = MVP)", 
     col = ifelse(nba_data$MVP == 1, "red", "blue"), 
     pch = 19)


#Correlation Matrix


# Compute correlation matrix
correlation_matrix <- cor(nba_data[, c("Points", "Rebounds", "Assists", "Steals", "Blocks", "Minutes.Played", "Games.Played")])

# Activate library
library(ggplot2)
library(reshape2)

# Convert correlation matrix to long format
correlation_melted <- melt(correlation_matrix)

# Plot heatmap
ggplot(correlation_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0,
                       name = "Correlation",
                       limits = c(-1, 1),
                       breaks = seq(-1, 1, by = 0.2)) +
  theme_minimal() +
  labs(title = "Correlation Matrix Heatmap of NBA Statistics",
       x = "Statistic",
       y = "Statistic")



#Stacked Bar Plot of summary statistics for MVP and non-MVP seasons

# Calculate summary statistics for MVP and non-MVP seasons
mvp_summary <- colMeans(mvp_data[, c("Points", "Rebounds", "Assists", "Steals", "Blocks", "Minutes.Played", "Games.Played")])
non_mvp_summary <- colMeans(non_mvp_data[, c("Points", "Rebounds", "Assists", "Steals", "Blocks", "Minutes.Played", "Games.Played")])


# Create a matrix with summary statistics for both MVP and non-MVP seasons
summary_matrix <- rbind(mvp_summary, non_mvp_summary)

# Define the statistics names
stat_names <- names(mvp_summary)

# Plot the stacked bar plot
barplot(summary_matrix, beside = TRUE, 
        legend.text = c("MVP", "Non-MVP"), 
        args.legend = list(x = "topleft"), 
        col = c("red", "blue"),
        ylim = c(0, 90),
        main = "Contribution of Statistics to MVP and Non-MVP Seasons",
        xlab = "Statistics",
        ylab = "Average Value")

# Add legend
legend("topleft", legend = c("MVP", "Non-MVP"), fill = c("red", "blue"))




#Model Training

# Train logistic regression model
logit_model <- glm(MVP ~ ., data = train, family = "binomial")

# Print model summary
summary(logit_model)


#Using Logistic Regression Model for prediction

# Predict probabilities for the test set
probabilities <- predict(logit_model, newdata = test, type = "response")

# Set threshold for binary classification
threshold <- 0.5

# Convert probabilities to binary predictions (0 or 1) using the threshold
predictions <- ifelse(probabilities > threshold, 1, 0)


# Predictions on the test set
predictions_df <- data.frame(Actual = test$MVP, Predicted = predictions)
print(predictions_df)


# Interpretation of coefficients
coefficients <- coef(logit_model)
print(coefficients)


# Visualize coefficients
barplot(coefficients[-1], names.arg = names(coefficients[-1]), 
        main = "Coefficients of Logistic Regression Model",
        xlab = "Features", ylab = "Coefficient Value", ylim = c(-15, 10), las = 2)




#Model Evaluation


# Evaluate model performance using confusion matrix
conf_matrix <- table(test$MVP, predictions)
print(conf_matrix)


# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))


# Calculate precision
precision <- conf_matrix[2, 2] / sum(predictions)
print(paste("Precision:", precision))


# Calculate recall (sensitivity)
recall <- conf_matrix[2, 2] / sum(test$MVP)
print(paste("Recall:", recall))


# Calculate F1-score
f1_score <- 2 * precision * recall / (precision + recall)
print(paste("F1-score:", f1_score))


# Plot ROC curve and calculate AUC
library(pROC)
roc_curve <- roc(test$MVP, probabilities)
plot(roc_curve, main = "ROC Curve")
auc <- auc(roc_curve)
print(paste("AUC:", auc))








