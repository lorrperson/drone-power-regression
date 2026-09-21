# ISYE 6414 Regression Analysis- Drone FInal Project
remove.packages("dplyr")
install.packages("dplyr")
library(dplyr)
library(readr)

# Read Data
flight_data <- read.csv("../data/raw/flights.csv")

flight_avg <- flight_data %>%
  mutate(power_consumption = battery_voltage * battery_current) %>%
  group_by(flight) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))

# Write and export "flight_avgs.csv"
write_csv(flight_avg, "../data/processed/flight_avgs.csv")

#Power Test and Sample Size (to see how many observations we should take into account)
install.packages("pwr")
library(pwr)

# f2= effect size: .02= small, .15 = medium, .35=large

pwr.f2.test(
  u = 10,        # number of predictors I plan to use
  v = 209-10-1,  #residual degress of freedom(n - predictors- 1)
  f2 = .15,      #medium effect size
  sig.level = .05 
)

# You can also solve for minimum sample size:
pwr.f2.test(
  u = 10,
  f2 = .15,
  sig.level = .05,
  power = .80
)

# install and load packages
install.packages("corrplot")
library(corrplot)
library(readr)
library(dplyr)

# Read Data
avgflight_data <- read.csv("../data/processed/flight_avgs.csv")

# Check dimensions and structure
dim(avgflight_data)
str(avgflight_data)
summary(avgflight_data)

#Check missing values
colSums(is.na(avgflight_data))    #if column shows >0 then there is missing data

# Remove useless predictors (current and voltage)
avgflight_data_clean <- avgflight_data %>%
  select(-flight, -time, -position_x, -position_y, -position_z, -battery_voltage, -battery_current)

# Remove flights with power consumption below 200 watts (clear outliers)
avgflight_data_clean <- avgflight_data_clean[avgflight_data_clean$power_consumption > 200, ]

# See what is left
names(avgflight_data_clean)

#----------
# Exploratory Visualizations
#----------
#Summary Stat of response variable
hist(avgflight_data_clean$power_consumption,
     main = "Distribution of Power Consumption",
     xlab = "Power (Watts)",
     col = "#B3A369",
     border = "#003057",
     breaks = 15)

# Descrptive Statistics
summary(avgflight_data)

# Correlation Matrix
cor_matrix <- cor(avgflight_data_clean, use = "complete.obs")
print(round(cor_matrix, 2))

# GT-themed Correlation Plot
buzzgold <- "#B3A369"
gtblue <- "#003057"
col <- colorRampPalette(c(gtblue, "white", buzzgold))(200)

corrplot(cor_matrix, method = "number", type = "upper",
         tl.cex = 0.7, tl.col = "black",
         number.cex = 0.5,
         col = col)

# Boxplot for most important independent variable ~ paylod (categorical data=boxplot)
boxplot(power_consumption ~ as.factor(payload), 
        data = avgflight_data_clean,
        main = "Power vs. Payload",
        xlab = "Payload (grams)",
        ylab = "Power Consumption (Watts)",
        col = c("#B3A369", "white", "#003057"), # Gold, White, Blue
        border = "black")
table(avgflight_data_clean$payload)
# Scatter plots: power vsblinear_acceleration_y , wind_speed (Continuous) + trendline
plot(avgflight_data_clean$linear_acceleration_y, avgflight_data_clean$power_consumption,
     main = "Power vs. Y-Acceleration",
     xlab = "Linear Acceleration Y (m/s^2)",
     ylab = "Power Consumption (Watts)",
     pch = 19, # Solid dots
     col = rgb(0, 48, 87, 150, maxColorValue=255)) # Semi-transparent GT Blue

abline(lm(power_consumption ~ linear_acceleration_y, data = avgflight_data_clean), 
       col = "red", lwd = 3)

plot(avgflight_data_clean$wind_speed, avgflight_data_clean$power_consumption,
     main = "Power vs. Wind Speed",
     xlab = "Wind Speed (m/s)",
     ylab = "Power Consumption (Watts)",
     pch = 19, 
     col = rgb(179, 163, 105, 150, maxColorValue=255)) # Semi-transparent GT Gold

abline(lm(power_consumption ~ wind_speed, data = avgflight_data_clean), 
       col = "red", lwd = 3)

plot(avgflight_data_clean$wind_angle, avgflight_data_clean$power_consumption,
     main = "Power vs. Wind Angle",
     xlab = "Wind Angle (m/s)",
     ylab = "Power Consumption (Watts)",
     pch = 19, # Solid dots
     col = rgb(0, 48, 87, 150, maxColorValue=255)) # Semi-transparent GT Blue

abline(lm(power_consumption ~ wind_angle, data = avgflight_data_clean), 
       col = "red", lwd = 3)
# plot back to normal
par(mfrow = c(1, 1))

# Check which variables correlate most with power consumption
power_cors <- cor_matrix[, "power_consumption"]
power_cors_sorted <- sort(abs(power_cors), decreasing = TRUE)
print(round(power_cors_sorted, 3))

#-------------------
# LASSO Regression
#-------------------
install.packages("glmnet")
library(glmnet)

# Separate predictors (X) and response (Y)
y.train <- avgflight_data_clean$power_consumption
X.train <- as.matrix(avgflight_data_clean %>% select(-power_consumption))

# Run LASSO w/ cross-validation to find optimal lambda
set.seed(123)
cv_lasso <- cv.glmnet(X.train, y.train, alpha = 1)

# Plot cross validation results
plot(cv_lasso)

# Best lambda value
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Check lambda.1se simple model
cat("Lambda 1se:", cv_lasso$lambda.1se, "\n")

# Which variables LASSO kept (non-zero coefficients)
lasso_coefs <- coef(cv_lasso, s = "lambda.min")
print(lasso_coefs)

# Clean view: only show non-zero coefficients
lasso_df <- as.data.frame(as.matrix(lasso_coefs))
lasso_df$variable <- rownames(lasso_df)
names(lasso_df)[1] <- "coefficient"
lasso_df <- lasso_df[lasso_df$coefficient != 0, ]
print(lasso_df[order(abs(lasso_df$coefficient), decreasing = TRUE), ])

#--------------------------------
# Multi Linear Regression
#--------------------------------

# Fit regression using only the variables LASSO kept (excluding speed and linear_acceleration_x)
model <- lm(power_consumption ~ wind_speed + wind_angle + orientation_x +
              orientation_z + orientation_w + velocity_y + velocity_z +
              angular_x + angular_y + angular_z + linear_acceleration_y +
              linear_acceleration_z + speed + payload,
            data = avgflight_data_clean)

summary(model)

#--------------------------------
# DIAGNOSTIC TESTS
#--------------------------------

# Residual Plots -----
par(mfrow = c(2, 2))  # show 4 plots in one window
plot(model)
par(mfrow = c(1, 1))  # reset to single plot

#  Normality of Residuals (Shapiro-Wilk Test) -----
shapiro.test(resid(model))
# p-value > 0.05 = residuals are normal (good)
# p-value < 0.05 = residuals are NOT normal (problem)

# 4C: Homoscedasticity (Breusch-Pagan Test) -----
install.packages("lmtest")
library(lmtest)
bptest(model)
# p-value > 0.05 = constant variance (good)
# p-value < 0.05 = heteroscedasticity (problem)

# ----- 4D: Multicollinearity (VIF) -----
install.packages("car")
library(car)
vif(model)
# VIF > 10 = serious multicollinearity problem
# VIF > 5  = moderate concern
# VIF < 5  = acceptable

# ----- 4E: Model Performance Summary -----
cat("\n===== MODEL SUMMARY =====\n")
cat("R-squared:", summary(model)$r.squared, "\n")
cat("Adjusted R-squared:", summary(model)$adj.r.squared, "\n")
cat("F-statistic p-value:", pf(summary(model)$fstatistic[1],
                               summary(model)$fstatistic[2], summary(model)$fstatistic[3],
                               lower.tail = FALSE), "\n")
# Reduced model with only significant variables
model_reduced <- lm(power_consumption ~ wind_speed + wind_angle +
                      velocity_z + payload,
                    data = avgflight_data_clean)

summary(model_reduced)

# Compare models
anova(model_reduced, model)
# If p-value > 0.05, the reduced model is just as good (use the simpler one)

# Diagnostic plots
par(mfrow = c(2, 2))
plot(model_reduced)
par(mfrow = c(1, 1))

# Normality test
shapiro.test(resid(model_reduced))

# Homoscedasticity test
library(lmtest)
bptest(model_reduced)

# VIF
library(car)
vif(model_reduced)

# install packages and librarie
install.packages("sandwich")
library(sandwich) 
library(lmtest)
 
# Robust standard erros (HC1 = standard robut)
coeftest(model_reduced, vcov = vcovHC(model_reduced, type = "HC1"))

# ==========================================
# FINAL STEP: OUT-OF-SAMPLE VALIDATION
# ==========================================
install.packages("Metrics")
library(Metrics)

# 1. Create an 80/20 Train-Test Split on your aggregated data
set.seed(123) # For reproducibility
sample_size <- floor(0.8 * nrow(avgflight_data_clean))
train_idx <- sample(seq_len(nrow(avgflight_data_clean)), size = sample_size)

train_data <- avgflight_data_clean[train_idx, ]
test_data <- avgflight_data_clean[-train_idx, ]

# 2. Re-train your winning 4-variable model on ONLY the training data
final_validation_model <- lm(power_consumption ~ wind_speed + wind_angle + 
                              velocity_z + payload,
                             data = train_data)

# 3. Predict the power consumption for the unseen 20% of flights
test_predictions <- predict(final_validation_model, newdata = test_data)

# 4. Calculate Root Mean Square Error (RMSE)
model_rmse <- rmse(test_data$power_consumption, test_predictions)
cat("\nFinal Model Out-of-Sample RMSE:", round(model_rmse, 2), "Watts\n")

# 5. Plot Actual vs Predicted (The ultimate proof of a good model)
plot(test_data$power_consumption, test_predictions,
     main = "Model Validation: Actual vs. Predicted Power",
     xlab = "Actual Power Consumption (Watts)",
     ylab = "Predicted Power Consumption (Watts)",
     pch = 19, col = "#003057") # GT Blue
abline(0, 1, col = "#B3A369", lwd = 3) # GT Gold perfect prediction line

# Standardized coefficients for comparison
std_model <- lm(scale(power_consumption) ~ scale(wind_speed) + scale(wind_angle) + scale(velocity_z) 
                  + scale(payload),
                data = avgflight_data_clean)

std_coefs <- coef(std_model)[-1]
names(std_coefs) <- c("Wind Speed", "Wind Angle","Velocity Z", "Payload")
                      


# Retrain
final_validation_model <- lm(power_consumption ~ wind_speed + wind_angle + 
                               velocity_z + payload,
                             data = train_data)

# Predict and evaluate
test_predictions <- predict(final_validation_model, newdata = test_data)
model_rmse <- rmse(test_data$power_consumption, test_predictions)
cat("\nCleaned Model RMSE:", round(model_rmse, 2), "Watts\n")

# Plot
plot(test_data$power_consumption, test_predictions,
     main = paste("Cleaned Validation (RMSE:", round(model_rmse, 2), ")"),
     xlab = "Actual Power Consumption (Watts)",
     ylab = "Predicted Power Consumption (Watts)",
     pch = 19, col = "#003057")
abline(0, 1, col = "#B3A369", lwd = 3)
par(mar = c(6,9,2,3))

barplot(sort(abs(std_coefs), decreasing  = FALSE),
        main = "Standardized Coefficient Magnitude",
        col = "#003057",
        border = "#B3A369",
        las = 1,
        xlab = "Absolute Standardized Coefficient",
        horiz=TRUE)
par(mar = c(3,1,1,2))