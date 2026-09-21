# ==============================================================================
# PROJECT: Drone Power Consumption Regression Analysis
# COURSE: ISYE6414 Group 2
# PURPOSE: Full Analytical Pipeline (Selection, Diagnostics, ANOVA, Validation)
# ==============================================================================

# --- 1. ENVIRONMENT SETUP ---
rm(list = ls())
if(!is.null(dev.list())) dev.off() 

# Required Libraries
packages <- c("readr", "car", "corrplot", "lmtest")
for (p in packages) {
  if (!require(p, character.only = TRUE)) install.packages(p, dependencies = TRUE)
  library(p, character.only = TRUE)
}

# --- 2. DATA IMPORT & PREP ---
df <- read_csv("../data/processed/drone_data.csv")

# FIX: Explicitly define numeric columns to prevent "Factor" doubling errors
numeric_cols <- c("total_power", "time", "altitude", "speed", "payload", 
                  "velocity_z", "linear_acceleration_z", "orientation_x", 
                  "orientation_y", "orientation_z", "wind_angle")

# Force numeric conversion and ensure flight_phase is the only categorical factor
df[numeric_cols] <- lapply(df[numeric_cols], function(x) as.numeric(as.character(x)))
df$flight_phase <- as.factor(df$flight_phase)

# Remove any rows with NAs (often caused by non-numeric text in numeric columns)
df <- na.omit(df)

# --- 3. EXPLORATORY ANALYSIS (Correlation Matrix) ---
cat("\n--- Generating Correlation Matrix ---\n")
cor_data <- cor(df[, numeric_cols], use = "complete.obs")

# FIX: Layout adjustments to prevent the "smooshed" look
dev.new(width = 12, height = 10)
corrplot(cor_data, 
         method = "color", 
         type = "upper", 
         tl.col = "black", 
         tl.srt = 45,            # Rotate labels for readability
         tl.cex = 0.8,           # Shrink label text
         addCoef.col = "black",  # Show correlation values
         number.cex = 0.6,       # Shrink value text
         diag = FALSE,           # Remove diagonal clutter
         mar = c(0, 0, 2, 0),    # Add top margin
         title = "Drone Telemetry Predictor Correlation")

# --- 4. TRAIN-TEST SPLIT ---
set.seed(42)
sample_size <- floor(0.8 * nrow(df))
train_idx <- sample(seq_len(nrow(df)), size = sample_size)
train_data <- df[train_idx, ]
test_data  <- df[-train_idx, ]

# --- 5. THE MODELS ---

# Model 1: The "Full" Physical Model
full_model <- lm(total_power ~ time + altitude + speed + payload + 
                   velocity_z + linear_acceleration_z + 
                   orientation_x + orientation_y + orientation_z + 
                   wind_angle + flight_phase, 
                 data = train_data)

print(summary(full_model))
# Model 2: The "Final" Optimized Model (Stepwise Selection)
final_model <- step(full_model, direction = "both", trace = FALSE)

# --- 6. STATISTICAL COMPARISON & DIAGNOSTICS ---
cat("\n--- ANOVA Model Comparison (Final vs Full) ---\n")
print(anova(final_model, full_model))

cat("\n--- Final Model Summary ---\n")
print(summary(final_model))

cat("\n--- Multicollinearity Check (VIF) ---\n")
if(length(coef(final_model)) > 2) {
  print(vif(final_model))
}

cat("\n--- Durbin-Watson (Autocorrelation) ---\n")
print(dwtest(final_model))

# --- 7. VISUAL DIAGNOSTICS (The Big Four) ---
dev.new(width = 12, height = 9)
par(mfrow = c(2, 2))
plot(final_model, main = "Final Model Diagnostics")

# --- 8. VALIDATION ON UNSEEN DATA ---
predictions <- predict(final_model, test_data)
rmse <- sqrt(mean((test_data$total_power - predictions)^2))

# Actual vs Predicted Plot
dev.new(width = 8, height = 6)
plot(test_data$total_power, predictions, 
     main = paste("Validation Results (RMSE:", round(rmse, 2), ")"),
     xlab = "Measured Power", 
     ylab = "Predicted Power",
     pch = 19, col = rgb(0.1, 0.4, 0.7, 0.4))
abline(0, 1, col = "red", lwd = 2) # Reference line
                 
