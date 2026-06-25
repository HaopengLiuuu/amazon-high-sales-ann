# =====================================================
# Predicting High-Monthly-Sales Amazon Products with an ANN
# -----------------------------------------------------
# Data (not bundled - see data/README.md): Amazon Products Dataset 2023 (Kaggle).
# Load before running:
#   amazon_products   <- read.csv("amazon_products.csv")
#   amazon_categories <- read.csv("amazon_categories.csv")
# =====================================================

# =====================================================
# 0) Required Packages
# =====================================================

library(neuralnet)
library(caret)
library(pROC)

# =====================================================
# STEP 1: Random Sampling
# =====================================================

set.seed(123)

amazon_sample <- amazon_products[
  sample(nrow(amazon_products), 100000),
]

# =====================================================
# STEP 2: Keep asin from the beginning
# =====================================================

amazon_clean <- amazon_sample[, c(
  "asin",
  "stars",
  "reviews",
  "price",
  "listPrice",
  "category_id",
  "isBestSeller",
  "boughtInLastMonth"
)]

# =====================================================
# STEP 3: Basic Cleaning
# =====================================================

amazon_clean <- amazon_clean[
  amazon_clean$price > 0 &
    amazon_clean$listPrice > 0 &
    amazon_clean$reviews >= 0 &
    amazon_clean$stars > 0,
]

amazon_clean <- na.omit(amazon_clean)

# =====================================================
# STEP 4: Define High_Sales (Top 30%)
# =====================================================

threshold <- quantile(amazon_clean$boughtInLastMonth, 0.70)

amazon_clean$High_Sales <- ifelse(
  amazon_clean$boughtInLastMonth > threshold,
  1,
  0
)

# =====================================================
# STEP 5: Feature Engineering
# =====================================================

amazon_clean$discount_rate <- 
  (amazon_clean$listPrice - amazon_clean$price) /
  amazon_clean$listPrice

amazon_clean$log_reviews <- log1p(amazon_clean$reviews)

# Preserve ID and category before modeling
product_ids <- amazon_clean$asin
product_category <- amazon_clean$category_id

# Remove leakage and unused variables
amazon_clean$asin <- NULL
amazon_clean$boughtInLastMonth <- NULL
amazon_clean$reviews <- NULL
amazon_clean$listPrice <- NULL

amazon_clean$category_id <- as.factor(amazon_clean$category_id)

# =====================================================
# STEP 6: One-hot Encoding
# =====================================================

dummy <- dummyVars(High_Sales ~ ., data = amazon_clean)

amazon_model <- as.data.frame(
  predict(dummy, newdata = amazon_clean)
)

amazon_model$High_Sales <- amazon_clean$High_Sales

# =====================================================
# STEP 7: Standardization
# =====================================================

x_cols <- setdiff(colnames(amazon_model), "High_Sales")

preproc <- preProcess(
  amazon_model[, x_cols],
  method = c("center", "scale")
)

amazon_model[, x_cols] <- predict(preproc, amazon_model[, x_cols])

# =====================================================
# STEP 8: Train/Test Split
# =====================================================

set.seed(42)

train_index <- sample(
  nrow(amazon_model),
  0.7 * nrow(amazon_model)
)

train <- amazon_model[train_index, ]
test  <- amazon_model[-train_index, ]

test_ids <- product_ids[-train_index]
test_category <- product_category[-train_index]

# =====================================================
# STEP 9: Train ANN
# =====================================================

formula_nn <- as.formula(
  paste("High_Sales ~", paste(x_cols, collapse = " + "))
)

nn_model <- neuralnet(
  formula_nn,
  data = train,
  hidden = 3,
  linear.output = FALSE
)

# =====================================================
# STEP 10: Prediction (No probability displayed)
# =====================================================

pred_prob <- neuralnet::compute(nn_model, test[, x_cols])$net.result
pred_class <- ifelse(pred_prob > 0.5, 1, 0)

# =====================================================
# STEP 11: Evaluation
# =====================================================

# Confusion matrix
conf_matrix <- table(
  Predicted = pred_class,
  Actual = test$High_Sales
)

print(conf_matrix)

# Extract values
TP <- conf_matrix["1", "1"]
TN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]

# Accuracy
accuracy <- (TP + TN) / sum(conf_matrix)

# Precision
precision <- TP / (TP + FP)

# Recall (Sensitivity)
recall <- TP / (TP + FN)

# F1 Score
F1 <- 2 * (precision * recall) / (precision + recall)

# F0.5 Score (Precision weighted higher)
beta <- 0.5
F05 <- (1 + beta^2) * (precision * recall) /
  ((beta^2 * precision) + recall)

# AUC
library(pROC)
roc_obj <- roc(test$High_Sales, as.numeric(pred_prob))
AUC <- auc(roc_obj)

# Print results
cat("Accuracy:", round(accuracy, 4), "\n")
cat("Precision:", round(precision, 4), "\n")
cat("Recall:", round(recall, 4), "\n")
cat("F0.5:", round(F05, 4), "\n")
cat("F1:", round(F1, 4), "\n")
cat("AUC:", round(AUC, 4), "\n")

# =====================================================
# STEP 12: Output Predicted High-Sales Products
# =====================================================

results <- data.frame(
  asin = test_ids,
  category_id = test_category,
  Predicted = pred_class
)

high_sales_products <- results[
  results$Predicted == 1,
]

# =====================================================
# STEP 13: Merge Category Name
# =====================================================

# Make sure types match
high_sales_products$category_id <- as.numeric(high_sales_products$category_id)
amazon_categories$id <- as.numeric(amazon_categories$id)

# Merge using correct column names
high_sales_products <- merge(
  high_sales_products,
  amazon_categories,
  by.x = "category_id",
  by.y = "id",
  all.x = TRUE
)

head(high_sales_products)

# Optional export
write.csv(
  high_sales_products,
  "predicted_high_sales_products_with_category.csv",
  row.names = FALSE
)
