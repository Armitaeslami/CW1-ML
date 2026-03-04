# Load libraries
library(data.table)

# Set seed 
set.seed(123)

# Import training data
trn <- fread('CW1_train.csv')
tst <- fread('CW1_test.csv') # This does not include true outcomes (obviously)

# Train your model (using a simple LM here as an example)
f <- lm(outcome ~ ., data = trn)

# Test set predictions
yhat_lm <- predict(f, tst)

# Format submission:
# This is a single-column CSV with nothing but your predictions
out <- data.table('yhat' = yhat_lm)
fwrite(out, 'CW1_submission_KNUMBER.csv') # Please use your k-number here

################################################################################

# At test time, we will use the true outcomes
tst <- fread('CW1_test_with_true_outcome.csv') # You do not have access to this

# This is the R^2 function
r2_fn <- function(yhat) {
  eps <- tst$outcome - yhat
  rss <- sum(eps^2)
  tss <- sum((tst$outcome - tst[, mean(outcome)])^2)
  r2 <- 1 - (rss / tss)
  return(r2)
}

# How does the linear model do?
r2_fn(yhat_lm)



