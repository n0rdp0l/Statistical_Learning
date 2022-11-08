### Felix Wente S3337731
set.seed(3337731)
library(Rfast)
library(class)
library(caret)
library(glmnet)
#### Part A. - Supervised Learning

## Crating Test and Train Data Set
library(readr)
data_set <- read_csv("Data3337731.csv")
train <- data_set[1:5000, ]
test <- data_set[5001:10000, ]

### Questions

### 3.

## a.

X_3_tr <- train[,2:7] # design matrix for question 3
X_3_te <- test[,2:7]

Y_tr <- train$Y
Y_te <- test$Y
#normalize date
X_3_tr_norm <- as.data.frame(lapply(X_3_tr, function(x) {
  (x - min(x)) / (max(x) - min(x))
}))

X_3_te_norm <- as.data.frame(lapply(X_3_te, function(x) {
  (x - min(x)) / (max(x) - min(x))
}))


k <- c(1:30) # number of neighbors to consider
folds <- 10 # number of folds

#cross validation to determine k
knn_cv <- Rfast::knn.cv(nfolds = folds, y = Y_tr, x = as.matrix(X_3_tr_norm), k = k, stratified = T)

plot(k, knn_cv$crit, type="b", bty ="l", ylab = "% of correct classification ",
     col=ifelse(k==which.max(knn_cv$crit), "red", "black"))

k_cv <- which.max(knn_cv$crit)
print(paste("optimal k:",k_cv))


# predict on the test set to determine accuracy/error rate
knn.pred <- class::knn(as.matrix(X_3_tr_norm), as.matrix(X_3_te_norm),Y_te , k = k_cv)
cm_knn <- caret::confusionMatrix(knn.pred, as.factor(Y_te))
cm_knn

## b.

#10-fold CV on training data to determine lambda 
lasso_cv = cv.glmnet( as.matrix(X_3_tr_norm)  , as.matrix(Y_tr) , alpha=1 , nfolds = folds ) 

plot(lasso_cv$lambda,lasso_cv$cvm , type="b", bty ="l", ylab = "mean cross-validated error ",
     xlab = "lambda",col=ifelse(lasso_cv$lambda==lasso_cv$lambda.min, "red", "black"))

# create model with optimal lambda to predict values 
lasso_model = glmnet( as.matrix(X_3_tr_norm) , as.matrix(Y_tr) , alpha=1 , lambda=lasso_cv$lambda.min )
predict_lasso_train = predict( lasso_model , as.matrix(X_3_te_norm), type = "response")
predict_lasso_train = ifelse(predict_lasso_train>0.5,1,0)
cm_lasso <- caret::confusionMatrix(as.factor(predict_lasso_train), as.factor(Y_te))
cm_lasso

### 4.

#### Part B. - Unsupervised learning

## import data
Coping <- read_csv("Coping.csv")


