### Felix Wente S3337731
set.seed(3337731)
library(Rfast)
library(class)
library(caret)
library(glmnet)
library(factoextra)
library(plotly)
library(cluster)
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


k <- c(1:100) # number of neighbors to consider
folds <- 10 # number of folds

#cross validation to determine k
set.seed(3337731)
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
set.seed(3337731)
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

## a.

X_4_tr <- train[,2:204] # design matrix for question 3
X_4_te <- test[,2:204]

Y_tr <- train$Y
Y_te <- test$Y
#normalize date
X_4_tr_norm <- as.data.frame(lapply(X_4_tr, function(x) {
  (x - min(x)) / (max(x) - min(x))
}))

X_4_te_norm <- as.data.frame(lapply(X_4_te, function(x) {
  (x - min(x)) / (max(x) - min(x))
}))


k <- c(1:100) # number of neighbors to consider
folds <- 10 # number of folds

#cross validation to determine k
set.seed(3337731)
knn_cv2 <- Rfast::knn.cv(nfolds = folds, y = Y_tr, x = as.matrix(X_4_tr_norm), k = k, stratified = T)

plot(k, knn_cv2$crit, type="b", bty ="l", ylab = "% of correct classification ",
     col=ifelse(k==which.max(knn_cv2$crit), "red", "black"))

k_cv2 <- which.max(knn_cv2$crit)
print(paste("optimal k:",k_cv2))


# predict on the test set to determine accuracy/error rate
knn.pred2 <- class::knn(as.matrix(X_4_tr_norm), as.matrix(X_4_te_norm),Y_te , k = k_cv2)
cm_knn2 <- caret::confusionMatrix(knn.pred2, as.factor(Y_te))
cm_knn2

## b.



#10-fold CV on training data to determine lambda 
set.seed(3337731)
lasso_cv2 = cv.glmnet( as.matrix(X_4_tr_norm)  , as.matrix(Y_tr) , alpha=1 , nfolds = folds ) 

plot(lasso_cv2$lambda,lasso_cv2$cvm , type="b", bty ="l", ylab = "mean cross-validated error ",
     xlab = "lambda",col=ifelse(lasso_cv2$lambda==lasso_cv2$lambda.min, "red", "black"))

# create model with optimal lambda to predict values 
lasso_model2 = glmnet( as.matrix(X_4_tr_norm) , as.matrix(Y_tr) , alpha=1 , lambda=lasso_cv2$lambda.min )
predict_lasso_train2 = predict( lasso_model2 , as.matrix(X_4_te_norm), type = "response")
predict_lasso_train2 = ifelse(predict_lasso_train2>0.5,1,0)
cm_lasso2 <- caret::confusionMatrix(as.factor(predict_lasso_train2), as.factor(Y_te))
cm_lasso2

#### Part B. - Unsupervised learning

## import data
Coping <- read_csv("Coping.csv")

## 1.
## PCA
pca.out <- prcomp(Coping, scale = TRUE)
summary(pca.out)

fviz_eig(pca.out)

components <- pca.out[["x"]]

components <- data.frame(components)

# components$PC2 <- -components$PC2
# 
# components$PC3 <- -components$PC3

#2d
fviz_pca_biplot(pca.out,geom = "point",
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)

# 3d
tot_explained_variance_ratio <- summary(pca.out)[["importance"]]['Proportion of Variance',]

tot_explained_variance_ratio <- 100 * sum(tot_explained_variance_ratio)


tit = paste("Total Explained Variance =",tot_explained_variance_ratio)


fig <- plot_ly(components, x = ~PC1, y = ~PC2, z = ~PC3 ) %>%
  
  add_markers(size = 12)
fig


## 2.
X2 <- pca.out$x[,1:2]
X3 <- pca.out$x[,1:3]


gskmn2 <- clusGap(X2, kmeans, K.max = 10)
plot(gskmn2, bty = "l", main ="Gap Statistics - K-means")

hclusCut <- function(x, k, meth = "euclidean")list(cluster = cutree(hclust(dist(x, method=meth)), k=k))

gshclust <- clusGap(X2, hclusCut, K.max = 10)
plot(gshclust, bty = "l", main ="Gap Statistics - Hierarchical")

set.seed(3337731)
km.out <- kmeans(X2, 3, nstart = 20)
plot(X2, col = (km.out$cluster +1),
     main = "K-Means Clustering Results with K = 3", pch = 20,
     bty = "l", xlab = "PC1", ylab = "PC2")

fig2 <- plot_ly(components, x = ~PC1, y = ~PC2, z = ~PC3 ,color = ~km.out$cluster) %>%
  add_markers(size = 12)
fig2
