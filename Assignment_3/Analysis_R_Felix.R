# library
library(ggplot2)
library(ggthemes)
library(corrplot)
library(plyr)
library(dplyr)
library(ggcorrplot)
library(glmnet)
library(fastDummies)
library(caret)
library(factoextra)
library(readr)
library(parallel)
library(doParallel)
library(xgboost)
library(Ckmeans.1d.dp)

# loading data 
train <- read.csv("data/train.csv", stringsAsFactors = F)
test <- read.csv("data/test.csv", stringsAsFactors = F)

# removing id, combining train and test
test_labels <- test$Id
test$Id <- NULL
train$Id <- NULL

test$SalePrice <- NA
all <- rbind(train, test)

all$PoolQC[is.na(all$PoolQC)] <- 'None'
Qualities <- c('None' = 0, 'Po' = 1, 'Fa' = 2, 'TA' = 3, 'Gd' = 4, 'Ex' = 5)
all$PoolQC<-as.integer(revalue(all$PoolQC, Qualities))


all$MiscFeature[is.na(all$MiscFeature)] <- 'None'
all$MiscFeature <- as.factor(all$MiscFeature)


all$Alley[is.na(all$Alley)] <- 'None'
all$Alley <- as.factor(all$Alley)

all$Fence[is.na(all$Fence)] <- 'None'
all$Fence <- as.factor(all$Fence)

all$FireplaceQu[is.na(all$FireplaceQu)] <- 'None'
all$FireplaceQu<-as.integer(revalue(all$FireplaceQu, Qualities))

for (i in 1:nrow(all)){
  if(is.na(all$LotFrontage[i])){
    all$LotFrontage[i] <- as.integer(median(all$LotFrontage[all$Neighborhood==all$Neighborhood[i]], na.rm=TRUE)) 
  }
}
all$LotShape<-as.integer(revalue(all$LotShape, c('IR3'=0, 'IR2'=1, 'IR1'=2, 'Reg'=3)))
all$LotConfig <- as.factor(all$LotConfig)


all$GarageYrBlt[is.na(all$GarageYrBlt)] <- all$YearBuilt[is.na(all$GarageYrBlt)]
all$GarageCond[2127] <- names(sort(-table(all$GarageCond)))[1]
all$GarageQual[2127] <- names(sort(-table(all$GarageQual)))[1]
all$GarageFinish[2127] <- names(sort(-table(all$GarageFinish)))[1]
all$GarageCars[2577] <- 0
all$GarageArea[2577] <- 0
all$GarageType[2577] <- NA
all$GarageType[is.na(all$GarageType)] <- 'No Garage'
all$GarageType <- as.factor(all$GarageType)
table(all$GarageType)
all$GarageFinish[is.na(all$GarageFinish)] <- 'None'
Finish <- c('None'=0, 'Unf'=1, 'RFn'=2, 'Fin'=3)
all$GarageFinish<-as.integer(revalue(all$GarageFinish, Finish))
all$GarageQual[is.na(all$GarageQual)] <- 'None'
all$GarageQual<-as.integer(revalue(all$GarageQual, Qualities))
all$GarageCond[is.na(all$GarageCond)] <- 'None'
all$GarageCond<-as.integer(revalue(all$GarageCond, Qualities))


all$BsmtFinType2[333] <- names(sort(-table(all$BsmtFinType2)))[1]
all$BsmtExposure[c(949, 1488, 2349)] <- names(sort(-table(all$BsmtExposure)))[1]
all$BsmtCond[c(2041, 2186, 2525)] <- names(sort(-table(all$BsmtCond)))[1]
all$BsmtQual[c(2218, 2219)] <- names(sort(-table(all$BsmtQual)))[1]
all$BsmtQual[is.na(all$BsmtQual)] <- 'None'
all$BsmtQual<-as.integer(revalue(all$BsmtQual, Qualities))
all$BsmtCond[is.na(all$BsmtCond)] <- 'None'
all$BsmtCond<-as.integer(revalue(all$BsmtCond, Qualities))
all$BsmtExposure[is.na(all$BsmtExposure)] <- 'None'
Exposure <- c('None'=0, 'No'=1, 'Mn'=2, 'Av'=3, 'Gd'=4)
all$BsmtExposure<-as.integer(revalue(all$BsmtExposure, Exposure))
all$BsmtFinType1[is.na(all$BsmtFinType1)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
all$BsmtFinType1<-as.integer(revalue(all$BsmtFinType1, FinType))
all$BsmtFinType2[is.na(all$BsmtFinType2)] <- 'None'
FinType <- c('None'=0, 'Unf'=1, 'LwQ'=2, 'Rec'=3, 'BLQ'=4, 'ALQ'=5, 'GLQ'=6)
all$BsmtFinType2<-as.integer(revalue(all$BsmtFinType2, FinType))
all$BsmtFullBath[is.na(all$BsmtFullBath)] <-0
all$BsmtHalfBath[is.na(all$BsmtHalfBath)] <-0
all$BsmtFinSF1[is.na(all$BsmtFinSF1)] <-0
all$BsmtFinSF2[is.na(all$BsmtFinSF2)] <-0
all$BsmtUnfSF[is.na(all$BsmtUnfSF)] <-0
all$TotalBsmtSF[is.na(all$TotalBsmtSF)] <-0

all$MasVnrType[is.na(all$MasVnrType)] <- 'None'
Masonry <- c('None'=0, 'BrkCmn'=0, 'BrkFace'=1, 'Stone'=2)
all$MasVnrType<-as.integer(revalue(all$MasVnrType, Masonry))
all$MasVnrArea[is.na(all$MasVnrArea)] <-0

all$MSZoning[is.na(all$MSZoning)] <- names(sort(-table(all$MSZoning)))[1]
all$MSZoning <- as.factor(all$MSZoning)

all$KitchenQual[is.na(all$KitchenQual)] <- 'TA' #replace with most common value
all$KitchenQual<-as.integer(revalue(all$KitchenQual, Qualities))

all$Utilities <- NULL

all$Functional[is.na(all$Functional)] <- names(sort(-table(all$Functional)))[1]
all$Functional <- as.integer(revalue(all$Functional, c('Sal'=0, 'Sev'=1, 'Maj2'=2, 'Maj1'=3, 'Mod'=4, 'Min2'=5, 'Min1'=6, 'Typ'=7)))
table(all$Functional)

all$Exterior1st[is.na(all$Exterior1st)] <- names(sort(-table(all$Exterior1st)))[1]
all$Exterior1st <- as.factor(all$Exterior1st)
all$Exterior2nd[is.na(all$Exterior2nd)] <- names(sort(-table(all$Exterior2nd)))[1]
all$Exterior2nd <- as.factor(all$Exterior2nd)


all$Electrical[is.na(all$Electrical)] <- names(sort(-table(all$Electrical)))[1]
all$Electrical <- as.factor(all$Electrical)

all$SaleType[is.na(all$SaleType)] <- names(sort(-table(all$SaleType)))[1]
all$SaleType <- as.factor(all$SaleType)
all$SaleCondition <- as.factor(all$SaleCondition)
table(all$SaleCondition)


NAcol <- which(colSums(is.na(all)) > 0)
sort(colSums(sapply(all[NAcol], is.na)), decreasing = TRUE)

all$Foundation <- as.factor(all$Foundation)
all$Heating <- as.factor(all$Heating)
all$RoofStyle <- as.factor(all$RoofStyle)
all$LandContour <- as.factor(all$LandContour)
all$BldgType <- as.factor(all$BldgType)
all$HouseStyle <- as.factor(all$HouseStyle)
all$Neighborhood <- as.factor(all$Neighborhood)
all$Street<-as.factor(as.integer(revalue(all$Street, c('Grvl'=0, 'Pave'=1))))
all$MoSold <- as.factor(all$MoSold)
all$MSSubClass <- as.factor(all$MSSubClass)

# outliers 
all <- all[-c(524, 1299),] # two big houses with very low sales prices

# Find character columns in the data frame
char_cols <- sapply(all, is.character)

# Select character columns from the data frame
char_df <- all[,char_cols]

# Convert character columns to factors
char_df <- lapply(char_df, factor)

# Replace character columns in the original data frame with the factor columns
all[,char_cols] <- char_df

#Two sets - numeric and factor
data_factor <- select_if(all, is.factor)
data_numeric<- select_if(all, is.numeric)
data_numeric <- data_numeric[-1]

#SCALING numeric variables
data_numeric_scale<-data.frame(scale(data_numeric))
#Creating DUMMIES for factors
data_factor_dummy <- dummy_cols(data_factor, remove_first_dummy = TRUE)
data_factor_dummy <- select_if(data_factor_dummy, is.numeric)


data_num <- cbind(data_numeric, data_factor_dummy) #set without standarization numeric variables
data_scale <- cbind(data_numeric_scale, data_factor_dummy) #set with standardization numeric variables

train <- data_num[!is.na(data_num$SalePrice),]
test <- data_num[is.na(data_num$SalePrice),]

X_train <- train[, -48]
y_train <- train[,48]

X_test <- test[, -48]
y_test <- test[,48]

train_scale <- data_scale[!is.na(data_scale$SalePrice),]
test_scale <- data_scale[is.na(data_scale$SalePrice),]

X_train_scale <- train_scale[, -48]
y_train_scale <- train_scale[,48]

X_test_scale <- test_scale[, -48]
y_test_scale <- test_scale[,48]


# dependent variable 
ggplot(data=all[!is.na(all$SalePrice),], aes(x=SalePrice)) +
  geom_histogram(binwidth = 10000) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = scales::comma) +
  theme_stata() + scale_color_stata() +
  ggtitle("Outcome Variable")

# correlations amongst numerical variables 
var_num_ind <- which(sapply(all, is.numeric)) 
var_num_names <- names(var_num_ind)

var_num <- all[, var_num_ind]
var_num_cor <- cor(var_num, use="pairwise.complete.obs")

cor_sorted <- as.matrix(sort(var_num_cor[,'SalePrice'], decreasing = TRUE))
Cor_high <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
var_num_cor_high <- var_num_cor[Cor_high, Cor_high]

corrplot.mixed(var_num_cor_high, tl.col="black", tl.pos = "lt")
corrplot.mixed(var_num_cor, tl.col="black", tl.pos = "lt")

# multicollinearity
p.mat <- cor_pmat(var_num)
ggcorrplot(var_num_cor, hc.order = TRUE,
           type = "lower", p.mat = p.mat)





# Lasso
lasso_cv = cv.glmnet(as.matrix(X_train_scale)  , y_train_scale , alpha=1) 

plot(lasso_cv, type="b", bty ="l")

lasso_viz = glmnet( as.matrix(X_train_scale), y_train_scale, alpha=1)

plot(lasso_viz, xvar = "lambda")


# create model with optimal lambda to predict values 
lasso_model = glmnet( as.matrix(X_train_scale), y_train_scale, alpha=1 ,
                      lambda=lasso_cv$lambda.min)

# Extract the coefficients from the model
coefs <- coef(lasso_model)

# Sort the coefficients in descending order
sorted_coefs <- sort(coefs, decreasing = TRUE)

# Show the largest coefficients
head(sorted_coefs)


sorted_coefs 

sd(train$OverallQual)* sd(train$SalePrice)*coefs
sd(train$GrLivArea)* sd(train$SalePrice)*coefs


# predict
SalePrice_Lasso <- predict(lasso_model, newx = as.matrix(X_test_scale))
# unscale prediction
SalePrice_Lasso2 <- predict(lasso_model, newx = as.matrix(X_test_scale)) * sd(train$SalePrice) + mean(train$SalePrice)


#PCR
set.seed(3337731)
model_pcr <- train(SalePrice ~ ., data = train_scale, method = "pcr", scale = F,
                   trControl = trainControl("cv", number = 10), tuneLength = 170 )

model_pcr$bestTune

plot(model_pcr)

summary(model_pcr)
coef(model_pcr)

#predict
SalePrice_PCR <- predict(model_pcr, X_test_scale)
#unscale prediction
SalePrice_PCR2 <- predict(model_pcr, X_test_scale) * sd(train$SalePrice) + mean(train$SalePrice)

# PCA 
pca <- prcomp(X_train_scale, scale = F)

fviz_eig(pca, addlabels = TRUE, cumulative = TRUE)

summary(pca)

num_pcs <- 16  # select the first 3 principal components
pcs <- predict(pca, newdata = X_train_scale)[, 1:num_pcs]

model <- lm(Y ~ ., data = data.frame(pcs, Y = y_train_scale))


# XGBoost
cluster <- makeCluster(detectCores() - 2) 
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 10,
                           allowParallel = TRUE)


xgb_grid = expand.grid(
  nrounds = 1000,
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree=1,
  min_child_weight=c(1, 2, 3, 4 ,5),
  subsample=1
)

xgb_caret <- train(x=X_train, y=y_train, method='xgbTree', trControl= fitControl, tuneGrid=xgb_grid) 
xgb_caret$bestTune


# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(as.matrix(X_train_scale), label = y_train_scale)
dtest <- xgb.DMatrix(as.matrix(X_test_scale))

xgb_params <- list(
  booster = 'gbtree',
  objective = 'reg:linear',
  colsample_bytree=1,
  eta=0.05,
  max_depth=4,
  min_child_weight=3,
  alpha=0.3,
  lambda=0.4,
  gamma=0.01, # less overfit
  subsample=0.6,
  seed=5,
  silent=TRUE)

xgbcv <- xgb.cv(xgb_params, dtrain, nrounds = 5000, nfold = 4, early_stopping_rounds = 500)
#train the model using the best iteration found by cross validation
xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 454)

XGBpred <- predict(xgb_mod, dtest) * sd(train$SalePrice) + mean(train$SalePrice)

 
mat <- xgb.importance (feature_names = colnames(X_train_scale),model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:20], rel_to_first = F)

### submissions


# lasso
SalePrice <- SalePrice_Lasso2
Id <- test_labels
to_csv <- data.frame(Id, SalePrice)
colnames(to_csv) <- c("Id","SalePrice")
write_csv(to_csv, 'lasso_submission.csv')

# pcr

SalePrice <- SalePrice_PCR2
Id <- test_labels
to_csv <- data.frame(Id, SalePrice)
colnames(to_csv) <- c("Id","SalePrice")
write_csv(to_csv, 'pcr_submission_kaiser.csv')

# xgbboost

SalePrice <- XGBpred
Id <- test_labels
to_csv <- data.frame(Id, SalePrice)
colnames(to_csv) <- c("Id","SalePrice")
write_csv(to_csv, 'XGB_submission.csv')
