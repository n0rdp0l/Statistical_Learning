library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)
library(caret)
library("mgcv")


train <- read.csv("data/train.csv", stringsAsFactors = F)
test <- read.csv("data/test.csv", stringsAsFactors = F)


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



quick_RF <- randomForest(x=all[1:1460,-79], y=all$SalePrice[1:1460], ntree=100,importance=TRUE)

all<-all[!(all$SalePrice>5000000),]


GAModel <- gam(SalePrice ~ s(MSSubClass) + MSZoning + s(LotFrontage) + s(LotArea) + Street + Alley + LotShape + LandContour + LotConfig +
                 LandSlope + Neighborhood + Condition1 + Condition2 + BldgType + HouseStyle + s(OverallQual) +  
                 OverallCond + s(YearBuilt) + s(YearRemodAdd) + RoofStyle + RoofMatl + Exterior1st + Exterior2nd + MasVnrType +   
                 s(MasVnrArea) + ExterQual + ExterCond + Foundation + BsmtQual + BsmtCond + BsmtExposure + BsmtFinType1 +
                 s(BsmtFinSF1) + BsmtFinType2 + s(BsmtFinSF2) + s(BsmtUnfSF) + s(TotalBsmtSF) + Heating + HeatingQC + CentralAir + 
                 Electrical + s(X1stFlrSF) + s(X2ndFlrSF) + s(LowQualFinSF) + s(GrLivArea) + BsmtFullBath + BsmtHalfBath + FullBath  +   
                 HalfBath + BedroomAbvGr + KitchenAbvGr + KitchenQual + s(TotRmsAbvGrd) + Functional + Fireplaces + FireplaceQu +  
                 GarageType + s(GarageYrBlt) + GarageFinish + GarageCars + s(GarageArea) + GarageQual + GarageCond + PavedDrive  + 
                 s(WoodDeckSF) + s(OpenPorchSF) + s(EnclosedPorch) + s(X3SsnPorch) + s(ScreenPorch) + PoolArea + PoolQC + Fence +       
                 MiscFeature + s(MiscVal) + s(MoSold) + YrSold + SaleType + SaleCondition,
               data = all[1:1459,],select = T, family = gaussian(), method='GCV.Cp')
summary(GAModel)
plot(GAModel)

GAModel <- gam(dep_sev_fu ~ disType + Sexe+s(Age)+aedu+s(IDS)+s(BAI)+s(FQ)+s(LCImax)+pedigree+alcohol+bTypeDep+bSocPhob+bGAD+bPanic+bAgo+s(AO)+RemDis+sample+ADuse+PsychTreat,
               data = MH_data[train,],select = T, family = gaussian(), method='GCV.Cp')


RandForest <- randomForest(SalePrice ~ ., data = all[1:1459,],
                           importance = TRUE, ntree=1000, mtry=9)
#Make prediction on test set
prediction_RF <- predict(RandForest, newdata = MH_data[-train, ])
#Compute MAE
mae(MH_data[-train, ]$dep_sev_fu, prediction_RF)
#Summary and plot for interpretation
importance(quick_RF)
varImpPlot(quick_RF, cex = 1)
