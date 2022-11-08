### Felix Wente S3337731

### Part A. - Supervised Learning

## Crating Test and Train Data Set
library(readr)
data_set <- read_csv("Data3337731.csv")
train <- data_set[1:5000, ]
test <- data_set[5001:10000, ]

## Questions

## 1.

