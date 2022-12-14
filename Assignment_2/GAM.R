MH_data <- read.table("MHpredict - Copy.csv", sep = ",", header = TRUE)

library(dplyr)
library(mgcv)

# Loop through the columns of the data frame
for (col in colnames(MH_data)) {
  # Check if the column is categorical
  if (!is.factor(MH_data[[col]]) && !is.numeric(MH_data[[col]])) {
    # Convert the column to a factor
    MH_data[[col]] <- as.factor(MH_data[[col]])
  }
}

GAM <- gam(dep_sev_fu ~ f(0)+f(1)+f(8)+f(9)+f(10)+f(11)+f(12)+f(13)+f(14)+f(16)+f(17)+f(18)+f(19)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7)+s(15),data = MH_data, method = "REML")

