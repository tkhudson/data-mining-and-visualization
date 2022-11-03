CarPrice<-read.table(file="C:/Users/tyler/OneDrive/Documents/school2k20/Fall2020/CIS3339/hw5/car-price-prediction/CarPrice_Assignment.csv",sep=",",
                     header=TRUE)
attach(CarPrice)

install.packages("datarium")
library(datarium)

head(CarPrice)

cor(CarPrice[c(-(0:20))])

o1<-summary(lm(price~horsepower))
o2<-summary(lm(price~horsepower+compressionratio+peakrpm+citympg+highwaympg))
o3<-summary(lm(price~horsepower+compressionratio+peakrpm+citympg+highwaympg+carbody))
