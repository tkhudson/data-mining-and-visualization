CarPrice<-read.table(file="C:/Users/tyler/OneDrive/Documents/school2k20/Fall2020/CIS3339/hw5/car-price-prediction/CarPrice_Assignment.csv",sep=",",
                     header=TRUE)

cor(CarPrice[c()])

price<-
  mean(price)
sd(price)
quantile(price,prob=c(.25,.75))
IQR<-105475-79900
LowerBound<-79900-1.5*IQR
UpperBound<-105475-1.5
price>UpperBound
price(price>UpperBound)
which(price>UpperBound)
CarPrice[243,]
price[which(price<LowerBound)]

head(CarPrice)

boxplot(price-Area)

plot (price~size)

x<-c(3,5,7)
y<-c(1,4,4)
r=cov(x,y)/(stdev(x)*stdev(y))
cov(x,y)
sdx<-(x)
sdy<-(y)

cov(x,y)/(sdx*sdy)