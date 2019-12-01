library(dplyr)
library(stringr)
library(readr)
library(tidyr)
library(lubridate)
library(purrr)
library(readxl)
library(data.table) #fread
library(ggplot2)
library(ggforce) #this is for paginating the wrap
library(sqldf)
library(rebus)
library(scales)
library(broom)
library(corrplot) #for correlation plot
library(pROC)  # for ROC Curve




library(caret) #8 major arguments, 1. Y~X, data, model=, trControl=trainControl, tuneGrid or tuneLength, preProcess
library(mlbench)  #for sample data for machine learning
library(fastAdaboost) #AdaBoost Classification Trees
library(ipred)      #Stabilized Linear Discriminant Analysis
library(earth)      #Multivariate Adaptive Regression Spline model
library(caTools)    #Boosted Logistic Regression	model
library(C50)        #C50 ensemble model
library(xgboost)    #Xtreme gradiant boosting
library(h2o)        #glmnet model
library(naivebayes) #For Naivebayes models
library(pls)        #Principal Component Analysis	Models
library(randomForest)# For Random Forest Models
library(rrcov)      #Robust Linear Discriminant Analysis models
library(elasticnet) #elastic-net models
library(gbm)     #for generalised boosted regression models
library(ellipse) #for drawing ellipse in feature plots
library(glmnet)  #For lasso and Elastic-Net Regularized Generalized Linear Model



#Logistic Regression:
#_______________________
#m_log<-glm(y~x1+x2+x3,data=my_dataset,family="binomial")

library(ISLR) #It contain dataset for logistic regression
data(Smarket)

#Exploring Data
names(Smarket)

summary(Smarket)

#Do box plot and check the outliers in the Xs

par(mfrow=c(1,8))
for(i in 1:8) {
  boxplot(Smarket[,i], main=names(Smarket)[i])
}


dev.off() #When using the par always dev.off() after the work is over


#Missing data have have a big impact on modeling. Thus, you can use a missing plot to get a quick idea of the amount of missing data in the dataset. The x-axis shows attributes and the y-axis shows instances. Horizontal lines indicate missing data for an 
#instance, vertical blocks represent missing data for an attribute.


library(Amelia) 
#Plots a missingness map showing where missingness occurs in the dataset passed to amelia.
#Amelia is a program for missingness in the data
missmap(Smarket, col=c("blue","red"), legend=FALSE)

library(corrplot)
correlations <- cor(Smarket[,1:8])
print(correlations)
corrplot(correlations, method="circle")

#scatterplot matrix
pairs(Smarket, col=Smarket$Direction)

x <- Smarket[,1:8]
y <- Smarket[,9]
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)



#Building Logistic Regression Model
# Logistics Regression
glm.fit <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, data = Smarket, family = binomial)

summary(glm.fit)

glm.probs <- predict(glm.fit,type = "response")  
predict(glm.fit,type = "terms") #as per the model what should be the value of the Xs
predict(glm.fit,type = "link")


#Now, when you ask the predict to return type = link, you get values of f(y)
#The type = terms will return a matrix given fit of each observation on a linear scale.



glm.pred <- ifelse(glm.probs > 0.5, "Up", "Down")

table(glm.pred,Smarket$Direction)

library(pROC)
#It is simple roc function form the pROC package contain the predicted and the actual values
ROC<-roc(glm.pred,Smarket$Direction) #It wont work we need to convert the binomial variables to number 1 and 0

plot(ROC)
