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

plot(ROC,col="red")
auc(ROC)

#Interaction Effect:
glm(Direction~Lag1*Lag2, data=Smarket, family="binomial") #Lag1*Lag2 means Lag1, Lag2 and Lag1&Lag2








#Naive Bayes
#Conditional Probability: Many times the info available that an event has occured and one is required
#to find out the probability of occurance of another event B using the information about A.

library(naivebayes)
m_nb<-naive_bayes(location~time_of_day, data=location_history)

#making predictions with the naive bayes
future_location<-predict(m_nb,future_condition)


#predict location at x time:
predict(m_nb,all_Xs_no_Y_for_new_data)

#obtaining the probability
predict(m_nb,all_Xs_no_Y_for_new_data,type="prob")

#Its called Naive because it makes assumption of event independence



#Decision Tree in r
#_____________________________________________________________________________

library(rpart)

m_dt<-rpart(outcome~loanamount+credit_score, data=loans, method="class")
#class means for classification problems
#anova is for regression problems


p_dt<-predict(m, test_data, type="class")

#visualizing the classification tree in r
library(rpart.plot)
rpart.plot(m_dt)
#plotting with customized setting
rpart.plot(m_dt,type=3,box.palette=c("red","green"),fallen.leaves=TRUE)

# type
# Type of plot. Possible values:
# 
# 0 Draw a split label at each split and a node label at each leaf.
# 
# 1 Label all nodes, not just leaves. Similar to text.rpart's all=TRUE.
# 
# 2 Default. Like 1 but draw the split labels below the node labels. Similar to the plots in the CART book.
# 
# 3 Draw separate split labels for the left and right directions.
# 
# 4 Like 3 but label all nodes, not just leaves. Similar to text.rpart's fancy=TRUE. See also clip.right.labs.

# extra
# Display extra information at the nodes.
# 0 No extra information.
# 1 Display the number of observations that fall in the node
# 2 Class models: display the classification rate at the node, expressed as the number of correct classifications and the number of observations in the node.
# 3 Class models: misclassification rate at the node, 
# 4 Class models: probability per class of observations in the node 
# 
# 
# fallen.leaves
# Default TRUE to position the leaf nodes at the bottom of the graph. 
# It can be helpful to use FALSE if the graph is too crowded and the text size is too small.


# cex
# Default NULL, meaning calculate the text size automatically. 
# Since font sizes are discrete, the cex you ask for may not be exactly the cex you get.


# box.palette
# Palette for coloring the node boxes based on the fitted value. 
# This is a vector of colors, for example box.palette=c("green", "green2", "green4"). 


#Growing a tree
loan_model<-rpart(outcome~.,data=loans_train,method="class",control=rpart.control(...))


#control=rpart.control
#Control For Rpart Fits: arious parameters that control aspects of the rpart fit.
# rpart.control(minsplit = 20, minbucket = round(minsplit/3), cp = 0.01, 
#               maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, xval = 10,
#               surrogatestyle = 0, maxdepth = 30, …)

# minsplit
# the minimum number of observations that must exist in a node in order for a split to be attempted.

# minbucket
# the minimum number of observations in any terminal <leaf> node. If only one of minbucket or minsplit is specified, 
# the code either sets minsplit to minbucket*3 or minbucket to minsplit/3, as appropriate.

# cp
# complexity parameter. Any split that does not decrease the overall lack of fit by a factor of cp 
# is not attempted. For instance, with anova splitting, this means that the overall R-squared must 
# increase by cp at each step. The main role of this parameter is to save computing time by pruning 
# off splits that are obviously not worthwhile.

# maxdepth
# Set the maximum depth of any node of the final tree, with the root node counted as depth 0.
# Values greater than 30 rpart will give nonsense results on 32-bit machines.



#pre-pruning
prunce_control<-rpart.control(maxdepth = 20, minsplit=20) #min no of obs for which tree is allowed to split

m<-rpart(repaid~credit_score+request_amt, data=loans, method="class", control=prunce_control)

#we check the complexity at different cp by plotcp
plotcp(m)
m$cptable

#why do tree benefit from pruning?
#Classification tree can grow indefinitely untill they are told to stop or run out of data to divide and conquer
#Just like trees in nature . classification tree that grow overly large ccan require pruning to reduce the excess
#growth. However this generally result in a tree that classify fewer traning examples correctly.

#Why then are pre-pruning amd post pruning are almost always used?
#1.simplier tree easier to interpret
#2.simpler tree using early stopping are faster to train
#3.simpler tree may perform better on testing data








