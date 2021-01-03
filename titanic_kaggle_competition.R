#Loading data
test <- read.csv("test.csv")
train <- read.csv("train.csv")



#We need to predict if passenger survived or no so in the end we are 
#interested in column PassengerId and Survived

#Exploring data
head(train)
tail(train)
head(test)

summary(train)
#survival - did passenger survive - 0=no, 1=yes
#pclass - ticket class - 1=1st, 2=2nd, 3=3rd
#name - if passengerID are unique we may drop it but maybe later we will bring it back
#sex - sex - m/f <-- might want to do feature enginerring and set 0=male, 1=female
#age - age in years <-- some NA's - hard to replace but it is important feature so we will remove it for now
#sibsp - number of siblings / spouses aboard the Titanic <-- one generation
#parch - number of parents / children aboard the Titanic <-- generation above or below
#ticket - ticket number <-- for now we will leave it - hard to get information from it
#fare - passenger fare
#cabin - cabin number <-- many blind spots - for now we will leave it
#embarked - port of embarkation - C=Cherbourg, Q=Quennstown, S=Southampton <-- 1=Cherbourg, 2=Queenstown, 3=Southampton

#removing NA\s
sum(is.na(train$Age))
df<-na.omit(train)
summary(df)

df_t<-test
#feature enginerring
df[df$Sex=="male","Sex"]<-0
df[df$Sex=="female","Sex"]<-1
df$Sex<-as.numeric(df$Sex)

sum(df$Sex) #261 female passengers
length(df$Sex)-sum(df$Sex) #453 male passengers

df_t[df_t$Sex=="male","Sex"]<-0
df_t[df_t$Sex=="female","Sex"]<-1
df_t$Sex<-as.numeric(df_t$Sex)
#leaving ticket and cabin
df$Cabin<-NULL
df$Ticket<-NULL
summary(df)

df_t$Cabin<-NULL
df_t$Ticket<-NULL
#checking if passengerID are unique
sum(duplicated(df$PassengerId))
#they are so for now we drop name column
df_name<-df
df$Name<-NULL
summary(df)

df_t$Name<-NULL
#feature enginerring on Embarked
df[df$Embarked=="C",]$Embarked <- 1
df[df$Embarked=="Q",]$Embarked <- 2
df[df$Embarked=="S",]$Embarked <- 3
df$Embarked<-as.numeric(df$Embarked)
summary(df)

df_t[df_t$Embarked=="C",]$Embarked <- 1
df_t[df_t$Embarked=="Q",]$Embarked <- 2
df_t[df_t$Embarked=="S",]$Embarked <- 3
df_t$Embarked<-as.numeric(df_t$Embarked)
#we got 2 NA's, will replace them with random number
df[is.na(df$Embarked)==TRUE,]$Embarked <- c(1,2)
summary(df)

summary(df_t)
df_t[is.na(df_t$Age)==TRUE,]$Age<-round(rnorm(86, mean(df$Age),sd(df$Age)))
df_t[is.na(df_t$Fare)==TRUE,]$Fare<-round(rnorm(1, mean(df$Age),sd(df$Age)))

#creating correlation matrix and visualizing it
corrmat<-round(cor(df),2)
head(corrmat)

library(reshape2)
melted_cormat <- melt(corrmat)
head(melted_cormat)

library(ggplot2)
#get upper trainge of the correlation matrix
get_upper_tri <- function(corrmat){
  corrmat[lower.tri(corrmat)]<-NA
  return(corrmat)
}

upper_tri <- get_upper_tri(corrmat)
upper_tri
# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Heatmap
ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

#negative correlation:
#1. survived ~ pclass 
#2. survived ~ embarked
#positive correlation:
#1. survived ~ sex
#2. survived ~ fare

#another analysis
library(GGally)
df$Survived<-as.factor(df$Survived)
ggpairs(df,columns=c(1,3:9),ggplot2::aes(colour=Survived))

#most people which did not survive were
#-from 3rd class
#-males
#-paid low fare
#-embarked from Southampton

#we use SVM to predict if someone survived
#at first we are going to use pclass, sex, fare and embarked as predictors
library(e1071)
#using linear kernel
out=svm(Survived~Pclass+Sex+Fare+Embarked, data=df, kernel="linear",cost=1)
summary(out)

table(out$fitted,df$Survived)
(64+93)/(360+197+64+93) #21.99% error rates
#using radial kernel
out=svm(Survived~Pclass+Sex+Fare+Embarked, data=df, kernel="radial", gamma=1, cost=1)
summary(out)

table(out$fitted,df$Survived)
(118+19)/(405+172+118+19) #19.19% error rates
#using polynomial kernel

out=svm(Survived~Pclass+Sex+Fare+Embarked, data=df, kernel="polynomial", gamma=1, cost=1)
summary(out)

table(out$fitted,df$Survived)
(127+14)/(127+14+410+163) #19.75% error rates

#it seems that radial kernel is the best option
#lets check for the best gamma and cost parameters using tune() function
tune.out<-tune(svm, Survived~Pclass+Sex+Fare+Embarked, data=df, kernel="radial", 
               range = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
summary(tune.out)
out<-tune.out$best.model
#we got cost=1 and gamma=0.5 as best parameters

#first predictions
pred.te=predict(out,df_t)
head(pred.te)
results<-data.frame(PassengerId=892:1309, Survived=pred.te) #first submission

write.csv(results,file="svm_results1.csv", row.names=F) #got 77.27% prediction score on Kaggle
#--------------------------------------------------------------------------------------------
#Improvements from here
