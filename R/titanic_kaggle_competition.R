#Libraries
library(e1071)
library(reshape2)
library(ggplot2)
library(GGally)
library(fitdistrplus)
#Loading data
test <- read.csv("test.csv")
train <- read.csv("train.csv")


#We need to predict if passenger survived or no so in the end we are 
#interested in column PassengerId and Survived

#Exploring data
head(df)
tail(df)
summary(df)
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


melted_cormat <- melt(corrmat)
head(melted_cormat)

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

df$Survived<-as.factor(df$Survived)
ggpairs(df,columns=c(1,3:9),ggplot2::aes(colour=Survived))

#most people which did not survive were
#-from 3rd class
#-males
#-paid low fare
#-embarked from Southampton

#we use SVM to predict if someone survived
#at first we are going to use pclass, sex, fare and embarked as predictors

#using linear kernel
out=svm(Survived~Pclass+Sex+Fare+Embarked, data=df, kernel="linear",cost=1)
summary(out)

table(out$fitted,df$Survived)
(64+93)/(360+197+64+93) #21.99% training error rates
#using radial kernel
out=svm(Survived~Pclass+Sex+Fare+Embarked, data=df, kernel="radial", gamma=1, cost=1)
summary(out)

table(out$fitted,df$Survived)
(118+19)/(405+172+118+19) #19.19% training error rates
#using polynomial kernel

out=svm(Survived~Pclass+Sex+Fare+Embarked, data=df, kernel="polynomial", gamma=1, cost=1)
summary(out)

table(out$fitted,df$Survived)
(127+14)/(127+14+410+163) #19.75% training error rates

#it seems that radial kernel is the best option
#lets check for the best gamma and cost parameters using tune() function
tune.out<-tune(svm, Survived~Pclass+Age+Sex+Fare+Embarked, data=df, kernel="radial", 
               range = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
summary(tune.out)
out<-tune.out$best.model
#we got cost=1 and gamma=0.5 as best parameters
table(out$fitted,df$Survived) #16.95% training error rates
#first predictions

pred.te=predict(out,df_t)

results<-data.frame(PassengerId=892:1309, Survived=pred.te) #first submission

write.csv(results,file="svm_results1.csv", row.names=F) #got 78.29% prediction score on Kaggle
#--------------------------------------------------------------------------------------------
#Improvements from here

#guessing age instead of dropping
#trying to get some informations from names

df2 <- train
df2_t <- test
#What is Age distributed?
age<-df2[is.na(df2$Age)==FALSE, ]$Age
age_t<-df2_t[is.na(df2_t$Age)==FALSE, ]$Age

descdist(age,discrete=FALSE,boot=1000) 
#Looks like normal, lognormal or gamma distribution but will check also weibull
#Fitting those distributions and checking which one fits best the data
fln<-fitdist(age,"lnorm")
fn<-fitdist(age,"norm")
fw<-fitdist(age,"weibull")
fg<-fitdist(age,"gamma")
plot.legenda<-c("Lognormal","Normal","Weibull","Gamma")
denscomp(list(fln,fn,fw,fg),legendtext=plot.legenda)
qqcomp(list(fln,fn,fw,fg),legendtext=plot.legenda)
cdfcomp(list(fln,fn,fw,fg),legendtext=plot.legenda)
#Looks like Weibull and Normal
#Lets see the criteria information
dists <- data.frame("Normal"=c(fn$loglik,fn$aic),"Gamma"=c(fg$loglik, fg$aic),
                       "Weibull"=c(fw$loglik,fw$aic), "Lognormal"=c(fln$loglik,fln$aic),
                       row.names = c("Loglikelihood","AIC"))
#Depending on the results we assume that age is normally distributed
#Now, we will generate from this distribution random number in the place of NA's
summary(df2$Age) #177 NA's
set.seed(1)
df2[is.na(df2$Age)==TRUE, ]$Age <- round(rnorm(177, mean(age),sd(age)))

summary(df2_t$Age) #86 NA's
df2_t[is.na(df2_t$Age)==TRUE, ]$Age <- round(rnorm(86, mean(age_t),sd(age_t)))
df2_t$Age <- round(df2_t$Age)

#feature enginerring
df2[df2$Sex=="male","Sex"]<-0
df2[df2$Sex=="female","Sex"]<-1
df2$Sex<-as.numeric(df2$Sex)

sum(df2$Sex) #261 female passengers
length(df2$Sex)-sum(df2$Sex) #453 male passengers


df2_t[df2_t$Sex=="male","Sex"]<-0
df2_t[df2_t$Sex=="female","Sex"]<-1
df2_t$Sex<-as.numeric(df2_t$Sex)

#leaving ticket and cabin
df2$Cabin<-NULL
df2$Ticket<-NULL
summary(df2)

df2_t$Cabin<-NULL
df2_t$Ticket<-NULL

#Trying to reach out the information from names
df2["Title"]<-sub("\\s.*","",sub(".*,\\s","",df2$Name))
df2["Title"]<-factor(df2$Title)
meaning<-unique(df2$Title)
df2$Title<-as.numeric(df2$Title)
numeric<-unique(df2$Title)
meaning <- data.frame(numeric=numeric, meaning=meaning)
meaning

#Adding to test data
df2_t["Title"]<-sub("\\s.*","",sub(".*,\\s","",df2_t$Name))
df2_t["Title"]<-factor(df2_t$Title)
meaning_t<-unique(df2_t$Title)
df2_t$Title<-as.numeric(df2_t$Title)
numeric_t<-unique(df2_t$Title)
meaning_t <- data.frame(numeric=numeric_t, meaning=meaning_t)

#Now we may drop Name column for the 2nd attempt
df2$Name <- NULL
df2_t$Name <- NULL
#feature enginerring on Embarked
df2[df2$Embarked=="C",]$Embarked <- 1
df2[df2$Embarked=="Q",]$Embarked <- 2
df2[df2$Embarked=="S",]$Embarked <- 3
df2$Embarked<-as.numeric(df2$Embarked)
summary(df2)

df2_t[df2_t$Embarked=="C",]$Embarked <- 1
df2_t[df2_t$Embarked=="Q",]$Embarked <- 2
df2_t[df2_t$Embarked=="S",]$Embarked <- 3
df2_t$Embarked<-as.numeric(df2_t$Embarked)

#we got 2 NA's, will replace them with random number
df2[is.na(df2$Embarked)==TRUE,]$Embarked <- sample(c(1,2,3),2)
summary(df2)
df2

#Getting rid off NA in Fare test data
summary(df2_t)
fare <- df2_t[is.na(df2_t$Fare)==FALSE,]$Fare
df2_t[is.na(df2_t$Fare)==TRUE,]$Fare<-rnorm(1,mean(fare),sd(fare))

#creating correlation matrix and visualizing it
corrmat<-round(cor(df2),2)
head(corrmat)

melted_cormat <- melt(corrmat)
head(melted_cormat)

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


#another analysis
df2$Survived<-as.factor(df2$Survived)
ggpairs(df,columns=c(1,3:10),ggplot2::aes(colour=Survived))


#most people which did not survive were
#-from 3rd class
#-male / Mr
#-paid low fare
#-embarked from Southampton

#we use SVM to predict if someone survived
#at first we are going to use pclass, sex, fare and embarked as predictors
#including also title
#we might get improvement from more observations and another predictor corresponding to the last time

#using linear kernel
out=svm(Survived~Pclass+Sex+Fare+Embarked+Title, data=df2, kernel="linear",cost=1)
summary(out)

table(out$fitted,df2$Survived)#21.32% training error rates - slightly better
#using radial kernel
out=svm(Survived~Pclass+Sex+Fare+Embarked+Title, data=df2, kernel="radial", gamma=1, cost=1)
summary(out)

table(out$fitted,df2$Survived)#15.6% training error rates - much better -- 4%!

#using polynomial kernel
out=svm(Survived~Pclass+Sex+Fare+Embarked+Title, data=df2, kernel="polynomial", gamma=1, cost=1)
summary(out)

table(out$fitted,df2$Survived)#17.62% training error rates - slightly better

#it seems that radial kernel is the best option
#lets check for the best gamma and cost parameters using tune() function
tune.out<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+Title+Age, data=df2, kernel="radial", 
               range = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
summary(tune.out)
out<-tune.out$best.model
#we got cost=10 and gamma=1 as best parameters
table(out$fitted,df2$Survived) #15.49% training error rates

#first predictions
pred.te=predict(out,df2_t)

results2<-data.frame(PassengerId=892:1309, Survived=pred.te) #first submission

write.csv(results2,file="svm_results2.csv", row.names=F) #got 71.05% prediction score on Kaggle
#--------------------------------------------------------------------------------------------

#Next attempt
#Adding feature on family size
#We will use last dataset but will change Title variable more effictively
head(df2)
#Make main titles and the rest of rare
#Mr <- 1, Mrs <-2, Ms, Miss <-3, Rest <- 4
t(meaning)
df2[df2$Title!=9 & df2$Title!=12 & df2$Title!=13 & df2$Title!=14,]$Title <-4
df2[df2$Title==12,]$Title <- 1
df2[df2$Title==13,]$Title <- 2
df2[df2$Title==9 | df2$Title==14,]$Title <- 3

t(meaning_t)
df2_t[df2_t$Title!=6 & df2_t$Title!=7& df2_t$Title!=5& df2_t$Title!=8,]$Title <-4
df2_t[df2_t$Title==6,]$Title <- 1
df2_t[df2_t$Title==7,]$Title <- 2
df2_t[df2_t$Title==5 | df2_t$Title==8,]$Title <- 3

#Adding Family size as Fsize <- includes Parch + SibSp + 1
df2["Fsize"]<-df2$Parch + df2$SibSp + 1
df2_t["Fsize"] <- df2_t$Parch + df2_t$SibSp +1
head(df2)
head(df2_t)

#Creating correlogram to see relationships between features
df2$Survived<-as.factor(df2$Survived)
ggpairs(df2,columns=c(1,3:11),ggplot2::aes(colour=Survived))

histo <- function(data,var){
  ggplot(data, aes(x=var, fill=Survived)) + geom_histogram(stat="count")
}

densit <- function(data,var){
  ggplot(data,aes(x=var,fill=Survived)) + geom_density(alpha=0.5)
}

colnames(df2)
#Pclass
histo(df2,df2$Pclass)
#in the 1st class there were more survivors
#in the 3rd the number of non-survivors is relatively high comparing to the number of survivors

#Sex
histo(df2,df2$Sex)
#as thought, most females survived while most males did not

#Age
densit(df2,df2$Age)
#more people who did survive were kids and adults, while the rest
#who did not survive were mostly young adults and in the elderly age

#Fare
densit(df2,df2$Fare)
#the less is the charge the less survivors

#Embarked
histo(df2,df2$Embarked)
#the are more survivors in Cherbourg 
#but less in Queenstown or Southampton
ggplot(df2,aes(x=Fare,fill=factor(Embarked))) + geom_density(alpha=0.5)
#as we can see in Queenstown or Southampton most Fares are in the low interval
#while in the Cherbourg we may notice that Fares were aswell low as high
#so the reason standing behind more survivors in Cherbourg are probably best paid cabins

#Title
histo(df2,df2$Title)
#most Mr - men did not survive
#most Ms or Mrs - women did survive
#the rest titles - there is kinda equal division between survivors and non-survivors
#Title might not be necessary in the model, we will see results with and without it
#So far in the last models we could notice that model without Title predictor got better results

#Fsize
histo(df2,df2$Fsize)
#in the families of size 2-4 there were more survivors
#in the rest families of size 1 or 5 and more there were more non-survivors
#this predictor might bring added value with itself

#splitting train data  df2 into train and test using 10-fold cross validation
#randomly shuffle the data
df2_2<-df2[sample(nrow(df2)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(df2_2)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- df2_2[testIndexes, ]
  trainData <- df2_2[-testIndexes, ]
  #Use the test and train data partitions however you desire...
}

#Performing SVM using radial kernel and choosing best parameters with tune()
#with Title
tune.out_with<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+Title+Age+Fsize, data=trainData, kernel="radial", 
               range = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
summary(tune.out_with)
out_with<-tune.out_with$best.model
#we got cost=1 and gamma=0.5 as best parameters
table(out_with$fitted,trainData$Survived) #13.97% training error rates

#predictions
pred.with=predict(out_with,testData)
table(predict=pred.with,true=testData$Survived) #19.1% error rates

results<-data.frame(PassengerId=892:1309, Survived=pred.te) #first submission

#without Title
tune.out_less<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+Age+Fsize, data=trainData, kernel="radial", 
                    range = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
summary(tune.out_less)
out_less<-tune.out_less$best.model
#we got cost=1 and gamma=0.5 as best parameters
table(out_less$fitted,trainData$Survived) #14.71% training error rates

#predictions
pred.less=predict(out_less,testData)
table(predict=pred.less,true=testData$Survived) #19.1% error rates

#we get similar level of error rates with and without Title predictor
#in this case we are perform two models and see on Kaggle which results better
tune.out_with<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+Title+Age+Fsize, data=df2, kernel="radial", 
               range = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
tune.out_less<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+Age+Fsize, data=df2, kernel="radial", 
                    range = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))

summary(tune.out_with)#we got cost=0.1 and gamma=1 as best parameters
summary(tune.out_less)#we got cost=0.1 and gamma=0.5 as best parameters

out_with<-tune.out_with$best.model
out_less<-tune.out_less$best.model

table(out_with$fitted,df2$Survived) #15.93% error training rates
table(out_less$fitted,df2$Survived) #17.5% error training rates

#Final predictions
pred.with=predict(out_with,df2_t)
pred.less=predict(out_less,df2_t)

results.with<-data.frame(PassengerId=892:1309, Survived=pred.with) 
results.less<-data.frame(PassengerId=892:1309, Survived=pred.less) 

write.csv(results.with,file="svm_results_with.csv", row.names=F) #got 77.27% prediction score on Kaggle
write.csv(results.less,file="svm_results_less.csv", row.names=F) #got 77.03% prediction score on Kaggle
#Title slightly improves model
#Still the best results come from the first model so we will try to improve it by adding Fsize predictor to it
df["Fsize"] <- df$Parch + df$SibSp +1
df_t["Fsize"] <- df_t$Parch + df_t$SibSp +1

tune.out<-tune(svm, Survived~Pclass+Age+Sex+Fare+Embarked+Fsize, data=df, kernel="radial", 
               range = list(cost=c(0.1,1,10,100,1000), gamma=c(0.5,1,2,3,4)))
summary(tune.out)
out<-tune.out$best.model
#we got cost=0.1 and gamma=0.5 as best parameters
table(out$fitted,df$Survived) #16.71% error rates
#Predictions

pred.te=predict(out,df_t)

results3<-data.frame(PassengerId=892:1309, Survived=pred.te) #first submission

write.csv(results3,file="svm_results3.csv", row.names=F) #got 76.79% prediction score on Kaggle



#----------------------- NEW APPROACH -----------
## Loading data
df_t <- read.csv("test.csv")
df <- read.csv("train.csv")


## First look at data

head(df)
tail(df)
summary(df)

#* Survival - need to transfer into factor
#* Pclass - seems okay
#* Name - could extract title from it
#* Sex - need to transfer into numeric factor
#* Age - contains missing values, need to replace them
#* Sibsp and Parch - can get information of family size from here
#* Ticket and Cabin - hard to get information so we will drop it for now
#* Fare - seems okay, some NA's
#* Embarked - need to transfer into numeric factor

# Feature Engineering 

## Age
#Finding the proper distribution.

age<-df[is.na(df$Age)==FALSE, ]$Age
age_t<-df_t[is.na(df_t$Age)==FALSE, ]$Age

descdist(age,discrete=FALSE,boot=1000) 

#Looks like Normal, Lognormal or Gamma distribution but Weibull will also be checked.
fln<-fitdist(age,"lnorm")
fn<-fitdist(age,"norm")
fw<-fitdist(age,"weibull")
fg<-fitdist(age,"gamma")
plot.legenda<-c("Lognormal","Normal","Weibull","Gamma")

#Plotting the density.
denscomp(list(fln,fn,fw,fg),legendtext=plot.legenda)

#Plotting the Q-Q plot.
qqcomp(list(fln,fn,fw,fg),legendtext=plot.legenda)

#Plotting the cumulative distributant plot.
cdfcomp(list(fln,fn,fw,fg),legendtext=plot.legenda)

#Looking at information criteria.
dists <- data.frame("Normal"=c(fn$loglik,fn$aic),"Gamma"=c(fg$loglik, fg$aic),
                    "Weibull"=c(fw$loglik,fw$aic), "Lognormal"=c(fln$loglik,fln$aic),
                    row.names = c("Loglikelihood","AIC"))
dists
#Depending on the results we assume that age is normally distributed
#Now, we will generate from this distribution random number in the place of NA's.
summary(df$Age) #177 NA's
set.seed(1)
df[is.na(df$Age)==TRUE, ]$Age <- round(rnorm(177, mean(age),sd(age)))

summary(df_t$Age) #86 NA's
df_t[is.na(df_t$Age)==TRUE, ]$Age <- round(rnorm(86, mean(age_t),sd(age_t)))
df_t$Age <- round(df_t$Age)

## Sex
df[df$Sex=="male","Sex"]<-(-1)
df[df$Sex=="female","Sex"]<-1
df$Sex<-as.numeric(df$Sex)

df_t[df_t$Sex=="male","Sex"]<-(-1)
df_t[df_t$Sex=="female","Sex"]<-1
df_t$Sex<-as.numeric(df_t$Sex)

## Ticket and Cabin
df$Cabin<-NULL
df$Ticket<-NULL

df_t$Cabin<-NULL
df_t$Ticket<-NULL

## Name
df["Title"]<-sub("\\s.*","",sub(".*,\\s","",df$Name))
df["Title"]<-factor(df$Title)
meaning<-unique(df$Title)
df$Title<-as.numeric(df$Title)
numeric<-unique(df$Title)
meaning <- data.frame(numeric=numeric, meaning=meaning)
t(meaning)
df[df$Title!=9 & df$Title!=12 & df$Title!=13 & df$Title!=14,]$Title <-1 #Other
df[df$Title==12,]$Title <- (-1) #Mr
df[df$Title==13,]$Title <- (-0.5) #Mrs
df[df$Title==9 | df$Title==14,]$Title <- 0.5 #Miss/Ms


df_t["Title"]<-sub("\\s.*","",sub(".*,\\s","",df_t$Name))
df_t["Title"]<-factor(df_t$Title)
meaning_t<-unique(df_t$Title)
df_t$Title<-as.numeric(df_t$Title)
numeric_t<-unique(df_t$Title)
meaning_t <- data.frame(numeric=numeric_t, meaning=meaning_t)
t(meaning_t)
df_t[df_t$Title!=6 & df_t$Title!=7& df_t$Title!=5& df_t$Title!=8,]$Title <-1
df_t[df_t$Title==6,]$Title <- (-1)
df_t[df_t$Title==7,]$Title <- (-0.5)
df_t[df_t$Title==5 | df_t$Title==8,]$Title <- 0.5

df$Name <- NULL
df_t$Name <- NULL


## Embarked
df[df$Embarked=="C",]$Embarked <- (-1)
df[df$Embarked=="Q",]$Embarked <- 0
df[df$Embarked=="S",]$Embarked <- 1
df$Embarked<-as.numeric(df$Embarked)

df_t[df_t$Embarked=="C",]$Embarked <- (-1)
df_t[df_t$Embarked=="Q",]$Embarked <- 0
df_t[df_t$Embarked=="S",]$Embarked <- 1
df_t$Embarked<-as.numeric(df_t$Embarked)
summary(df)

#We got 2 NA's, will replace them with random number

df[is.na(df$Embarked)==TRUE,]$Embarked <- sample(c(1,2,3),2)
summary(df)

## Fare
#Getting rid off NA in Fare test data
summary(df_t)
fare <- df_t[is.na(df_t$Fare)==FALSE,]$Fare
df_t[is.na(df_t$Fare)==TRUE,]$Fare<-rnorm(1,mean(fare),sd(fare))

## Family Size
df["Fsize"]<-df$Parch + df$SibSp + 1
df_t["Fsize"] <- df_t$Parch + df_t$SibSp +1
df[df$Fsize==1,]$Fsize <- (-1) #single
df[df$Fsize>1 & df$Fsize<5,]$Fsize <- 0 #family
df[df$Fsize>=5,]$Fsize <- 1 #large family

df_t[df_t$Fsize==1,]$Fsize <- (-1) #single
df_t[df_t$Fsize>1 & df_t$Fsize<5,]$Fsize <- 0 #family
df_t[df_t$Fsize>=5,]$Fsize <- 1 #large family

#Factorize Survived
df$Survived <- factor(df$Survived)

# Exploratory Data Analysis (EDA)
#Creating correlogram to see relationships between features.
df$Survived<-as.factor(df$Survived)
ggpairs(df,columns=c(1,3:11),ggplot2::aes(colour=Survived))

## Pclass
ggplot(df, aes(x=Pclass, fill=Survived)) + geom_histogram(stat="count")
#We see that in the 1st class there were more survivors and in the 3rd class the number of non-survivors is relatively high comparing to the number of survivors.


## Sex
ggplot(df, aes(x=Sex, fill=Survived)) + geom_histogram(stat="count")
#Most of the survivors were females. This is a well-known fact that women with children are in the first place to be saved.

## Age
ggplot(df, aes(x=Age, fill=Survived)) + geom_density(alpha=0.5)
#Most people who did survive were kids and adults, while the rest were mostly young adults and in the elderly age.

## Fare
ggplot(df, aes(x=Fare, fill=Survived)) + geom_density(alpha=0.5)
#The less is the charge the less survivors. It might be caused with a fact that well paid cabins could be more durable or be placed closer to rescue boats.

## Embarked
ggplot(df, aes(x=Embarked, fill=Survived)) + geom_histogram(stat="count")
#The are more survivors in Cherbourg but less in Queenstown or Southampton. Let's see if there is some relationship with Fares.
ggplot(df,aes(x=Fare,fill=factor(Embarked))) + geom_density(alpha=0.5)
#As we can see in Queenstown or Southampton most Fares are in the low interval while in the Cherbourg we may notice that Fares were aswell low as high so the reason standing behind more survivors in Cherbourg are probably better paid cabins.

## Title
ggplot(df, aes(x=Title, fill=Survived)) + geom_histogram(stat="count")
#We see that most of the non-survivors were title Mr. while survivors - Ms. or Mrs..

## Fsize
ggplot(df, aes(x=Fsize, fill=Survived)) + geom_histogram(stat="count")
#In the families of size 2-4 there were more survivors and in the families of size 1 or 5 and more there were more non-survivors. Most families of size 2-4 consists of married couples and children and as we said, they were a priority to be saved.

# Creating SVM model
#Now we will perform SVM using radial kernel and choosing best parameters with tune().
tune.out<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+Title+Age+Fsize, data=df,
               kernel="radial",type="C-classification", 
               range = list(cost=10^(-2:2),gamma=10^(-5:5)))


summary(tune.out)
svm_fit<-tune.out$best.model

#Checking training error rates.
table(svm_fit$fitted,df$Survived)

#Making final prediction and saving result into data frame.
prediction=predict(svm_fit,df_t)
svm_p<-data.frame(PassengerId=892:1309, Survived=prediction) 


#Writing into .csv file ready for a submission on Kaggle.
write.csv(svm_p,file="svm_prediction.csv", row.names=F)



#--------SEEKING FOR IMPROVEMENTS-------------
#splitting train data  df2 into train and test using 10-fold cross validation
#randomly shuffle the data
td<-df[sample(nrow(df)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(td)),breaks=10,labels=FALSE)

#Perform 10 fold cross validation
for(i in 1:10){
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  td.test <- td[testIndexes, ]
  td.train <- td[-testIndexes, ]
  #Use the test and train data partitions however you desire...
}
mod<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+Title+Age+Fsize, data=td.train,
               kernel="radial",type="C-classification", 
               range = list(cost=10^(-2:2),gamma=10^(-5:5)))
mod1<-mod$best.model
p=predict(mod1,td.test)
w<-table(p,td.test$Survived)
1-(w[2,1]+w[1,2])/sum(w)

#Adding Age bins
df["AgeBin"]<-NA
df[df$Age < 18,]$AgeBin <- (-1) #kid
df[df$Age >=18 & df$Age<50,]$AgeBin <- 0 #adult
df[df$Age>=50,]$AgeBin <- 1 #elderly

df_t["AgeBin"]<-NA
df_t[df_t$Age < 18,]$AgeBin <- (-1) #kid
df_t[df_t$Age >=18 & df_t$Age<50,]$AgeBin <- 0 #adult
df_t[df_t$Age>=50,]$AgeBin <- 1 #elderly


mod<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+AgeBin+Fsize, data=td.train,
          kernel="radial",type="C-classification", 
          range = list(cost=10^(-2:2),gamma=10^(-5:5)))
mod1<-mod$best.model
p=predict(mod1,td.test)
w<-table(p,td.test$Survived)
1-(w[2,1]+w[1,2])/sum(w)

#Seeing if improves
tune.out2<-tune(svm, Survived~Pclass+Sex+Fare+Embarked+Title+AgeBin+Fsize, data=df,
               kernel="radial",type="C-classification", 
               range = list(cost=10^(-2:2),gamma=10^(-5:5)))


summary(tune.out2)
svm_fit2<-tune.out2$best.model

#Checking training error rates.
table(svm_fit2$fitted,df$Survived)

#Making final prediction and saving result into data frame.
prediction2=predict(svm_fit2,df_t)
svm_p2<-data.frame(PassengerId=892:1309, Survived=prediction2) 


#Writing into .csv file ready for a submission on Kaggle.
write.csv(svm_p2,file="svm_prediction2.csv", row.names=F)


