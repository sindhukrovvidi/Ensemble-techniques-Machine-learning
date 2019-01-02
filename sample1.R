library('caret')
set.seed(1)
data<-read.csv("loan.csv", header=T, na.strings=c("","NA"))
#data<-read.csv(file="loan.csv",header = TRUE)
summary(data)
library(plyr)
data$Dependents <- revalue(data$Dependents, c("3+"="3"))
#library(mice)
#imp <- mice(data)
#data <- complete(imp,2)
sum(is.na(data))
sapply(data, function(x) sum(is.na(x)))
preProcValues <- preProcess(data, method = c("medianImpute","center","scale"))
data_processed <- predict(preProcValues, data)
sapply(data_processed, function(x) sum(is.na(x)))
index <- createDataPartition(data_processed$Loan_Status, p=0.75, list=FALSE)
trainSet <- data_processed[ index,]
testSet <- data_processed[-index,]

fitControl <- trainControl(method = "cv", number = 5, savePredictions = 'final', classProbs = T)
outcomeName<-'Loan_Status'
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome","CoapplicantIncome")
model_dt<-train(trainSet[,predictors],trainSet[,outcomeName],method='rpart',trControl=fitControl,tuneLength=3)
#plot(model_dt)
#rpart.plot(model_dt)
testSet$pred_dt<-predict(object = model_dt,testSet[,predictors])
cm <- confusionMatrix(testSet$Loan_Status,testSet$pred_dt)
b<-cm$overall['Accuracy']
print(b)
#prp(testSet$pred_dt, box.palette="Reds", tweak=1.2)
#prp(pred_dt, box.palette="Reds", tweak=1.2)
#rpart.plot(testSet$pred_dt)
#getRpartModel(model_dt,data)
#rpart.plot(testSet$pred_dt)
model_knn<-train(trainSet[,predictors],trainSet[,outcomeName],method='knn',trControl=fitControl,tuneLength=3)

testSet$pred_knn<-predict(object = model_knn,testSet[,predictors])
cm <- confusionMatrix(testSet$Loan_Status,testSet$pred_knn)
a<-cm$overall['Accuracy']
print(a)
model_svm<-train(trainSet[,predictors],trainSet[,outcomeName],method='svmLinear',trControl=fitControl,tuneLength=3)
testSet$pred_svm<-predict(object = model_svm,testSet[,predictors])
cm <-confusionMatrix(testSet$Loan_Status,testSet$pred_svm)
c<-cm$overall['Accuracy']
print(c)

trainSet$OOF_pred_dt<-model_dt$pred$Y[order(model_dt$pred$rowIndex)]
trainSet$OOF_pred_knn<-model_knn$pred$Y[order(model_knn$pred$rowIndex)]
trainSet$OOF_pred_svm<-model_svm$pred$Y[order(model_svm$pred$rowIndex)]
testSet$OOF_pred_dt<-predict(model_dt,testSet[predictors],type='prob')$Y
testSet$OOF_pred_knn<-predict(model_knn,testSet[predictors],type='prob')$Y
testSet$OOF_pred_svm<-predict(model_svm,testSet[predictors],type='prob')$Y
predictors_top<-c('OOF_pred_dt','OOF_pred_knn','OOF_pred_svm')
model_dct<-train(trainSet[,predictors_top],trainSet[,outcomeName],method='rpart',trControl=fitControl,tuneLength=3)
testSet$dct_stacked<-predict(model_dct,testSet[,predictors_top])
cm <- confusionMatrix(testSet$Loan_Status,testSet$dct_stacked)
d <-cm$overall["Accuracy"]
print(d)

#plotting different models accuracy for comparision 
df<-c("Dt","knn","svm","comb")
df1<-c(b,a,c,d)
cols <- c("red","lightblue","green","yellow")
barplot(df1,names.arg = df,xlab="Algorithms",ylab="Accuracy",col=cols, ylim = c(0,1*1), main="Comparision of algorithms")

