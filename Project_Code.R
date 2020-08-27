#Imports
library(leaps)
library(glmnet)
library(dplyr)
library(gam)
library(splines)
library (gbm)
library(ggplot2)
library(nnet)
library(boot)
library(MASS)
library(tree)
library(randomForest)
library(caret)



#Data Cleaning
loans = read.csv("loan.csv", nrows = 10000)
loans = loans[, colSums(is.na(loans)) < nrow(loans)/2]
loans = Filter(function(x) length(unique(x))>1, loans)
head(loans)




loans=loans[ , which(names(loans) %in% c("funded_amnt", "term", "int_rate", "emp_length", "home_ownership", "annual_inc", "verification_status",  "purpose", "delinq_2yrs", "earliest_cr_line", "open_acc", "pub_rec", "revol_bal", "initial_list_status", "acc_open_past_24mths", "avg_cur_bal", "num_sats", "pub_rec_bankruptcies"))]

# turn date format into an int
dates = paste('01', loans$earliest_cr_line)
loans$earliest_cr_line = as.Date(dates, format = "%d %b-%Y")
loans$earliest_cr_line = as.integer(loans$earliest_cr_line)

# emp_length had some N/A stored as a string
loans = loans[!(loans$emp_length=="n/a"),]
loans$emp_length = factor(loans$emp_length)
loans = na.omit(loans)


head(loans)



#Data Exploration

# term vs. int_rate 

ggplot(data=loans, aes(x=term, y=int_rate)) +
  geom_boxplot()

# funded amount vs. int_rate 

ggplot(data=loans, aes(x=emp_length, y=int_rate)) +
  geom_boxplot()



# home_ownership vs. int_rate

ggplot(data=loans, aes(x=home_ownership, y=int_rate)) +
  geom_boxplot()


# annual_inc vs. int_rate

ggplot(data=loans, aes(x=annual_inc, y=int_rate)) +
  geom_point()


plot(loans[,c(1,3,6, 9,10)])
plot(loans[,c(3,11, 15:17)])
for (col in c(1,3,6, 9:11, 15:17)){
  hist(loans[,col], main = names(loans[col]))
}
for(col in c(2,4,5,7,8,12,14,18)){
  print(names(loans[col]))
  print(table(loans[,col]))
}



# Variable Selection



set.seed(2)
loans = na.omit(loans)
train = sample(nrow(loans), nrow(loans)/2)
loans_train=loans[train,]
loans_test=loans[-train,]



# Forward stepwise

predict.regsubsets =function (object ,newdata ,id ,...){
  form=as.formula (object$call [[2]])
  mat=model.matrix (form ,newdata )
  coefi =coef(object ,id=id)
  xvars =names (coefi )
  mat[,xvars ]%*% coefi
}

forwardm = regsubsets(int_rate~., data = loans_train, method = "forward", nvmax = ncol(loans)-1)

val.errors = rep(NA, ncol(loans)-1)
for (i in 1:(ncol(loans)-1)) {
  pred = predict.regsubsets(forwardm, loans_test, i)
  val.errors[i] = mean((loans_test$int_rate - pred)^2)
}
min(val.errors)


# Backward stepwise

backwardm = regsubsets(int_rate~., data = loans_train, method = "backward", nvmax = ncol(loans)-1)

val.errors = rep(NA, ncol(loans)-1)
for (i in 1:(ncol(loans)-1)) {
  pred = predict.regsubsets(backwardm, loans_test, i)
  val.errors[i] = mean((loans_test$int_rate - pred)^2)
}

min(val.errors)


#Ridge Regression

library(glmnet)

x = model.matrix(int_rate ~ ., loans)[,-1]
y = loans$int_rate

cv.out = cv.glmnet(x[train,], y[train], alpha = 0)
bestlam = cv.out$lambda.min
ridgem = glmnet(x[train,],y[train],alpha =0, lambda =bestlam)
ridge.pred = predict(ridgem, s = bestlam, newx = x[-train,])
mean((y[-train] - ridge.pred)^2)


#LASSO

cv.out = cv.glmnet(x[train,], y[train], alpha = 1)
bestlam = cv.out$lambda.min
lassom = glmnet(x[train,],y[train],alpha =1, lambda =bestlam)
lasso.pred = predict(lassom, s = bestlam, newx = x[-train,])
mean((y[-train] - lasso.pred)^2)





coef(forwardm, which.min(val.errors))

loans = loans[,-c(4,11,12,16,18,19)]
#changing character data to factors
loans$home_ownership = as.factor(loans$home_ownership)
loans$term = as.factor(loans$term)
loans$verification_status = as.factor(loans$verification_status)
loans$purpose = as.factor(loans$purpose)
loans$initial_list_status = as.factor(loans$initial_list_status)





# Model exploration


set.seed(2)
loans = na.omit(loans)
trainsm = sample(nrow(loans), 500)

#filtering out variables that are not in both test and train data
loans = filter(loans, home_ownership != 'ANY')
loans = filter(loans, !purpose %in% c('house', 'renewable_energy', 'moving'))


loans_trainsm=loans[trainsm,]
loans_testsm=loans[-trainsm,]
trainsm = sample(nrow(loans), 500)
loans_trainsm=loans[trainsm,]
loans_testsm=loans[-trainsm,]

trainmed=sample(nrow(loans), 1000)
loans_trainmed=loans[trainmed,]
loans_testmed=loans[-trainmed,]

trainlar = sample(nrow(loans), nrow(loans)/2)
loans_trainlar=loans[trainlar,]
loans_testlar=loans[-trainlar,]



#Multiple Regression

mlrsm = glm(data = loans_trainsm, int_rate~.)
mlrsmte = mean((loans_testsm$int_rate - predict(mlrsm, loans_testsm))^2)
mlrsmte
summary(mlrsm)
mlrmed = glm(data = loans_trainmed, int_rate~.)
mlrmedte = mean((loans_testmed$int_rate - predict(mlrmed, loans_testmed))^2)
mlrmedte
summary(mlrmed)
mlrlar = glm(data = loans_trainlar, int_rate~.)
mlrlarte = mean((loans_testlar$int_rate - predict(mlrlar, loans_testlar))^2)
mlrlarte
summary(mlrlar)



#Polynomial Regression


poly_degrees = c(2,3,5,10)
cv.error = rep(0,4)
for (i in 1:4){
  glm.fit = glm(data = loans, int_rate~poly(funded_amnt, poly_degrees[i])+ poly(annual_inc,poly_degrees[i])+poly(delinq_2yrs, poly_degrees[i]) + poly(earliest_cr_line, poly_degrees[i])+ poly(revol_bal, poly_degrees[i])+ poly(acc_open_past_24mths, poly_degrees[i])+poly(num_sats, poly_degrees[i]) + term + home_ownership + verification_status+purpose+initial_list_status)
  cv.error[i] = cv.glm(loans, glmfit = glm.fit, K=5)$delta[1]
}
cv.error
#Minimum is polynomial of degree 3
plm.fitsm= glm(data = loans_trainsm, int_rate~poly(funded_amnt, 3)+ poly(annual_inc,3)+poly(delinq_2yrs, 3) + poly(earliest_cr_line, 3)+ poly(revol_bal, 3)+ poly(acc_open_past_24mths, 3)+poly(num_sats, 3) + term + home_ownership + verification_status+purpose+initial_list_status)

plm.fitmed = glm(data = loans_trainmed, int_rate~poly(funded_amnt, 3)+ poly(annual_inc,3)+poly(delinq_2yrs, 3) + poly(earliest_cr_line, 3)+ poly(revol_bal, 3)+ poly(acc_open_past_24mths, 3)+poly(num_sats, 3) + term + home_ownership + verification_status+purpose+initial_list_status)

plm.fitlar = glm(data = loans_trainlar, int_rate~poly(funded_amnt, 3)+ poly(annual_inc,3)+poly(delinq_2yrs, 3) + poly(earliest_cr_line, 3)+ poly(revol_bal, 3)+ poly(acc_open_past_24mths, 3)+poly(num_sats, 3) + term + home_ownership + verification_status+purpose+initial_list_status)


plmsmte = mean((loans_testsm$int_rate - predict(plm.fitsm, loans_testsm))^2)
plmsmte
summary(plm.fitsm)
plmmedte = mean((loans_testmed$int_rate - predict(plm.fitmed, loans_testmed))^2)
plmmedte
summary(plm.fitmed)
plmlarte = mean((loans_testlar$int_rate - predict(plm.fitlar, loans_testlar))^2)
plmlarte
summary(plm.fitlar)



# Regression Tree


treesm = tree(int_rate~., data = loans_trainsm)
plot(treesm)
text(treesm, pretty=0)
preds = predict(treesm, newdata = loans_testsm)
mean((loans_testsm$int_rate - preds)^2)


treemed = tree(int_rate~., data = loans_trainmed)
plot(treemed)
text(treemed, pretty=0)
preds = predict(treemed, newdata = loans_testmed)
mean((loans_testmed$int_rate - preds)^2)


treelar = tree(int_rate~., data = loans_trainlar)
plot(treelar)
text(treelar, pretty=0)
preds = predict(treelar, newdata = loans_testlar)
mean((loans_testlar$int_rate - preds)^2)




cv.treesm = cv.tree(treesm)
plot(cv.treesm$size, cv.treesm$dev, type = 'b')
cv.treemed = cv.tree(treemed)
plot(cv.treemed$size, cv.treemed$dev, type = 'b')
cv.treelar = cv.tree(treelar)
plot(cv.treelar$size, cv.treelar$dev, type = 'b')



#the largest is just the full model.

prune.treesm = prune.tree(treesm, best = 2)
plot(prune.treesm)
text(prune.treesm, pretty=0)
preds = predict(prune.treesm, newdata = loans_testsm)
treesmte = mean((loans_testsm$int_rate - preds)^2)
treesmte
summary(prune.treesm)

prune.treemed = prune.tree(treemed, best = 4)
plot(prune.treemed)
text(prune.treemed, pretty=0)
preds = predict(prune.treemed, newdata = loans_testmed)
treemedte = mean((loans_testmed$int_rate - preds)^2)
treemedte
summary(prune.treemed)

plot(treelar)
text(treelar, pretty=0)
preds = predict(treelar, newdata = loans_testlar)
treelarte = mean((loans_testlar$int_rate - preds)^2)
treelarte
summary(treelar)



#Random Forests

set.seed(1)

rfmed = randomForest(int_rate~., data = loans_trainmed)
yhat.rfmed = predict(rfmed, newdata = loans_testmed)
mean((yhat.rfmed - loans_testmed$int_rate)^2)

rfsm = randomForest(int_rate~., data = loans_trainsm)
yhat.rfsm = predict(rfsm, newdata = loans_testsm)
mean((yhat.rfsm - loans_testsm$int_rate)^2)

rfmain = randomForest(int_rate~., data = loans_trainlar)
yhat.rf = predict(rfmain, newdata = loans_testlar)
mean((yhat.rf - loans_testlar$int_rate)^2)



#Tuning M
set.seed(1)
bestm = tuneRF(loans_trainlar[,-3], loans_trainlar$int_rate, stepFactor = 2)
bestmmed = tuneRF(loans_trainmed[,-3], loans_trainmed$int_rate, stepFactor = 2)
bestmsm = tuneRF(loans_trainsm[,-3], loans_trainsm$int_rate, stepFactor = 2)
bestm
bestmmed
bestmsm


set.seed(1)
rfmain = randomForest(int_rate~., data = loans_trainlar, mtry = 2)
yhat.rf = predict(rfmain, newdata = loans_testlar)
summary(rfmain)
rflarte = mean((yhat.rf - loans_test$int_rate)^2)
rflarte

rfmed = randomForest(int_rate~., data = loans_trainmed, mtry = 2)
yhat.rfmed = predict(rfmed, newdata = loans_testmed)
summary(rfmed)
rfmedte = mean((yhat.rfmed - loans_testmed$int_rate)^2)
rfmedte

rfsm = randomForest(int_rate~., data = loans_trainsm, mtry =8)
yhat.rfsm = predict(rfsm, newdata = loans_testsm)
summary(rfsm)
rfsmte = mean((yhat.rfsm - loans_testsm$int_rate)^2)
rfsmte


# GAM


# GAM n = 500

gam_small = gam(int_rate ~ funded_amnt + term + home_ownership + s(annual_inc, df = 4) + verification_status + purpose + delinq_2yrs + earliest_cr_line + revol_bal + initial_list_status + acc_open_past_24mths + num_sats, data = loans_trainsm)
summary(gam_small)




gam_small.preds=predict(gam_small,newdata = loans_testsm)
gamsmte = mean((loans_testsm$int_rate - gam_small.preds) ^ 2)
gamsmte




# GAM n = 1000

gam_med = gam(int_rate ~ funded_amnt + term + home_ownership + s(annual_inc, df = 4) + verification_status + purpose + delinq_2yrs + earliest_cr_line + revol_bal + initial_list_status + acc_open_past_24mths + num_sats, data = loans_trainmed)
summary(gam_med)




gam_med.preds=predict(gam_med,newdata = loans_testmed)
gammedte = mean((loans_testmed$int_rate - gam_med.preds) ^ 2)
gammedte




# GAM n = 4591


gam_large = gam(int_rate ~ funded_amnt + term + home_ownership + s(annual_inc, df = 4) + verification_status + purpose + delinq_2yrs + earliest_cr_line + revol_bal + initial_list_status + acc_open_past_24mths + num_sats, data = loans_trainlar)
summary(gam_large)




gam_large.preds=predict(gam_large, newdata = loans_testlar)
gamlarte = mean((loans_testlar$int_rate - gam_large.preds) ^ 2)
gamlarte





# Boosting 


# Boosting n = 500

set.seed(1)
lambda.seq = seq(from = .01, to = .05, by = .01)
trees.seq = seq(from = 500, to =  1000, by = 100)
depth.seq = seq(from=1, to = 5, by  = 1)
index= 0
test.errs = numeric(125)

for (i in 1:length(lambda.seq)) {
  for(j in 1: length(trees.seq)){
    for(k in 1:length(depth.seq)){
      combo = c(lambda.seq[i], trees.seq[j], depth.seq[k])
      index = index +1 
      boost.interest = gbm(int_rate~., data = loans_trainsm, distribution = "gaussian",
                           n.trees = trees.seq[j], shrinkage = lambda.seq[i], interaction.depth = depth.seq[k])
      yhat.boost = predict(boost.interest, newdata = loans_testsm, n.trees = trees.seq[j])
      test.errs[index] = mean((yhat.boost - loans_testsm$int_rate)^2)
    }
  }
}
#plot(lambda.seq, test.errs, type = "b")

tab_matrix = matrix(nrow = 3, ncol = 2)
colnames(tab_matrix) = c("lambda", "test error")
rownames(tab_matrix) = c("training n=500", "training n=1000", "training n=4591")
tab_matrix[1,1] = lambda.seq[which.min(test.errs)]
tab_matrix[1,2] = min(test.errs)


which.min(test.errs)
boostsmte = min(test.errs)
boostsmte





set.seed(1)
test.errs = numeric(length(lambda.seq))

for (i in 1:length(lambda.seq)) {
  boost.interest = gbm(int_rate~., data = loans_trainmed, distribution = "gaussian",
                       n.trees = 1000, shrinkage = lambda.seq[i])
  yhat.boost = predict(boost.interest, newdata = loans_testmed, n.trees = 1000)
  test.errs[i] = mean((yhat.boost - loans_testmed$int_rate)^2)
}
plot(lambda.seq, test.errs, type = "b")

tab_matrix[2,1] = lambda.seq[which.min(test.errs)]
tab_matrix[2,2] = min(test.errs)

lambda.seq[which.min(test.errs)]
boostmedte = min(test.errs)
boostmedte




set.seed(1)
test.errs = numeric(length(lambda.seq))

for (i in 1:length(lambda.seq)) {
  boost.interest = gbm(int_rate~., data = loans_trainlar, distribution = "gaussian",
                       n.trees = 1000, shrinkage = lambda.seq[i])
  yhat.boost = predict(boost.interest, newdata = loans_testlar, n.trees = 1000)
  test.errs[i] = mean((yhat.boost - loans_testlar$int_rate)^2)
}
plot(lambda.seq, test.errs, type = "b")

tab_matrix[3,1] = lambda.seq[which.min(test.errs)]
tab_matrix[3,2] = min(test.errs)

lambda.seq[which.min(test.errs)]
boostlarte = min(test.errs)
boostlarte




tab_matrix



# Neural Networks

# Neural Networks n = 500


set.seed(1)
size.seq = seq(1,35, by = 1)
test.errs = numeric(length(size.seq))

for(i in 1:length(size.seq)){
  loans.nn = nnet(int_rate~., data = loans_trainsm, size = size.seq[i],
                  linout = TRUE, trace = FALSE)
  
  yhat.nn = predict(loans.nn, newdata = loans_testsm)
  test.errs[i] = mean((yhat.nn - loans_testsm$int_rate)^2)
}

colnames(tab_matrix) = c("size", "test error")
rownames(tab_matrix) = c("training n=500", "training n=1000", "training n=4591")

tab_matrix[1,1] = size.seq[which.min(test.errs)]
tab_matrix[1,2] = min(test.errs)
nnsmte = min(test.errs)
nnsmte
plot(size.seq, test.errs, type = "b")




set.seed(1)
test.errs = numeric(length(size.seq))

for(i in 1:length(size.seq)){
  loans.nn = nnet(int_rate~., data = loans_trainmed, size = size.seq[i],
                  linout = TRUE, trace = FALSE)
  
  yhat.nn = predict(loans.nn, newdata = loans_testmed)
  test.errs[i] = mean((yhat.nn - loans_testmed$int_rate)^2)
}


tab_matrix[2,1] = size.seq[which.min(test.errs)]
tab_matrix[2,2] = min(test.errs)

size.seq[which.min(test.errs)]
nnmedte = min(test.errs)
nnmedte
plot(size.seq, test.errs, type = "b")





set.seed(1)
test.errs = numeric(length(size.seq))

for(i in 1:length(size.seq)){
  loans.nn = nnet(int_rate~., data = loans_trainlar, size = size.seq[i],
                  linout = TRUE, trace = FALSE)
  
  yhat.nn = predict(loans.nn, newdata = loans_testlar)
  test.errs[i] = mean((yhat.nn - loans_testlar$int_rate)^2)
}


tab_matrix[3,1] = size.seq[which.min(test.errs)]
tab_matrix[3,2] = min(test.errs)

size.seq[which.min(test.errs)]
nnlarte = min(test.errs)
nnlarte
plot(size.seq, test.errs, type = "b")




tab_matrix




#Comparison visualization

Models = rep(c('Multiple Reg', 'Poly Reg', 'Reg Tree', 'Ran Forests', 'GAM', 'Boosting', 'Neural Network'), each = 3)
Size = rep(c('Small', 'Medium', 'Large'), 7)
testerrors = c(mlrsmte, mlrmedte, mlrlarte, plmsmte, plmmedte, plmlarte, treesmte, treemedte, treelarte, rfsmte, rfmedte, rflarte, gamsmte, gammedte, gamlarte, boostsmte, boostmedte, boostlarte, nnsmte, nnmedte, nnlarte)
results = as.data.frame(cbind(Models, Size, testerrors))
results$testerrors = as.numeric(results$testerrors)
resmeans = aggregate(results[, 3], list(results$Models), mean)
resmeans2 = aggregate(results[, 3], list(results$Size), mean)
results


ggplot(data = resmeans) + 
  geom_col(aes(x=Group.1, y=x), fill = "#00abff") + geom_text(aes(x=Group.1, y=x,label=round(x,3)), position=position_dodge(width=0.9), vjust=-0.25) + labs(x = 'Type of Model', y = 'Avg Test Error')
ggplot(data = resmeans2) + 
  geom_col(aes(x=Group.1, y=x), fill = "#00abff") + geom_text(aes(x=Group.1, y=x,label=round(x,3)), position=position_dodge(width=0.9), vjust=-0.25) + labs(x = 'Training Data Size', y = 'Avg Test Error')
ggplot(data = results) + geom_bar(aes(x = Models, y = testerrors, fill = Size), position = 'dodge', stat = 'identity') + labs(x = 'Type of Model', y = 'Avg Test Error') + scale_fill_discrete(name = 'Size of Training Data')



ggplot(data = results[-c(4,5,6),]) + geom_bar(aes(x = Models, y = testerrors, fill = Size), position = 'dodge', stat = 'identity') + labs(x = 'Type of Model', y = 'Avg Test Error')+ scale_fill_discrete(name = 'Size of Training Data')



