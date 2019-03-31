library(tidyverse)
library(ranger)
library(Boruta)
library(caret)
library(parallel)
library(foreach)
library(iterators)
library(doParallel)
library(snow)
library(doSNOW)
library(readxl)
library(rpart)
library(adabag)
library(DiagrammeR)
library(mxnet)
library(gbm)

# load data
sqAnoDf <- read_tsv(file = "anonymized-sq-dataset.tsv")

#Separate dataset par each media ----------------------------------------
medias <- unique(sqAnoDf$MEDIA)
comDdata <- list()
data_media<-list()

for(i in 1:length(medias)){
  
  data_media[[i]]<-sqAnoDf %>%
    filter(MEDIA==medias[i])
}

df<-list()

for( i in 1:length(medias)){
  df[[i]] <- data_media[[i]] %>%
    dplyr::mutate(LABEL = as.factor(LABEL))%>%
    dplyr::select(responsesOfInterest,
                  featuresOfInterest)
}


# Separate training data and test data ------------------------------------

df.train<-vector("list",4)
df.test<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

for(i in 1:length(medias)) {
  
  partition_indexes <- createDataPartition(df[[i]]$LABEL, times = 1, p = 0.75, list = FALSE)
  df.train[[i]] <- df[[i]][partition_indexes, ]
  df.test[[i]] <- df[[i]][-partition_indexes, ]
}

# feature selection -------------------------------------------------------

resboruta.train<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

ptm <- Sys.time()
for(i in 1:length(medias)){
  resboruta.train[[i]] <- Boruta(x = df.train[[i]][, featuresOfInterest],
                           y = df.train[[i]][, responsesOfInterest[indResp]][[1]],
                           doTrace = 2,
                           num.threads = nCores - 1,
                           ntree = 200)
}
timer(ptm)

for(i in 1:length(medias)){
  print(resboruta.train[[i]])
}



# Data containing only important variables --------------------------------

trainDf<-vector("list",4)
testDf<-vector("list",4)

for(i in 1:length(medias)){
  
  selected<-getSelectedAttributes(resboruta.train[[i]])
  trainDf[[i]]<-df.train[[i]][c("LABEL",selected)]
  testDf[[i]]<-df.test[[i]][c("LABEL",selected)]
}

# Timing function --------------------------
ptm <- Sys.time()
timer <- function(start_time) {
  start_time <- as.POSIXct(start_time)
  dt <- difftime(Sys.time(), start_time, units="secs")
  format(.POSIXct(dt,tz="GMT"), "%H:%M:%S")
}
timer(ptm)

# Knn algorithm for each media --------------------------------------------
caret_knn<-vector("list",4)
preds_knn<-vector("list",4)
confusionMatrix_knn<-vector("list",4)


if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid",classProbs = TRUE)
tune.grid.knn <- expand.grid(k = c(2,3,4,5,6,7,8,9,10,11,12))
caret_knn <- foreach(i=1:length(medias)) %dopar% {
caret::train(LABEL ~ ., data = trainDf[[i]],preProcess=c("center","scale"),tuneGrid=tune.grid.knn,method = "knn",trControl = train.control)
 }
timer(ptm)

stopCluster(myCluster)

caret_knn[1]


foreach(i=1:length(medias))%do% {
preds_knn[[i]] <- predict(caret_knn[[i]], testDf[[i]])
confusionMatrix_knn[[medias[i]]]<-confusionMatrix(preds_knn[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_knn<-dotplot(resamples(list(M0 = caret_knn[[1]],M1=caret_knn[[2]],M2=caret_knn[[3]],M3=caret_knn[[4]])),metric="Accuracy")

# XgbTree algorithm for each media ----------------------------------------
caret_xgb<-vector("list",4)
preds_xgb<-vector("list",4)
confusionMatrix_xgb<-vector("list")

tune.grid.xgb <- expand.grid(eta = c(0.05, 0.075, 0.1),
                             nrounds = c(50, 75, 100),
                             max_depth = 6:8,
                             min_child_weight = c(2.0, 2.25, 2.5),
                             colsample_bytree = c(0.3, 0.4, 0.5),
                             gamma = 0:3,
                             subsample = 1
                             )


myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
caret_xgb<-foreach(i = 1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]], tuneGrid=tune.grid.xgb,method = "xgbTree", trControl = train.control)
}
timer(ptm)

stopCluster(cl)

caret_xgb

foreach(i = 1:length(medias))%do%{
  preds_xgb[[i]] <- predict(caret_xgb[[medias[i]]], testDf[[i]])
  confusionMatrix_xgb[[medias[i]]]<-confusionMatrix(preds_xgb[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_xgb<-dotplot(resamples(list(M0 = caret_xgb[["M0"]],M1=caret_xgb[["M1"]],M2=caret_xgb[["M2"]],M3=caret_xgb[["M3"]])),metric="Accuracy")


# Model performance summary and comparison --------------------------------

cv.values = resamples(list(knn_M0 = caret_knn[[1]],xgb_M0=caret_xgb[["M0"]],knn_M1 = caret_knn[[2]],xgb_M1=caret_xgb[["M1"]],
                           knn_M2 = caret_knn[[3]],xgb_M2=caret_xgb[["M2"]],knn_M3 = caret_knn[[4]],xgb_M3=caret_xgb[["M3"]]))
summary(cv.values)

dotplot(cv.values,metric = "Accuracy")


# Ranger algorithm for each media  -------------------------------------------

ran.params <- getModelInfo("ranger")
ran.params$ranger$parameters

caret_ran<-vector("list",4)
preds_ran<-vector("list",4)
confusionMatrix_ran<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid",classProbs = TRUE)
tune.grid.ran <- expand.grid(min.node.size = c(2,3,5,6,7,8),
                             mtry=c(6,7,8,9,10,20,21,22),
                             splitrule="gini"
                             )
caret_ran <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],tuneGrid=tune.grid.ran,method = "ranger",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_ran


foreach(i=1:length(medias))%do% {
  preds_ran[[i]] <- predict(caret_ran[[i]], testDf[[i]])
  confusionMatrix_ran[[medias[i]]]<-confusionMatrix(preds_ran[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_ran<-dotplot(resamples(list(M0 = caret_ran[[1]],M1=caret_ran[[2]],M2=caret_ran[[3]],M3=caret_ran[[4]])),metric="Accuracy")


# Rf algorithm for each media -----------------------------------------

rf.params <- getModelInfo("rf")
rf.params$rf$parameters

caret_rf<-vector("list",4)
preds_rf<-vector("list",4)
confusionMatrix_rf<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid",classProbs = TRUE)
tune.grid.rf <- expand.grid(mtry=c(4,5,6,7,8,24,25,26,27))
caret_rf <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],tuneGrid=tune.grid.rf,method = "rf",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_rf


foreach(i=1:length(medias))%do% {
  preds_rf[[i]] <- predict(caret_rf[[i]], testDf[[i]])
  confusionMatrix_rf[[medias[i]]]<-confusionMatrix(preds_rf[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_rf<-dotplot(resamples(list(M0 = caret_rf[[1]],M1=caret_rf[[2]],M2=caret_rf[[3]],M3=caret_rf[[4]])),metric="Accuracy")



# Rpart algorithm for each media -----------------------------------------

rpart.params <- getModelInfo("rpart")
rpart.params$rpart$parameters

caret_rp<-vector("list",4)
preds_rp<-vector("list",4)
confusionMatrix_rp<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid",classProbs = TRUE)
caret_rp <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],method = "rpart",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_rp


foreach(i=1:length(medias))%do% {
  preds_rp[[i]] <- predict(caret_rp[[i]], testDf[[i]])
  confusionMatrix_rp[[medias[i]]]<-confusionMatrix(preds_rp[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_rp<-dotplot(resamples(list(M0 = caret_rp[[1]],M1=caret_rp[[2]],M2=caret_rp[[3]],M3=caret_rp[[4]])),metric="Accuracy")



# Glmnet algorithm for each media -----------------------------------------

glm.params <- getModelInfo("glmnet")
glm.params$glmnet$parameters

caret_glm<-vector("list",4)
preds_glm<-vector("list",4)
confusionMatrix_glm<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid")
tune.grid.glm <- expand.grid(alpha = c(0,1),
                             lambda=seq(0,0.5,0.001))
caret_glm <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],preProcess=c("center","scale"),tuneGrid=tune.grid.glm,method = "glmnet",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_glm


foreach(i=1:length(medias))%do% {
  preds_glm[[i]] <- predict(caret_glm[[i]], testDf[[i]])
  confusionMatrix_glm[[medias[i]]]<-confusionMatrix(preds_glm[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_glm<-dotplot(resamples(list(M0 = caret_glm[[1]],M1=caret_glm[[2]],M2=caret_glm[[3]],M3=caret_glm[[4]])),metric="Accuracy")


# Glmnet2 algorithm for each media ----------------------------------------

caret_glm2<-vector("list",4)
preds_glm2<-vector("list",4)
confusionMatrix_glm2<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid")
tune.grid.glm2 <- expand.grid(alpha = c(0,1),
                             lambda=seq(0,0.5,0.01))
caret_glm2 <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],preProcess=c("center","scale"),tuneGrid=tune.grid.glm2,method = "glmnet",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_glm2


foreach(i=1:length(medias))%do% {
  preds_glm2[[i]] <- predict(caret_glm2[[i]], testDf[[i]])
  confusionMatrix_glm2[[medias[i]]]<-confusionMatrix(preds_glm2[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_glm2<-dotplot(resamples(list(M0 = caret_glm2[[1]],M1=caret_glm2[[2]],M2=caret_glm2[[3]],M3=caret_glm2[[4]])),metric="Accuracy")


# Svmlinear algorithm for each media --------------------------------------

svm.params <- getModelInfo("svmLinear")
svm.params$svmLinear$parameters


caret_svm<-vector("list",4)
preds_svm<-vector("list",4)
confusionMatrix_svm<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid")
tune.grid.svm <- expand.grid(C=c(0.005,0.1,0.15,0.2,0.3,0.4))
caret_svm <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],preProcess=c("center","scale"),tuneGrid=tune.grid.svm,method = "svmLinear",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_svm


foreach(i=1:length(medias))%do% {
  preds_svm[[i]] <- predict(caret_svm[[i]], testDf[[i]])
  confusionMatrix_svm[[medias[i]]]<-confusionMatrix(preds_svm[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_svm<-dotplot(resamples(list(M0 = caret_svm[[1]],M1=caret_svm[[2]],M2=caret_svm[[3]],M3=caret_svm[[4]])),metric="Accuracy")


# AdaBag algorithm for each media --------------------------------------

ada.params <- getModelInfo("AdaBag")
ada.params$AdaBag$parameters

caret_ada<-vector("list",4)
preds_ada<-vector("list",4)
confusionMatrix_ada<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid")
tune.grid.ada <- expand.grid(mfinal=c(5,10,15,150,200),
                              maxdepth=c(5,10,15,20))
caret_ada <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],tuneGrid=tune.grid.ada,method = "AdaBag",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_ada


foreach(i=1:length(medias))%do% {
  preds_ada[[i]] <- predict(caret_ada[[i]], testDf[[i]])
  confusionMatrix_ada[[medias[i]]]<-confusionMatrix(preds_ada[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_ada<-dotplot(resamples(list(M0 = caret_ada[[1]],M1=caret_ada[[2]],M2=caret_ada[[3]],M3=caret_ada[[4]])),metric="Accuracy")


# Gbm algorithm for each media --------------------------------------------

gbm.params <- getModelInfo("gbm")
gbm.params$gbm$parameters

caret_gbm<-vector("list",4)
preds_gbm<-vector("list",4)
confusionMatrix_gbm<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid")
caret_gbm <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],method = "gbm",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_gbm


foreach(i=1:length(medias))%do% {
  preds_gbm[[i]] <- predict(caret_gbm[[i]], testDf[[i]])
  confusionMatrix_gbm[[medias[i]]]<-confusionMatrix(preds_gbm[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_gbm<-dotplot(resamples(list(M0 = caret_gbm[[1]],M1=caret_gbm[[2]],M2=caret_gbm[[3]],M3=caret_gbm[[4]])),metric="Accuracy")

# Neural network algorithm for each media ---------------------------------

ann.params <- getModelInfo("mxnetAdam")
ann.params$mxnetAdam$parameters

caret_ann<-vector("list",4)
preds_ann<-vector("list",4)
confusionMatrix_ann<-vector("list",4)

if(useSeed)
  set.seed(nSeed)

myCluster <- makeCluster(6, type = "SOCK")
registerDoParallel(myCluster)

ptm <- Sys.time()
train.control <- trainControl(method = "cv", number = 10, search = "grid")
tune.grid.ann <- expand.grid(layer1=500,
                             layer2=400,
                             layer3=300,
                             learningrate=c(0.006,0.008,0.01,0.015,0.02,0.03,0.04),
                             beta1=0.9,
                             beta2=0.999,
                             dropout=0.5,
                             activation="relu")
caret_ann <- foreach(i=1:length(medias)) %dopar% {
  caret::train(LABEL ~ ., data = trainDf[[i]],tuneGrid=tune.grid.ann,preProcess=c("center","scale"),method = "mxnetAdam",trControl = train.control)
}
timer(ptm)

stopCluster(myCluster)

caret_ann


foreach(i=1:length(medias))%do% {
  preds_ann[[i]] <- predict(caret_ann[[i]], testDf[[i]])
  confusionMatrix_ann[[medias[i]]]<-confusionMatrix(preds_ann[[i]], testDf[[i]]$LABEL)
}

plotaccuracy_ann<-dotplot(resamples(list(M0 = caret_ann[[1]],M1=caret_ann[[2]],M2=caret_ann[[3]],M3=caret_ann[[4]])),metric="Accuracy")


# All algorithm for M0-------------------------------------------

trainConfigDf <- read_xlsx(path = "Training Configuration.xlsx",
                           sheet = "Sheet1") %>%
  dplyr::filter(use == 1)

fitlist<-vector("list")

cl <- makePSOCKcluster(nCores - 1)
registerDoParallel(cl)

foreach(i=1:length(medias))%do%{
fitList[[i]] <- foreach(iCond = 1:nrow(trainConfigDf)) %do% {
  feMethod <- trainConfigDf$method[iCond]
  fePreProc <- if(!is.na(trainConfigDf$preProc[iCond])){
    unlist(strsplit(trainConfigDf$preProc[iCond], split = "\\s"))
  }else{
    NULL
  }
  
  feTrainMetric <- trainConfigDf$trainMetric[iCond]
  
  feTrainControl <-  if(!is.na(trainConfigDf$trainControl[iCond])){
    eval(parse(text = paste0("trainControl(",
                             trainConfigDf$trainControl[iCond],
                             ")")))
  }else{
    trainControl()
  }
  
  
  feTuneGrid <- if(!is.na(trainConfigDf$tuneGrid[iCond])){
    eval(parse(text = paste0("expand.grid(",
                             trainConfigDf$tuneGrid[iCond],
                             ")")))
  }else{
    NULL
  }
  
  if(useSeed)
    set.seed(nSeed)
  
  train(as.formula(paste0(responsesOfInterest[indResp],
                          " ~",".")),
        data = trainDf[[1]],
        method = feMethod,
        preProcess = fePreProc,
        tuneGrid =  feTuneGrid,
        metric = feTrainMetric,
        trControl = feTrainControl)
  
  
  
}
names(fitList[[i]]) <- trainConfigDf$name
}
stopCluster(cl)


# summarize performance  --------------------------------------------------
reSampledFit<-vector("list",4)

foreach(i=1:length(medias))%do%{
reSampledFit[[i]] <- resamples(fitList)
summary(reSampledFit[[i]])
}
plotlist<-vector("list",4)
foreach(i=1:length(medias))%do%{
plotList[[i]] <- dotplot(reSampledFit[[i]],
                         metric="Accuracy")
}

#Save model

Model<-tribble(
  ~name, ~model,~confusionMatrix,~plotaccuracy,
  "knn",caret_knn,confusionMatrix_knn,plotaccuracy_knn,
  "rpart",caret_rp,confusionMatrix_rp,plotaccuracy_rp,
  "ranger",caret_ran,confusionMatrix_ran,plotaccuracy_ran,
  "svmlinear",caret_svm,confusionMatrix_svm,plotaccuracy_svm,
  "glmnet1",caret_glm,confusionMatrix_glm,plotaccuracy_glm,
  "glmnet2",caret_glm2,confusionMatrix_glm2,plotaccuracy_glm2,
  "rf",caret_rf,confusionMatrix_rf,plotaccuracy_rf,
  "gbm",caret_gbm,confusionMatrix_gbm,plotaccuracy_gbm,
  "ada",caret_ada,confusionMatrix_ada,plotaccuracy_ada,
  "xgboost",caret_xgb,confusionMatrix_xgb,plotaccuracy_xgb,
  "ann",caret_ann,confusionMatrix_ann,plotaccuracy_ann
  )

save(Model,
     file="analysisCaret.RData")

Model%>%
  filter(name=="Knn")%>%
  .[["model"]]
  



