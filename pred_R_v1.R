rm(list=ls())
#library(biglm)
#library(rfPermute)
#library(parallel)
#library(randomForestExplainer)
#library(randomForest)
#library(caret)
library(doParallel)
library(ranger)
library(pryr)
library(stringr)
library(hdlm)
library(pROC)
library(caret)
#require(h5) # available on CRAN
#library(biglasso)
#library(bigmemory)
library(vroom)
library(RJSONIO)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072*1000)


cl<-makePSOCKcluster(2) 
registerDoParallel(cl) 

options(help_type = "html") 
options(expressions = 5e5)

days  <- c(0, 12, 24)
day  <- 180

num_trees  <- c(10, 100, 500)
mtries  <- c(10, 50, 250)

rf_grid  <- expand.grid(num_tree=num_trees, mtry=mtries)

for (dstart in days) {
        cli_d <- fromJSON('hf_fp_dictionary_mod7_v3.6.json')
        #
        fname <- paste0("full_pre_ICD10/fullDataframe_", dstart ,"_", -1, "Weeks_full_code.csv.gz")
        df <- vroom(fname, delim= ",", n_max = 10^7)
        df  <- df[colSums(df) > 10]
        print(fname)
        for (i in 1:1) {
        my_uuid <- system("uuid",intern=T)
        uid  <- strsplit(my_uuid, '-')[[1]][1]

        #library(reticulate)
        #pkl <- import("pickle")
        #py <- import_builtins()
        #with(py$open("Dictionary/diag_proc_med_lab_dicts.pkl", "rb") %as% f, {
        #cli = pkl$load(f)
        #cli_r = pkl$load(f)
        #cli_d = pkl$load(f)
        #})
        #df = pd$read_hdf(paste0('Dataframes/fullDataframe_', days, 'Days_full_code_int.hd5'))
        #pd  <- import("pandas")
        #df = pd$read_hdf(paste0('Dataframes/fullDataframe_', day, 'Days_full_code_int.hd5'))
        #
        trainIndex <- createDataPartition(df$Y,p=0.7,list=FALSE)
        df_train  <- df[trainIndex, ]
        df_test   <- df[-trainIndex, ]
        rm(df)
        for (k in 1:dim(rf_grid)[1]) {
            mtry  <- rf_grid$num_tree[k]
            num_tree  <- rf_grid$mtry[k]

            rf.model<-ranger(x=df_train[, !colnames(df_train) %in% c("X", "Y")], y=factor(df_train$Y), importance='impurity_corrected', num.trees=num_tree, mtry=mtry , probability=TRUE)

            pred.iris <- predict(rf.model, data = df_test[, !colnames(df_test) %in% c("X", "Y")])

            roc.s <- roc(as.factor(df_test$Y), pred.iris$predictions[, 2])
            auc(roc.s)

            df_result  <- data.frame(y_test=as.factor(df_test$Y), predictions=pred.iris$predictions[, 2]) 

            write.csv(df_result, paste0('results/rf_predtestset_', dstart, '_', 'mtry_', mtry, 'ntree_', num_tree, '_', uid, '.csv'))

        }
   }
  }
