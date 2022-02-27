#!/usr/bin/env Rscript

rm(list=ls())
library(doParallel)
library(ranger)
library(pryr)
library(stringr)
library(hdlm)
library(RJSONIO)


options(help_type = "html") 
options(expressions = 5e5)

dstart <- 24
dend <- -1


library(reticulate)
library(vroom)

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072*1000)


#fname <- paste0("full_pre_ICD10/fullDataframe_", dstart ,"_", dend, "Weeks_full_code.csv.gz")
fname <- paste0("test_data.csv.gz")
print(fname)
df <- vroom(fname, delim= ",", n_max = 10^7)

rf.model<-ranger(x=df[, !colnames(df) %in% c("X", "Y")], y=factor(df$Y), data=df, importance='impurity_corrected')

cli_d <- fromJSON('hf_fp_dictionary_mod7_v3.6.json')

rf.ip  <- importance_pvalues(rf.model, method = "janitza")
nval  <- sum(rf.ip[, "pvalue"] == 0)
pval_rnd  <- runif(nval, min = 1e-9, max = 1e-7)
rf.ip[, "pvalue"][rf.ip[, "pvalue"] == 0]  <- pval_rnd
rf.ip  <-  cbind(rf.ip, p.adjust(rf.ip[, "pvalue"] , "bonferroni"))
colnames(rf.ip)[3]  <- "pvalue_adj"
rf.ip  <- rf.ip[rf.ip[, "pvalue_adj"] < 2, ]
rf.ip  <- rf.ip[order(rf.ip[, "importance"], decreasing=T), ]

names(cli_d)  <- str_remove_all(names(cli_d), '00000$')

rnames  <- rownames(rf.ip)
lidx  <- str_starts(rnames, 'L')
rownames(rf.ip)[lidx]  <- str_replace_all(rnames[lidx], '\\.', '-')
v  <- c()
for (c in rownames(rf.ip)) {
if (!is.null(cli_d[[c]])) {
v  <- c(v, cli_d[[c]])
}
else {
v  <- c(v, 'NULL')
}

#print(c)
#print(cli_d[[c]])
}


rf.ipd  <- as.data.frame(rf.ip)
rf.ipd$description  <- v

#rf.ipd[rf.ipd$importance >=2, ]


dim(rf.ipd)

fresult <- 'RF_importance_test.csv'
write.csv(rf.ipd, fresult)
