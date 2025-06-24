library(lhs)
library(wordspace)
library(foreach)
library(umap)
library(dplyr)
library(ggplot2)
library(stats)
library(optparse)
library(gplite)

set.seed(430)
# Command Line Arguments
option_list <- list(
    make_option(c("--task_id"), type = "numeric", default = 0,
                help = "Task id [default: %default]"),
    make_option(c("--new_digit"), type = "numeric", default = 5,
                help = "New digit to add [default: %default]"),
    make_option(c("--models_dir"), type= "character", default = "AE500_Indpoints_withGP_f64(generate100pc-GP-use-Nonfiltered)/task0-4/RData/",
                help = "Directory containing the GP models [default: %default]"),   
)

parser <- OptionParser(option_list = option_list)
args <- parse_args(parser)

# 
GP1 <- gp_load("AE500_Indpoints_withGP_f64(generate100pc-GP-use-Nonfiltered)/task0-4/RData/indpts_Rdata_ntr1500_nts3000_noctr50_f64/GPmodel_c0.rda")

print(GP1)