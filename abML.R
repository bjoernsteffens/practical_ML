setwd("~/mycloud/Private_DataScience/Coursera/10 Data Science Specialisation/70 Practical Machine Learning/Assignments")
rm(list=ls())

require(dplyr)
require(caret)
require(randomForest)
require(rpart)
require(rpart.plot)

#
# Acquire and load the files
tstFile  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trnFile <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if ( !file.exists("tst.csv") && !file.exists("trn.csv") ) {
    
    download.file(tstFile, "tst.csv")
    download.file(trnFile, "trn.csv")
    
}

tst = read.csv("tst.csv",na.strings=c("NA","#DIV/0!"))
trn = read.csv("trn.csv",na.strings=c("NA","#DIV/0!"))

# ** CLEAN THE DATA
#
# Remove columns that may not influence
# the prediction
dropCols <- c("X","user_name",
              "raw_timestamp_part_1",
              "raw_timestamp_part_2",
              "cvtd_timestamp",
              "num_window", "new_window")
tst[,dropCols] <- NULL
trn[,dropCols] <- NULL

#
# There is a ton of cols with NAs. Remove them too
tst  <- tst[,colSums(is.na(tst)) == 0]
trn <- trn[,colSums(is.na(trn)) == 0]

# ** START PREPARING THE DATA
# Split up the training set to enable
# cross validation
trnArray  <- createDataPartition(y=trn$classe, p=0.99885, list=FALSE)
trnLarge  <- trn[ trnArray,]
trnSubset <- trn[-trnArray,]


#
# How does this look like?
plot(trnSubset$classe, col="blue", main="Bar Plot of levels of the variable classe within the subTraining data set", xlab="classe levels", ylab="Frequency")

# ** CREATE MODELS AND COMPARE
# Train the model, create the prediction 
# and then verify how acurate it is
mod1  <- rpart(classe ~ ., data=trnLarge, method="class")
pred1 <- predict(mod1, trnSubset, type = "class")
rpart.plot(mod1, main="Tree View", branch = 1, fallen.leaves = TRUE, uniform = TRUE)
confusionMatrix(pred1, trnSubset$classe)

#
# Train the other model, create the prediction 
# and then verify how acurate it is
mod2  <- randomForest(classe ~. , data=trnLarge, method="class")
pred2 <- predict(mod2, trnSubset, type = "class")
confusionMatrix(pred2, trnSubset$classe)

#
# Random Forests seems to be more acurate vs deciscion tree.
# Apply the better model to the test set
predFin1 <- predict(mod1, test, type = "class")
predFin2 <- predict(mod2, test, type = "class")

# Less Accurate Model
print(predFin1)

# More Accurate Model
print(predFin2)