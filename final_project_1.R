#Ashutosh 
#Nidhi 

#351773354

cat('\014')  # Clear the console
rm(list=ls()) # Clear all user objects from the environment!!!

# settingup the directory
setwd('/Users/ashutoshjha/OneDrive_SU/da_project/dataset')

#importing dataset
city_data <- read.csv(file.choose())
#Keeping only relevant attributes in the dataset 
variables_to_keep <- c("id","host_id","host_response_time","host_response_rate","host_acceptance_rate","host_total_listings_count", "neighbourhood_cleansed", "property_type","room_type","accommodates","bedrooms","beds","price","minimum_nights","maximum_nights","number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication",
                       "review_scores_location","review_scores_value","instant_bookable","has_availability", "availability_30", "availability_60", "availability_90", "availability_365" )
city_data = city_data[variables_to_keep]


# creating a function to remove NA values from all the dataset \
removeRowsWithNA <- function(data, desiredCols) {
  completeVec <- complete.cases(data[, desiredCols])
  return(data[completeVec, ])
}

# cleaning the dataset 

# Convert host_response_rate from factors to numeric values
city_data$host_response_rate <- as.numeric(gsub("%", "", as.character(city_data$host_response_rate)))/100
# Convert host_acceptance_rate from factors to numeric values
city_data$host_acceptance_rate <- as.numeric(gsub("%", "", as.character(city_data$host_acceptance_rate)))/100
# Convert price from factors to numeric values
city_data$price <- as.double(substring(gsub(",", "", as.character(city_data$price)),2))
city_data <- removeRowsWithNA(city_data)

#converting relevant columns to factors 
city_data$host_response_time <- as.factor(city_data$host_response_time)
city_data$instant_bookable <- as.factor(city_data$instant_bookable)
city_data$room_type <- as.factor(city_data$room_type)
city_data$property_type <- as.factor(city_data$property_type)


##split price into 3 bins: 
temp <- sort.int(city_data$price, decreasing = FALSE)
level_1 <- temp[round(length(temp)/3, digits = 0)]
level_2 <- temp[2*round(length(temp)/3, digits = 0)]
city_data$price_level[city_data$price <= level_1] <- "Low"
city_data$price_level[city_data$price > level_1 & city_data$price <= level_2] <- "Medium"
city_data$price_level[city_data$price > level_2] <- "High"
#converting the newly created column to factors 
city_data$price_level <- as.factor(city_data$price_level)

# data cleaning over 

#Feature selection from the dataset 

feature_set <- city_data[ c("neighbourhood_cleansed", "room_type", "host_response_time", "bedrooms", "host_response_rate",
                            "availability_365", "minimum_nights", "number_of_reviews", "instant_bookable", "price_level") ]

#feature set extracted 


#Divinding the dataset into test and train with 90 and 10 split
indexTrain <- createDataPartition(y = feature_set$price_level ,p=0.80,list = FALSE)
nb_training <- feature_set[indexTrain,]
nb_testing <- feature_set[-indexTrain,]

training_independent_vars = nb_training[ c("neighbourhood_cleansed", "room_type", "host_response_time", "bedrooms", "host_response_rate",
                                           "availability_365", "minimum_nights", "number_of_reviews", "instant_bookable") ]
training_price = nb_training$price_level


#Dataset Divide complete 

# Running Models

# ...... Start Decision Tree .....

#model
decision_rpart <- rpart(nb_training$price_level ~  .,
                        data = training_independent_vars,
                        method = 'class', 
                        control = rpart.control(maxdepth = 5))
rpart.plot(decision_rpart)

#model using caret function
controls <- trainControl(method = "repeatedcv",
                         number = 5,
                         repeats = 5)

decision_model <-  train(training_independent_vars,
                         training_price,
                         method = 'rpart',
                         trControl = controls)
#predict 
predict <- predict(decision_model, newdata = nb_testing)
# confusion matrix 
confusionMatrix(predict, nb_testing$price_level)
#precision recall f- measure 
confusionMatrix(predict, nb_testing$price_level , mode = "prec_recall", positive="1")
# roc curve
# **** we have tried to plot roc curve but that is only valid for binary classifier not
# **** multidimensional classifier

#..... decision tree... ended 

#... naive bayes start ... 
Contorls <- trainControl(method = 'repeatedcv',
                         number = 10)
naive_model <- train(training_independent_vars,
                     training_price,
                     method = 'nb',
                     trControl = Contorls)

predict <- predict(naive_model, newdata = nb_testing)
confusionMatrix(predict, nb_testing$price_level)
confusionMatrix(predict, nb_testing$price_level , mode = "prec_recall", positive="1")

# ..... naive bayes ended....


# ensemble method 

# .....range method start ...

Contorls <- trainControl(method = 'cv',
                         number = 5)
range_model<- train(training_independent_vars,
                   training_price,
                   method = 'ranger',
                   trControl = Contorls)


predict<-  predict(range_model, newdata = nb_testing)
confusionMatrix(predict, nb_testing$price_level)
confusionMatrix(predict, nb_testing$price_level , mode = "prec_recall", positive="1")

#.... tree method end 


# svm start

svm_data = city_data[ c("neighbourhood_cleansed", "room_type", "host_response_time", "bedrooms", "host_response_rate",
                        "availability_365", "minimum_nights", "number_of_reviews", "instant_bookable", "price_level") ]


# Creating Dummy columns of feature for SVM

svm_data <- dummy_cols(svm_data, select_columns = c('neighbourhood_cleansed',
                                                    'room_type',
                                                    'host_response_time',
                                                    'instant_bookable'))



# removing columns whose svm dummy variable is created 

svm_data=select(svm_data,-c(neighbourhood_cleansed,room_type,host_response_time,instant_bookable))

# svm _algo start 
set.seed(120)
inTraining <- createDataPartition(svm_data$price_level, p = .90, list = FALSE)

training_set <- svm_data[inTraining,] %>% select(-price_level)
training_labels <- svm_data[inTraining,]$price_level

test_labels <- svm_data[-inTraining,]$price_level # Extracting test labels
test_set_final  <- svm_data[-inTraining,]%>% select(-price_level) 

# Fitting SVM to the Training set 
Controls <- trainControl(method='cv',number=10)
Grid_lin <- expand.grid(C = seq(0, 2, length = 11))

linear_SVM_ads <- train(training_set,
                        training_labels,
                        method = 'svmLinear',
                        trControl= Controls,
                        tuneGrid = Grid_lin)

linear_SVM_ads

#prediction
predict<-  predict(linear_SVM_ads, newdata = test_set_final)
#confusion matrix 
conf_matrix_lin <- confusionMatrix(test_labels, predict)
#Precision, recall, F-Measure 
confusionMatrix(predict, test_labels , mode = "prec_recall", positive="1")











