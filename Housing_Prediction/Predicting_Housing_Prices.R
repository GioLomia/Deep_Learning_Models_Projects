library(keras)
library(rsample)
library(tidyverse)
library(ISLR)
library(tidyverse)
library(polycor)
library(esquisse)
library(recipes)

glimpse(housing)
raw_data<-housing

raw_data<-raw_data %>%select(median_house_value,everything())

glimpse(raw_data)


glimpse(train_test_split)
hetcor(train_test_split)

train_tbl <- training(train_test_split)
test_tbl <-testing(train_test_split)

recepie_obj <- recipe(median_house_value ~ ., data=train_tbl)%>%
  step_dummy(all_nominal(), -all_outcomes(),one_hot = TRUE)%>%
  step_bagimpute(all_predictors(),-all_outcomes())%>% 
  step_log(total_rooms)%>%
  step_center(all_predictors(),-all_outcomes())%>%
  step_scale(all_predictors(),-all_outcomes())%>%
  prep(data=train_tbl)


x_train<-bake(recepie_obj, new_data=train_tbl%>% select(-median_house_value))
x_test<-bake(recepie_obj, new_data=test_tbl%>% select(-median_house_value))

dim(x_train)
dim(x_test)

str(x_train)
str(x_test)
#For Higgs data set Homework 3 
#y_train<-ifelse(pull(train_tbl, median_house_value)=="Yes",1,0)

y_train<-train_tbl[,1]
y_test<-test_tbl[,1]

glimpse(y_train)


model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "selu", input_shape = (13)) %>%
  layer_dense(units = 256, activation = "selu", regularizer_l1_l2(0.0001)) %>%
  layer_dense(units = 100, activation = "selu") %>%
  layer_dense(units = 176, activation = "selu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_nadam(),
  loss = "mse",
  metrics = c("mae")
)

history <- model %>% fit(
  as.matrix(x_train),
  y_train,
  epochs = 12,
  batch_size = 128,
  validation_split = .2
)

metrics<-model%>%evaluate(as.matrix(x_test),y_test)
metrics
