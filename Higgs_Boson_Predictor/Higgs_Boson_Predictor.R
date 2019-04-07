library(keras)
library(rsample)
library(tidyverse)
library(ISLR)
library(tidyverse)
library(polycor)
library(esquisse)
library(recipes)

raw_data<-Higgs

glimpse(raw_data)
dim(raw_data)
str(raw_data)

plot_missing(raw_data)

new_data<-raw_data%>%select(-EventId)
new_data<-new_data%>%select(Label,everything())

glimpse (raw_data)
set.seed(1234)
train_test_split<-initial_split(new_data,0.8)

train_tbl <- training(train_test_split)
test_tbl<-testing(train_test_split)


recepie_obj <- recipe(Label ~ ., data=train_tbl)%>%
  step_center(all_predictors(),-all_outcomes())%>%
  step_scale(all_predictors(),-all_outcomes())%>%
  prep(data=train_tbl)


x_train<-bake(recepie_obj, new_data=train_tbl)
x_test<-bake(recepie_obj, new_data=test_tbl)

y_train_vec<- 
y_test_vec<-ifelse(pull(x_test, Label) == "s",1,0)

x_train<-x_train%>%select(-Label)
x_test<-x_test%>%select(-Label)


######################MODEL 1#########################
mode <- keras_model_sequential() %>%
  layer_dense(units = 1200, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_batch_normalization()%>%
  layer_dense(units = 1,activation="sigmoid")
######################################################
mode %>% compile(
  optimizer = optimizer_nadam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- mode %>% fit(
  as.matrix.data.frame(x_train),
  as.vector(y_train_vec),
  epochs = 5,
  batch_size = 200,
  validation_split = .2
)

######################MODEL 2#########################
mode <- keras_model_sequential() %>%
  layer_dense(units = 120, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 60, activation = "relu") %>%
  layer_batch_normalization()%>%
  layer_dense(units = 1,activation="sigmoid")
######################################################
mode %>% compile(
  optimizer = optimizer_nadam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- mode %>% fit(
  as.matrix.data.frame(x_train),
  as.vector(y_train_vec),
  epochs = 5,
  batch_size = 200,
  validation_split = .2
)

######################MODEL 3#########################
mode <- keras_model_sequential() %>%
  layer_dense(units = 500, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 250, activation = "relu",regularizer_l1_l2(l1=0.001,l2=0.002)) %>%
  layer_dense(units = 125, activation = "relu") %>%
  layer_batch_normalization()%>%
  layer_dense(units = 1,activation="sigmoid")
######################################################
mode %>% compile(
  optimizer = optimizer_adam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- mode %>% fit(
  as.matrix.data.frame(x_train),
  as.vector(y_train_vec),
  epochs = 5,
  batch_size = 200,
  validation_split = .2
)


######################MODEL 4#########################
mode <- keras_model_sequential() %>%
  layer_dense(units = 800, activation = "selu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 150, activation = "selu") %>%
  layer_dense(units = 125, activation = "selu") %>%
  layer_dense(units = 8, activation = "selu") %>%
  layer_dense(units = 1,activation="sigmoid")
######################################################
mode %>% compile(
  optimizer = optimizer_sgd(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- mode %>% fit(
  as.matrix.data.frame(x_train),
  as.vector(y_train_vec),
  epochs = 5,
  batch_size = 128,
  validation_split = .2
)



metrics<-mode%>%evaluate(as.matrix.data.frame(x_test),y_test_vec)
metrics

