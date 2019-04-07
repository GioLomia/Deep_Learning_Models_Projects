library(keras)
library(rsample)
library(tidyverse)
library(ISLR)
library(tidyverse)
library(polycor)
library(esquisse)
library(recipes)

raw_data<-read.csv("D:\\deep learning\\BUS_CSC 386 Homework Data\\supersym.csv")

glimpse(raw_data)
dim(raw_data)

plot_missing(raw_data)


recepie_obj <- recipe(target ~ ., data=raw_data)%>%
  step_center(all_predictors(),-all_outcomes())%>%
  step_scale(all_predictors(),-all_outcomes())%>%
  prep(data=train_tbl)


x_train<-bake(recepie_obj, new_data=raw_data)

#################DATA##SPLIT##################
train_test_split<-initial_split(x_train,0.5)

first_part <- training(train_test_split)
second_part <-testing(train_test_split)


train_test_split<-initial_split(first_part,0.5)

first_quater <- training(train_test_split)
first_y_quater<-first_quater[,1]
second_quater <- testing(train_test_split)
second_y_quater<-second_quater[,1]


train_test_split<-initial_split(second_part,0.5)

third_quater <- training(train_test_split)
third_y_quater<-third_quater[,1]
forth_quater <- testing(train_test_split)
forth_y_quater<-forth_quater[,1]

first_quater<-first_quater%>%select(-target)
second_quater<-second_quater%>%select(-target)
third_quater<-third_quater%>%select(-target)
forth_quater<-forth_quater%>%select(-target)

remove(first_part)
remove(second_part)

#############################################
typeof(first_quater)
#############MODEL 1#########################
model1 <- keras_model_sequential() %>%
  layer_dense(units = 250, activation = "selu", input_shape = (28)) %>%
  layer_dense(units = 250, activation = "selu") %>%
  layer_dense(units = 250, activation = "selu") %>%
  layer_dense(units = 250, activation = "selu") %>%
  layer_dense(units = 250, activation = "selu") %>%
  layer_dense(units = 1,activation="sigmoid")

model1 %>% compile(
  optimizer = optimizer_sgd(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model1 %>% fit(
  as.matrix.data.frame(first_quater),
  as.matrix.data.frame(first_y_quater),
  epochs = 5,
  batch_size = 200,
  validation_split = .2
)
###########################################

#############MODEL 2#########################
model2 <- keras_model_sequential() %>%
  layer_dense(units = 12, activation = "relu", input_shape = (28)) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1,activation="sigmoid")

model2 %>% compile(
  optimizer = optimizer_nadam(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model2 %>% fit(
  as.matrix.data.frame(second_quater),
  as.matrix.data.frame(second_y_quater),
  epochs = 5,
  batch_size = 200,
  validation_split = .2
)
###########################################

#############MODEL 3#########################
model3 <- keras_model_sequential() %>%
  layer_dense(units = 520, activation = "relu", input_shape = (28)) %>%
  layer_dense(units = 300, activation = "relu") %>%
  layer_dense(units = 1,activation="sigmoid")

model3 %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model3 %>% fit(
  as.matrix.data.frame(third_quater),
  as.matrix.data.frame(third_y_quater),
  epochs = 5,
  batch_size = 200,
  validation_split = .2
)
###########################################

#############MODEL 4#########################
model4 <- keras_model_sequential() %>%
  layer_dense(units = 250, activation = "selu", input_shape = (28)) %>%
  initializer_he_normal()%>%
  layer_dense(units = 500, activation = "selu")%>%
  initializer_he_normal()%>%
  layer_dense(units = 1,activation=activation_hard_sigmoid())

model4 %>% compile(
  optimizer = optimizer_nadam(lr=0.00000001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model4 %>% fit(
  as.matrix.data.frame(forth_quater),
  as.matrix.data.frame(forth_y_quater),
  epochs = 5,
  batch_size = 200,
  validation_split = .2
)
###########################################










