library(keras)
library(recipes)
library(rsample)
library(DataExplorer)
library(tidyverse)

data(credit_data)
help("credit_data")
# STEP #1: Take a look at the credit data, find # of samples and predictors, and all data types
dim(credit_data)
glimpse(credit_data)

# STEP #2: Check for missing values; determine whether to omit missing rows/columns or to impute missing values
plot_missing(credit_data)

# STEP #3: Use rsample package to create training/test split and enter last 4 digits of B# as seed value
set.seed(854)

train_test_split <- initial_split(credit_data, prop = .8)

train_data <- training(train_test_split)
test_data <- testing(train_test_split)

# STEP #4: Check the dimensions of the train and test datasets here
str(train_data)
dim(test_data)


# STEP #5: Create recipe for data preprocessing [FIX THESE AND PUT IN THE RIGHT ORDER]
rec_obj <- recipe(Status ~ ., data = train_data) %>%
  step_bagimpute(all_predictors(),-all_outcomes())%>% 
  step_center(all_numeric(),-all_outcomes())%>%
  step_scale(all_numeric(),-all_outcomes())%>%
  step_dummy(all_nominal(), -all_outcomes(),one_hot = TRUE)%>%
  prep(data = train_data)

rec_obj

train_clean <- bake(rec_obj, new_data = train_data) %>% select(-Status)
test_clean  <- bake(rec_obj, new_data = test_data) %>% select(-Status)
plot_missing(test_clean)
# remove any remaining missing rows
train_clean<-na.omit(train_clean)
test_clean<-na.omit(test_clean)


# STEP #6: Take another glimpse at the train_clean and test_clean
glimpse(train_clean)
glimpse(test_clean)
# STEP #7: Create separate vectorS for response variables for training and testing sets [FIX THIS]
train_y <- ifelse(pull(train_data, Status) == "good", 1, 0)  # one-hot encode train Y
test_y  <- ifelse(pull(test_data, Status) == "good", 1, 0)   # one-hot encode test Y
train_y

# STEP 8 - DELETED


# STEP #9: Building our Artificial Neural Network [FIX THIS]

model <- keras_model_sequential() %>%
  layer_dense(units = 520, activation = "selu",regularizer_l1_l2(),input_shape = (ncol(train_clean))) %>%
  #layer_dropout(rate=.5)%>%
  layer_dense(units = 360,regularizer_l1_l2(l1=0.5),activation = "selu") %>%
  layer_dropout(rate=0.4)%>%
  layer_dense(units = 200,regularizer_l1_l2(l1=0.2), activation = "selu") %>%
  layer_dropout(rate=0.5)%>%
  layer_dense(units = 45, activation = "selu") %>%
  layer_dense(units = 100,regularizer_l1_l2(l1=0.02), activation = "selu") %>%
  layer_batch_normalization()%>%
  layer_dropout(rate=0.15)%>%
  layer_dense(units = 50,regularizer_l1_l2(l1=0.02), activation = "selu") %>%
  layer_dropout(rate=0.3)%>%
  layer_dense(units = 20,regularizer_l1_l2(l1=0.02), activation = "selu") %>%
  layer_dropout(rate=0.1)%>%
  layer_batch_normalization()%>%
  #ADD DENSE LAYER WITH ACTIVATION AND WEIGHT INITIALIZATION
  layer_dense(units = 1, activation = "sigmoid")

model

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# STEP 10: Fitting the model on the training data [FIX THIS]
model %>% fit(as.matrix.data.frame(train_clean),as.numeric(train_y), validation_split = .2, 
              epochs = 20, batch_size = 70)


# STEP 11: After fine tuning your model, complete ONE TEST RUN ONLY
results <- model %>% evaluate(as.matrix.data.frame(test_clean),test_y)
results


# STEP #12: This step creates the predicted class for every test sample so that we can make a confusion matrix 
pred <- model %>% predict_classes(as.matrix(test_clean), batch_size = 70)

# Confusion matrix
CM = table(pred, test_y)
CM