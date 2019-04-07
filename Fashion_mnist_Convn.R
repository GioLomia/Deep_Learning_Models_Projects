library(keras)

mnist <- dataset_fashion_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist
train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255

glimpse(train_labels)

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

glimpse(train_labels)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 10, kernel_size = c(3, 3), activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(rate=0.2)%>%
  layer_dense(units = 512, activation = "relu",regularizer_l1_l2()) %>%
  layer_dropout(rate=0.2)%>%
  layer_dense(units = 100, activation = "relu",regularizer_l1_l2())%>%
  #layer_batch_normalization()%>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
#####RUN for 4 warm restarts
train_labels
model %>% fit(
  train_images, train_labels,
  epochs = 5, batch_size=164,
  validation_split=0.2
)

metrics<-model%>%evaluate(test_images,test_labels)
metrics


