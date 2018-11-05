# Importing libraries and dataset
library(keras)
mnist <- dataset_mnist()

# Creating training and test sets for data
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshaping the data
# Reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# Rescale
x_train <- x_train / 255
x_test <- x_test / 255

# Creating categorical data.
# This is basically one hot encoding since
# we have to find out if the numbers are 0-9
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Defining the model
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# Compile the model with metrics, optimizer, and loss function
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Using TensorBoard to log results
tensorboard("logs/RMnist")

# Now we can train the model using the fit() function
# We will use 30 epochs, each batch of images is 128,
# and the validation is 20/80
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128,
  callbacks = callback_tensorboard("logs/RMnist"),
  validation_split = 0.2
)

# Plotting the comparison between training and the validation set.
plot(history)
# Looking at the performance in the test data
model %>% evaluate(x_test, y_test)
# Generating predictions 
model %>% predict_classes(x_test)
