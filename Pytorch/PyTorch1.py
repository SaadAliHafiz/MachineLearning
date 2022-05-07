#!/usr/bin/env python
# coding: utf-8

# #### Create Tensor

# #### Tensor Operations and gradients 

# # Introduction to Linear Regression

# ### Linear regression model from scratch

# ## Loss function
# 
# Before we improve our model, we need a way to evaluate how well our model is performing. We can compare the model's predictions with the actual targets using the following method:
# 
# * Calculate the difference between the two matrices (`preds` and `targets`).
# * Square all elements of the difference matrix to remove negative values.
# * Calculate the average of the elements in the resulting matrix.
# 
# The result is a single number, known as the **mean squared error** (MSE).

# ## Train the model using gradient descent
# 
# As seen above, we reduce the loss and improve our model using the gradient descent optimization algorithm. Thus, we can _train_ the model using the following steps:
# 
# 1. Generate predictions
# 
# 2. Calculate the loss
# 
# 3. Compute gradients w.r.t the weights and biases
# 
# 4. Adjust the weights by subtracting a small quantity proportional to the gradient
# 
# 5. Reset the gradients to zero
# 
# Let's implement the above step by step.

# ## Train for multiple epochs
# 
# To reduce the loss further, we can repeat the process of adjusting the weights and biases using the gradients multiple times. Each iteration is called an _epoch_. Let's train the model for 100 epochs.
