#!/usr/bin/env python
# coding: utf-8

# ## Linear regression using PyTorch built-ins
# 
# We've implemented linear regression & gradient descent model using some basic tensor operations. However, since this is a common pattern in deep learning, PyTorch provides several built-in functions and classes to make it easy to create and train models with just a few lines of code.
# 
# Let's begin by importing the `torch.nn` package from PyTorch, which contains utility classes for building neural networks.

# We are using 15 training examples to illustrate how to work with large datasets in small batches. 

# ## Dataset and DataLoader
# 
# We'll create a `TensorDataset`, which allows access to rows from `inputs` and `targets` as tuples, and provides standard APIs for working with many different types of datasets in PyTorch.

# The `TensorDataset` allows us to access a small section of the training data using the array indexing notation (`[0:3]` in the above code). It returns a tuple with two elements. The first element contains the input variables for the selected rows, and the second contains the targets.

# We'll also create a `DataLoader`, which can split the data into batches of a predefined size while training. It also provides other utilities like shuffling and random sampling of the data.

# We can use the data loader in a `for` loop. Let's look at an example.

# In each iteration, the data loader returns one batch of data with the given batch size. If `shuffle` is set to `True`, it shuffles the training data before creating batches. Shuffling helps randomize the input to the optimization algorithm, leading to a faster reduction in the loss.

# ## nn.Linear
# 
# Instead of initializing the weights & biases manually, we can define the model using the `nn.Linear` class from PyTorch, which does it automatically.

# PyTorch models also have a helpful `.parameters` method, which returns a list containing all the weights and bias matrices present in the model. For our linear regression model, we have one weight matrix and one bias matrix.

# ## Loss Function
# 
# Instead of defining a loss function manually, we can use the built-in loss function `mse_loss`.

# The `nn.functional` package contains many useful loss functions and several other utilities. 

# ## Optimizer
# 
# Instead of manually manipulating the model's weights & biases using gradients, we can use the optimizer `optim.SGD`. SGD is short for "stochastic gradient descent". The term _stochastic_ indicates that samples are selected in random batches instead of as a single group.

# Note that `model.parameters()` is passed as an argument to `optim.SGD` so that the optimizer knows which matrices should be modified during the update step. Also, we can specify a learning rate that controls the amount by which the parameters are modified.

# ## Train the model
# 
# We are now ready to train the model. We'll follow the same process to implement gradient descent:
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
# The only change is that we'll work batches of data instead of processing the entire training data in every iteration. Let's define a utility function `fit` that trains the model for a given number of epochs.

# Some things to note above:
# 
# * We use the data loader defined earlier to get batches of data for every iteration.
# 
# * Instead of updating parameters (weights and biases) manually, we use `opt.step` to perform the update and `opt.zero_grad` to reset the gradients to zero.
# 
# * We've also added a log statement that prints the loss from the last batch of data for every 10th epoch to track training progress. `loss.item` returns the actual value stored in the loss tensor.
# 
# Let's train the model for 100 epochs.

# The predicted yield of apples is 54.3 tons per hectare, and that of oranges is 68.3 tons per hectare.
