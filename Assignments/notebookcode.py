#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Neural-Network-Classifier" data-toc-modified-id="Neural-Network-Classifier-1">Neural Network Classifier</a></span></li><li><span><a href="#Apply-NeuralNetworkClassifier-to-Handwritten-Digits" data-toc-modified-id="Apply-NeuralNetworkClassifier-to-Handwritten-Digits-2">Apply <code>NeuralNetworkClassifier</code> to Handwritten Digits</a></span></li><li><span><a href="#Experiments" data-toc-modified-id="Experiments-3">Experiments</a></span></li><li><span><a href="#Grading" data-toc-modified-id="Grading-4">Grading</a></span></li><li><span><a href="#Extra-Credit" data-toc-modified-id="Extra-Credit-5">Extra Credit</a></span></li></ul></div>

# # Neural Network Classifier
# 
# You may start with your `NeuralNetwork` class from A2, or start with the [implementation defined here](https://www.cs.colostate.edu/~anderson/cs545/notebooks/A2solution.tar) in which all functions meant be called by other functions in this class start with an underscore character. Implement the subclass `NeuralNetworkClassifier` that extends `NeuralNetwork` as discussed in class.  Your `NeuralNetworkClassifier` implementation should rely on inheriting functions from `NeuralNetwork` as much as possible. 
# 
# Your `neuralnetworks.py` file (notice it is plural) will now contain two classes, `NeuralNetwork` and `NeuralNetworkClassifier`.
# 
# In `NeuralNetworkClassifier` replace the `error_f` function with one called `_neg_log_likelihood_f` and pass it instead of `error_f` into the optimization functions.

# Here are some example tests.

# In[41]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[42]:


import numpy as np
import neuralnetworks as nn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
T = np.array([[0], [1], [1], [0]])
X, T


# In[44]:


np.random.seed(111)
nnet = nn.NeuralNetworkClassifier(2, [10], 2)


# In[45]:


print(nnet)


# In[46]:


nnet.Ws


# The `softmax` function can produce errors if the denominator is close to zero.  Here is an implentation you may use to avoid some of those errors.  This assumes you have the following import in your `neuralnetworks.py` file.
# 
# `sys.float_info.epsilon` is also useful in your `_neg_log_likehood_f` function to avoid taking the `log` of zero.

# In[47]:


import sys  # for sys.float_info.epsilon 


# In[48]:


def _softmax(self, Y):
    '''Apply to final layer weighted sum outputs'''
    # Trick to avoid overflow
    maxY = Y.max()       
    expY = np.exp(Y - maxY)
    denom = expY.sum(1).reshape((-1, 1))
    Y = expY / (denom + sys.float_info.epsilon)
    return Y


# Replace the `error_f` function with `neg_log_likelihood`.  If you add some print statements in `_neg_log_likelihood` functions, you can compare your output to the following results.

# In[49]:


nnet.train(X, T, n_epochs=1, method='sgd', learning_rate=0.01)


# In[50]:


print(nnet)


# Now if you comment out those print statements, you can run for more epochs without tons of output.

# In[51]:


np.random.seed(111)
nnet = nn.NeuralNetworkClassifier(2, [10], 2)


# In[52]:


nnet.train(X, T, 100, method='scg')


# In[53]:


nnet.use(X)


# In[54]:


def percent_correct(Y, T):
    return np.mean(T == Y) * 100


# In[55]:


print(percent_correct(nnet.use(X)[0], T))
print(nnet.use(X)[0])
print(T)


# Works!  The XOR problem was used early in the history of neural networks as a problem that cannot be solved with a linear model.  Let's try it.  It turns out our neural network code can do this if we use an empty list for the hidden unit structure!

# In[56]:


nnet = nn.NeuralNetworkClassifier(2, [], 2)
nnet.train(X, T, 100, method='scg')


# In[57]:


nnet.use(X)


# In[58]:


percent_correct(nnet.use(X)[0], T)


# A second way to evaluate a classifier is to calculate a confusion matrix. This shows the percent accuracy for each class, and also shows which classes are predicted in error.
# 
# Here is a function you can use to show a confusion matrix.

# In[59]:


import pandas

def confusion_matrix(Y_classes, T):
    class_names = np.unique(T)
    table = []
    for true_class in class_names:
        row = []
        for Y_class in class_names:
            row.append(100 * np.mean(Y_classes[T == true_class] == Y_class))
        table.append(row)
    conf_matrix = pandas.DataFrame(table, index=class_names, columns=class_names)
    # cf.style.background_gradient(cmap='Blues').format("{:.1f} %")
    print('Percent Correct')
    return conf_matrix.style.background_gradient(cmap='Blues').format("{:.1f}")


# In[60]:


confusion_matrix(nnet.use(X)[0], T)


# # Apply `NeuralNetworkClassifier` to Handwritten Digits

# Apply your `NeuralNetworkClassifier` to the [MNIST digits dataset](http://deeplearning.net/tutorial/gettingstarted.html).

# In[61]:


import pickle
import gzip


# In[62]:


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

Xtrain = train_set[0]
Ttrain = train_set[1].reshape(-1, 1)

Xval = valid_set[0]
Tval = valid_set[1].reshape(-1, 1)

Xtest = test_set[0]
Ttest = test_set[1].reshape(-1, 1)

print(Xtrain.shape, Ttrain.shape,  Xval.shape, Tval.shape,  Xtest.shape, Ttest.shape)


# In[63]:


28*28


# In[64]:


print(Ttrain[0:10])
print(np.unique(Ttrain))


# In[65]:


def draw_image(image, label):
    plt.imshow(-image.reshape(28, 28), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(label)


# In[66]:


plt.figure(figsize=(7, 7))
for i in range(100):
    plt.subplot(10, 10, i+1)
    draw_image(Xtrain[i], Ttrain[i,0])
plt.tight_layout()


# In[67]:


nnet = nn.NeuralNetworkClassifier(784, [], 10)
nnet.train(Xtrain, Ttrain, n_epochs=40, method='scg')


# In[68]:


print(nnet)


# In[69]:


[percent_correct(nnet.use(X)[0], T) for X, T in zip([Xtrain, Xval, Xtest], [Ttrain, Tval, Ttest])]


# In[70]:


nnet = nn.NeuralNetworkClassifier(784, [20], 10)
nnet.train(Xtrain, Ttrain, n_epochs=40, method='scg')


# In[71]:


[percent_correct(nnet.use(X)[0], T) for X, T in zip([Xtrain, Xval, Xtest],
                                                    [Ttrain, Tval, Ttest])]


# # Experiments
# 
# For each method, try various hidden layer structures, learning rates, and numbers of epochs.  Use the validation percent accuracy to pick the best hidden layers, learning rates and numbers of epochs for each method (ignore learning rates for scg).  Report training, validation and test accuracy for your best validation results for each of the three methods.
# 
# Include plots of data likelihood versus epochs, and confusion matrices, for best results for each method.
# 
# Write at least 10 sentences about what you observe in the likelihood plots, the train, validation and test accuracies, and the confusion matrices.

# In[72]:


def print_nn(neural_net, method, rho, final, XTrain, TTrain, XVal, TVal, XTest, TTest):
    if method == 'scg':
            print( 'method = ', method, '\nnetwork structure = ', str(neural_net), '\n')
    else:
        print( 'method = ', method, '\nnetwork structure = ', str(neural_net), '\nlearning rate = ', rho, '\n')
    print(['Train', 'Validate'])
    print([percent_correct(neural_net.use(_X)[0], _T) for _X, _T in zip([XTrain, XVal],
                                                    [TTrain, TVal])], '\n')
    if final:
        print(['Test'])
        print([percent_correct(neural_net.use(_X)[0], _T) for _X, _T in zip([XTest],
                                                    [TTest])], '\n')
        display(confusion_matrix(neural_net.use(XTest)[0], TTest))
        plt.figure()
        plt.plot(neural_net.get_error_trace())
        plt.xlabel('Epoch')
        plt.ylabel('Data Likelihood');


# In[73]:


n_epochs = 100
rho = 0.01
structure = [20]


nnet_0_sgd = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_0_sgd.train(Xtrain, Ttrain, n_epochs, method='sgd', learning_rate=rho, verbose=False)
print_nn(nnet_0_sgd, 'sgd', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
nnet_0_adam = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_0_adam.train(Xtrain, Ttrain, n_epochs, method='adam', learning_rate=rho, verbose=False)     
print_nn(nnet_0_adam, 'adam', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)

nnet_0_scg = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_0_scg.train(Xtrain, Ttrain, n_epochs, method='scg', learning_rate=rho, verbose=False)
print_nn(nnet_0_scg, 'scg', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# In[74]:


n_epochs = 500
rho = 0.001
structure = [20, 20, 10]
    
nnet_1_sgd = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_1_sgd.train(Xtrain, Ttrain, n_epochs, method='sgd', learning_rate=rho, verbose=False)
print_nn(nnet_1_sgd, 'sgd', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
nnet_1_adam = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_1_adam.train(Xtrain, Ttrain, n_epochs, method='adam', learning_rate=rho, verbose=False)     
print_nn(nnet_1_adam, 'adam', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)

nnet_1_scg = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_1_scg.train(Xtrain, Ttrain, n_epochs, method='scg', learning_rate=rho, verbose=False)
print_nn(nnet_1_scg, 'scg', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# In[75]:


n_epochs = 50
rho = 0.05
structure = []
    
nnet_2_sgd = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_2_sgd.train(Xtrain, Ttrain, n_epochs, method='sgd', learning_rate=rho, verbose=False)
print_nn(nnet_2_sgd, 'sgd', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
nnet_2_adam = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_2_adam.train(Xtrain, Ttrain, n_epochs, method='adam', learning_rate=rho, verbose=False)     
print_nn(nnet_2_adam, 'adam', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)

nnet_2_scg = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_2_scg.train(Xtrain, Ttrain, n_epochs, method='scg', learning_rate=rho, verbose=False)
print_nn(nnet_2_scg, 'scg', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# In[76]:


n_epochs = 750
rho = 0.025
structure = [20, 15, 10, 5 ]
    
nnet_3_sgd = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_3_sgd.train(Xtrain, Ttrain, n_epochs, method='sgd', learning_rate=rho, verbose=False)
print_nn(nnet_3_sgd, 'sgd', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
nnet_3_adam = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_3_adam.train(Xtrain, Ttrain, n_epochs, method='adam', learning_rate=rho, verbose=False)     
print_nn(nnet_3_adam, 'adam', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)

nnet_3_scg = nn.NeuralNetworkClassifier(784, structure, 10)
nnet_3_scg.train(Xtrain, Ttrain, n_epochs, method='scg', learning_rate=rho, verbose=False)
print_nn(nnet_3_scg, 'scg', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# In[77]:


print('Best Cases')
print_nn(nnet_2_sgd, 'sgd', rho, True, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)     
print_nn(nnet_0_adam, 'adam', rho, True, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
print_nn(nnet_0_scg, 'scg', rho, True, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# # Discussion:
# The training, validation and test accuracy for the setups with the best validation results for each of the three methods are listed above. I also included plots of data likelihood versus epochs, and confusion matrices, for the best results for each method.</br>
# I constructed my experiments to test each method (sgd, adam, scg) using the same structure, learning rate, and number of epochs for a given experiment cell. I varied these 3 values across 4 total cells, testing 4 separate scenarios for each method for a total of 12 different scenarios tested. The printed results of these runs are available above the final validation output.</br>
# I began with n_epochs = 100, rho = 0.01, and a fairly simply structure of 1 hidden layer with 20 units. The results I attained were poor for sgd but very good for adam and scg. I decided to try something more complex n_epochs = 500, rho = 0.00 three hidden layers with 20, 20, and 10 layers respectively. The results here were even worse for sgd and adam with scg performing slightly less well based on the validation results. So I decided to go less complex. I used n_epochs = 50, rho = 0.05, zero hidden layers.The sgd performed significantly better on this run, adam was poor, and scg was similar to other runs. I then did a final run with increased complexity just for kicks. n_epochs = 750 rho = 0.025 and 4 hidden layers with 20, 15, 10, 5 units respectively. sgd was poor, adam was ok and scg was pretty good though not better then other attempts.</br>
# 
# After reviewing the available data I utilized sgd from my third run, adam from my first and scg from my first. I calculated the test set error on these three. In each case the validation error was very similar to the test error. The highest performer was scg, with a test percent correct of 93.23, adam was a close second with 92.51 and sgd was third with 87.87. It seemed like the fairly simple single hidden layer strategy produced the best results. More of an argument for keeping strategies simple. 

# # Grading
# 
# COMING SOON.  Download [A3grader.tar](https://www.cs.colostate.edu/~anderson/cs545/notebooks/A3grader.tar), extract `A3grader.py` before running the following cell.

# In[78]:


get_ipython().run_line_magic('run', '-i A3grader.py')


# # Extra Credit
# 
# Repeat the above experiments with a different data set.  Randonly partition your data into training, validaton and test parts if not already provided.  Write in markdown cells descriptions of the data and your results.

# In[33]:


def partition(X, T, train_fraction, validate_fraction):
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    np.random.shuffle(rows)
    
    n_train = round(n_samples * train_fraction)
    n_validate = round(n_samples * validate_fraction)
    
    Xtrain = X[rows[:n_train], :]
    Ttrain = T[rows[:n_train], :]
    Xvalidate = X[rows[n_train:n_train + n_validate], :]
    Tvalidate = T[rows[n_train:n_train + n_validate], :]
    Xtest = X[rows[n_train + n_validate:], :]
    Ttest = T[rows[n_train + n_validate:], :]
    
    return Xtrain, Ttrain, Xvalidate, Tvalidate, Xtest, Ttest


# In[34]:


import pandas
wine_data = pandas.read_csv('winequality-white.csv', delimiter=';', usecols=range(12))
wine_data = wine_data.dropna(axis=0)
wine_data.shape
T = wine_data.to_numpy()[:, -1].reshape(-1,1)
X = wine_data.to_numpy()[:, 0:11]
print(X.shape)
print(T.shape)
print(np.unique(T))
Xtrain, Ttrain, Xval, Tval, Xtest, Ttest = partition(X,T, 0.8, 0.1)


# In[36]:


n_epochs = 100
rho = 0.01
structure = [20]

wine_nnet_0_sgd = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_0_sgd.train(Xtrain, Ttrain, n_epochs, method='sgd', learning_rate=rho, verbose=False)
print_nn(wine_nnet_0_sgd, 'sgd', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
wine_nnet_0_adam = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_0_adam.train(Xtrain, Ttrain, n_epochs, method='adam', learning_rate=rho, verbose=False)     
print_nn(wine_nnet_0_adam, 'adam', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)

wine_nnet_0_scg = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_0_scg.train(Xtrain, Ttrain, n_epochs, method='scg', learning_rate=rho, verbose=False)
print_nn(wine_nnet_0_scg, 'scg', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# In[37]:


n_epochs = 500
rho = 0.001
structure = [20, 20, 10]

wine_nnet_1_sgd = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_1_sgd.train(Xtrain, Ttrain, n_epochs, method='sgd', learning_rate=rho, verbose=False)
print_nn(wine_nnet_1_sgd, 'sgd', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
wine_nnet_1_adam = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_1_adam.train(Xtrain, Ttrain, n_epochs, method='adam', learning_rate=rho, verbose=False)     
print_nn(wine_nnet_1_adam, 'adam', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)

wine_nnet_1_scg = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_1_scg.train(Xtrain, Ttrain, n_epochs, method='scg', learning_rate=rho, verbose=False)
print_nn(wine_nnet_1_scg, 'scg', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# In[38]:


n_epochs = 50
rho = 0.05
structure = []

wine_nnet_2_sgd = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_2_sgd.train(Xtrain, Ttrain, n_epochs, method='sgd', learning_rate=rho, verbose=False)
print_nn(wine_nnet_0_sgd, 'sgd', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
wine_nnet_2_adam = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_2_adam.train(Xtrain, Ttrain, n_epochs, method='adam', learning_rate=rho, verbose=False)     
print_nn(wine_nnet_0_adam, 'adam', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)

wine_nnet_2_scg = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_2_scg.train(Xtrain, Ttrain, n_epochs, method='scg', learning_rate=rho, verbose=False)
print_nn(wine_nnet_2_scg, 'scg', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# In[39]:


n_epochs = 2000
rho = 0.025
structure = [20, 15, 10, 5 ]

wine_nnet_3_sgd = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_3_sgd.train(Xtrain, Ttrain, n_epochs, method='sgd', learning_rate=rho, verbose=False)
print_nn(wine_nnet_3_sgd, 'sgd', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
    
wine_nnet_3_adam = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_3_adam.train(Xtrain, Ttrain, n_epochs, method='adam', learning_rate=rho, verbose=False)     
print_nn(wine_nnet_3_adam, 'adam', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)

wine_nnet_3_scg = nn.NeuralNetworkClassifier(Xtrain.shape[1], structure, len(np.unique(T)))
wine_nnet_3_scg.train(Xtrain, Ttrain, n_epochs, method='scg', learning_rate=rho, verbose=False)
print_nn(wine_nnet_3_scg, 'scg', rho, False, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# In[40]:


print('Best Cases')
print_nn(wine_nnet_0_sgd, 'sgd', rho, True, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)     
print_nn(wine_nnet_2_adam, 'adam', rho, True, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)
print_nn(wine_nnet_3_scg, 'scg', rho, True, Xtrain, Ttrain, Xval, Tval, Xtest, Ttest)


# I chose the Wine Quality dataset. The same one I used for assignment one: http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# 
# I did so because it was setup for classification and because I recently visited the Finger Lakes in upstate New York, which is known for producing good wine. I had the option of either the red wine or white wine dataset within these. I chose white because the Finger Lakes are known for their Rieslings. I also wanted to see if a neural network with several inputs performed better then a single variable approach. <br>
# 
# I started by importing the data and getting it cleaned up. I had to play around with it to get it into a shape amenable to training and testing. I separated this into training, validation and test data. I wanted to predict quality using the rest of the other variables.<br>
# 
# I ran it through the same training I did for the numbers dataset. The only difference was that in the final cell I did 2000 epochs. The results of the runs are above.
# The test percent correct were actually slightly better than validation in all three cases. All 3 were fairly close to their validation percent correct values. None did better than 60%/. However, a look at the confusion matrices showed that most predictions were usually mis-classifying directly to the right or left of the target. This data was for wine quality on a scale from 1 to 10, so essentially the network was producing mainly off by one or two errors. It was still well trained to be within one of the quality for a given wine. Which is pretty good. This would be a good argument for this particular problem being better served via a regression model (as I had done before with good results).  

# In[ ]:




