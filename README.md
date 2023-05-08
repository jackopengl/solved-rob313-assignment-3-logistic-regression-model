Download Link: https://assignmentchef.com/product/solved-rob313-assignment-3-logistic-regression-model
<br>
<h1></h1>

<strong>Q1) 4pts </strong>Use gradient descent to learn the weights of a logistic regression model.  Logistic regression is used for classification problems (i.e. <em>y</em><sup>(<em>i</em>) </sup>∈ {0<em>,</em>1} in the binary case which we will consider), and uses the Bernoulli likelihood

Pr((<strong>x</strong>;<strong>w</strong>) (<strong>x</strong>;<strong>w</strong>) <em>,</em>

where ) gives the class conditional probability of class 1 by mapping 1]. To ensure that the model gives a valid probability in the range [0,1], we write <em>f </em>as a logistic sigmoid acting on a linear model as follows

<em>f</em><sub>b</sub>(<strong>x</strong>;<strong>w</strong>) = sigmoid<em>,</em>

where sigmoid, and <strong>w </strong>= {<em>w</em><sub>0</sub><em>,w</em><sub>1</sub><em>,…,w<sub>D</sub></em>} ∈ R<em><sup>D</sup></em><sup>+<a href="#_ftn1" name="_ftnref1">[1]</a></sup>. Making the assumption that all training examples are <em>i.i.d.</em>, the log-likelihood function can be written as follows for the logistic regression model

<em>N </em>logPr(<strong>y</strong><em>.</em>

<em>i</em>=1

What will be the value of the log-likelihood if ) = 1, but the correct label is <em>y</em><sup>(<em>i</em>) </sup>= 0 for some <em>i</em>? Is this reasonable behaviour?

Initializing all weights to zero, find the maximum-likelihood estimate of the parameters using both full-batch gradient descent (GD), as well as stochastic gradient descent (SGD) with a mini-batch size of 1. Analyze the convergence trends of both optimization methods by plotting the loss versus iteration number on a single plot and report the learning rates used. The gradient of the log-likelihood function with respect to the weights can be written as follows

<em>N</em>

∇logPr(<strong>y</strong><em>,</em>

<em>i</em>=1

where we used the convenient form of the derivative of the sigmoid function sigmoid(<em>z</em>) = sigmoid(<em>z</em>) 1 − sigmoid .

Train the logistic regression model on the iris dataset, considering only the second response to determine whether the flower is an <em>iris virginica</em>, or not<sup>1</sup>. Use both the training and validation sets to predict on the test set, and present test accuracy as well as the test log-likelihood. Why might the test log-likelihood be a preferable performance metric?

<strong>Q2) 7pts </strong>In the previous question we computed gradients manually for a linear logistic regression problem. This question will consider training a more complicated model, a deep neural network, and to help us compute gradients we will use the automatic differentiation package autograd. To install autograd, run the following in a terminal (mac or linux), or Anaconda prompt (windows)

conda install -c conda-forge autograd

The ipython notebook used for the in-class autograd tutorial can be found on portal.

In this assignment you will train a fully connected neural network with two hidden layers on the MNISTsmall dataset using a categorical (generalized Bernoulli) likelihood. Using a mini-batch<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a> size of 250, train the weights and bias parameters of the neural network using stochastic gradient descent. Initialize the biases of the model to zero and initialize the weights randomly.

You are provided the python module a3mod.py which can be found on portal. A brief description of each function is provided here but more details can be found by reviewing the docstrings and inline comments.

a3mod.forwardpass computes the forward pass of a two layer neural network. The output layer activation function will need to be modified in this assignment.

a3mod.negativeloglikelihood computes the negative log-likelihood of the neural network defined in a3mod.forwardpass. This function will need to be modified in this assignment.

a3mod.nllgradients returns the negative log-likelihood computed by a3mod.negativeloglikelihood, as well as the gradients of this value with respect to all weights and biases in the neural network. You should not need to modify this function. a3mod.runexample this function demonstrates the computation of the negative log-likelihood and its gradients. It is intended as an example to get you familiar with the code and you may modify this function any way you wish.

Before beginning this question, you are encouraged to review the python code for these functions which is short and well documented. Running (and modifying) the a3mod.runexample function can also be helpful to understand the provided code.

<ol>

 <li><strong>2pts </strong>Since we plan to maximize the log-likelihood using a categorical likelihood, we would like our neural network to have 10 outputs, each a class-conditional log probability for each of the 10 classes for the mnistsmall The two hidden layer neural network defined in a3mod.forwardpass initially has a linear activation function on the output layer, however, these outputs do not define valid classconditional log probabilities. Modify a3mod.forwardpass so that a log-softmax activation function is used on the output layer. For your implementation, use only (autograd wrapped) numpy functions, do not use any loops, and ensure that your implementation is numerically stable. Briefly describe your implementation and why it is numerically stable. Hint: consider the LogSumExp trick we covered in class.</li>

 <li><strong>2pts </strong>The function negativeloglikelihood currently assumes a Gaussian likelihood, however, we would like to use a categorical likelihood. Modify this function such that the negative log-likelihood is returned for a mini-batch of inputs assuming that the outputs of a3mod.forwardpass are class conditional log probabilities. For your implementation, use only (autograd wrapped) numpy functions, and do not use any loops.</li>

 <li><strong>2pts </strong>Considering 100 neurons per hidden layer, plot the stochastic estimate of the training set negative log-likelihood (using a mini-batch size of 250) versus iteration number during training. In the same plot, also draw the validation set negative log-likelihood versus iteration number. How does the network’s performance differ on the training set versus the validation set during learning?</li>

 <li><strong>1pts </strong>Plot a few test set digits where the neural network is not confident of the classification output (i.e. the top class conditional probability is below some threshold),</li>

</ol>

and comment on them. You may find datautils.plotdigit helpful.

<a href="#_ftnref1" name="_ftn1">[1]</a> Use, y train, y valid, y test = y train[:,(1,)], y valid[:,(1,)], y test[:,(1,)]

<a href="#_ftnref2" name="_ftn2">[2]</a> At no point in the assignment should you need to perform an operation (e.g. a forward pass) with the full training batch. If you do this then you are doing something wrong and should re-read carefully.