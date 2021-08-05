Multilayer Perceptron -

Using 3Blue1Brown for a recap

Notes:

Value in a neuron is called activation.

Each layer is grouping of neurons.

Weighted sum is parsed into the second hidden layer.

Weighted sums are pumped into a sigmoid to convert values into a range of 0-1

 Bias can be applied with weighted sum before pumped into the sigmoid function.


Activation functions can range from sigmoid, relu

sigmoid:
a(1) = sigmoid(W*a(0) + b)
Slower than ReLU

relu:
ReLU(a) = max(0, a)
AKA: if x =< 0 then RELU(x) = 0 and if x > 0 then ReLU(x) = x

Cost Function:
  Adding up square of difference and value you want them to have.
  Then find the average of each cost

Gradient Descent:
  Using value from the cost function to find the local minimum via finding negative gradient of the cost function.
  The resulting minimum vector from the local minimum can be added back to the original parameters to shift all of them accordingly

Backpropagation:
  Increase Bias
  Increase Weights - Weights with neurons with larger activity (in proporton to ai)
  Or change previous weights - making second to last layer brighter (in proporton to wi)
  Sum of changes needed by all neurons in the output creating a list of change magnitiudes which can be applied backwards.
  Summing up all changes at the end is a value which is proportonal to the negitive gradient.

Stochastic gradient Descent:
  Breaking down training data in to smaller clumps which can each create there approximate gradient descent


LSTM -
