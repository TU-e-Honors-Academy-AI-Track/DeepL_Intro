# Introduction to Neural Networks for Absolute Beginners
Neural networks are a machine learning technique with which we intend to approximate a (possibly very complicated) function (or mapping) using some kind of graph constructed out of simple neurons. 

Lets take a look at such a neuron, one which has n inputs $x_1, ..., x_n$, with corresponding weights $w_1, ..., w_n$.
![0-NN/neuron.png](https://github.com/TU-e-Honors-Academy-AI-Track/DeepL_Intro/blob/00introtoNN/0-NN/neuron.png?raw=true)

This neuron would take its inputs, multiply them with their weights and sum them up, together with a bias $b$. Finally, it will run them through a non-linear activation function $g(x)$, as to allow the network to approximate non-linear functions (this can be done with the inaccuracy of floats, but thats less effective). So the output $y$ of the neuron would be $y=g(b +\sum$ $_{i=1}^{n} x_i*w_i)$.

A neural network would consists of those neurons, commonly divided in an input layer where the neurons get input from the outside, some amount of hidden layers and an output layer, in these neurons commonly get their inputs from the neurons in the previous layer. Here we have an example with a 3 neuron input layer (left), hidden layer (middle) and a 1 neuron output layer (right).

![0-NN/nn.png](https://github.com/TU-e-Honors-Academy-AI-Track/DeepL_Intro/blob/00introtoNN/0-NN/nn.png?raw=true)

We can express parts of this network using linear algebra, the input layer is just given a vector of all inputs, lets call it **x** $= \[x_1, x_2, x_3\]$. We can then express the outcome of the second, hidden, layer as **y**. If all weights of the second layer are stored in a matrix **A** ..... WORK IN PROGRESS
