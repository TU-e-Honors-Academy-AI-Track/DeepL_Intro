## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a category of deep neural networks designed specifically for processing structured grid data, such as images. The fundamental building block of a CNN is the convolutional layer, which performs a convolution operation. Mathematically, given an input matrix III and a kernel KKK, the convolution operation is defined as:

$$ (I \ast K)(x, y) = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} I(i, j) K(x-i, y-j) $$

In the context of CNNs, III represents a section of the input image, and KKK is a filter or kernel. The convolution operation involves sliding the kernel over the input image and computing the sum of element-wise products.
![Alt text](image.png)





### The Seminal LeNet Architecture

The Convolutional Neural Network (CNN) we are implementing here with PyTorch is the seminal LeNet architecture, first proposed by one of the grandfathers of deep learning, Yann LeCunn. By today’s standards, LeNet is a very shallow neural network, consisting of the following layers:


CONV⇒RELU⇒POOL)×2⇒FC⇒RELU⇒FC⇒SOFTMAX

### Implementing Simple CNN with PyTorch

```
pythonCopy codefrom torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax
from torch import flatten

class LeNet(Module):
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()

        # First set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # First (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()

        # Softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
```
The code can be visualised as follows([source](https://www.researchgate.net/publication/318972455_Automated_Breast_Ultrasound_Lesions_Detection_Using_Convolutional_Neural_Networks):
![Image](../Images/pytorch_cnn_lenet.webp "The architecture of simple LeNet CNN for image classification ")
1 / 1

Used **BrowserOp**

## Best Practices in Building Deep CNNs

Building a deep Convolutional Neural Network (CNN) requires a combination of architectural decisions, layer arrangements, and additional techniques to improve the network's performance and efficiency. Here are some practices and methodologies commonly adopted in the deep learning community:

### Layer Arrangement

1.  **Convolutional Layers**: These are the foundational layers in a CNN. They perform a convolution operation on the input data using filters or kernels to extract features. In a CNN, the input is typically a tensor with shape: (number of inputs) × (input height) × (input width) × (input channels). After passing through a convolutional layer, the image becomes abstracted to a feature map.
    
2.  **Pooling Layers**: These layers reduce the dimensions of data by combining the outputs of neuron clusters in one layer into a single neuron in the next layer. There are two common types of pooling: max and average. Max pooling uses the maximum value of each local cluster of neurons in the feature map, while average pooling takes the average value.
    
3.  **Fully Connected Layers**: These layers connect every neuron in one layer to every neuron in another layer, similar to a traditional multilayer perceptron neural network (MLP). The flattened matrix from the previous layers goes through a fully connected layer to classify the images.
    
4.  **Normalization Layers**: These layers are used to normalize the activations of the neurons, which can accelerate the training process and improve generalization.
    

### Advanced Techniques

1.  **Attention Mechanisms (computationally expensive)**: Attention mechanisms are a critical component in many state-of-the-art models, especially in the context of sequence-to-sequence tasks like machine translation, text summarization, and image captioning. The primary idea behind attention is to allow the model to focus on specific parts of the input when producing an output.

Here's a simple implementation of the attention mechanism in PyTorch, in this case its simple version callsed Soft Attention Mechanism, while the steps are as follows:



1.  **Calculate Attention Weights**: This is done using the current state of the decoder and all encoder outputs. The attention weights represent the importance of each item in the source sequence for the current decoding step.
    
2.  **Compute Context Vector**: The context vector is a weighted sum of the encoder outputs, using the attention weights.
    
3.  **Combine Context Vector with Current Decoder State**: This helps the decoder focus on relevant parts of the input sequence.
    

Here's a basic implementation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch size, dec hid dim]
        # encoder_outputs: [src len, batch size, enc hid dim]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # Calculate energy for each encoder output
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # Return normalized attention weights
        return F.softmax(attention, dim=1)


```

In a typical sequence-to-sequence model with attention:

1.  The encoder processes the input sequence and produces a set of outputs.
2.  The decoder starts producing the output sequence. At each step, it uses the attention mechanism to get a context vector from the encoder outputs.
3.  The context vector is combined with the decoder's state to produce the output for that step and the next state.

This is a basic form of attention called "soft attention". There are other forms like "hard attention" and more sophisticated mechanisms like "multi-head attention" used in models like the Transformer. The above code provides a foundation to understand the core concept.
2.  **Skip Connections (or Residual Connections)**: These are shortcuts or connections that skip one or more layers. They were introduced to solve the vanishing gradient problem in very deep networks. The idea is to add the output of a layer to the output of a layer a few steps further down the network. This can be easily implemented in PyTorch using the `nn.Sequential` and `nn.Module` classes.
    
    ```python
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
    
        def forward(self, x):
            residual = x
            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            x += self.skip(residual)
            return F.relu(x)
    
    ```
    
3.  **Dilated Convolutions**: These are used to increase the receptive field of a neuron without increasing the number of parameters. They introduce gaps in the kernel, allowing it to cover a larger area of the input.