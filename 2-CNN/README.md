## Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a category of deep neural networks designed specifically for processing structured grid data, such as images. The fundamental building block of a CNN is the convolutional layer, which performs a convolution operation. Mathematically, given an input matrix III and a kernel KKK, the convolution operation is defined as:

$$ (I \ast K)(x, y) = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} I(i, j) K(x-i, y-j) $$

In the context of CNNs, III represents a section of the input image, and KKK is a filter or kernel. The convolution operation involves sliding the kernel over the input image and computing the sum of element-wise products.
![Alt text](image.png)


In order to build a basic CNN in this section the following funcitons are crucial:

    
  `torch.nn.Conv2d`: PyTorch’s implementation of convolutional layers.

  
  `torch.nn.Linear`: Fully connected layers.

  `torch.nn.MaxPool2d`: Applies 2D max-pooling to reduce the spatial dimensions of the input volume.

  
  `torch.nn.ReLU`: Our ReLU function.

  
  `torch.nn.LogSoftmax`: Used when building our softmax classifier to return the predicted probabilities of each class.

  
  `torch.nn.flatten`: Flattens the output of a multi-dimensional volume (e.g., a CONV or POOL layer) such that we can apply fully connected layers to it.


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

The ordering of layers in a Convolutional Neural Network (CNN) can have a significant impact on the performance and training dynamics of the model. Let's discuss the two proposed schemes:

#### Scheme 1:

CONV/FC→ReLU (or other activation)→Dropout→BatchNorm→CONV/F

##### Explanation:

1.  **CONV/FC**: The convolutional (CONV) or fully connected (FC) layer is where the primary computation happens. It involves applying filters to the input data (in the case of CONV) or connecting every neuron to every other neuron (in the case of FC).
    
2.  **ReLU (or other activation)**: Activation functions introduce non-linearity into the model. ReLU (Rectified Linear Unit) is the most commonly used activation function in CNNs due to its simplicity and effectiveness.
    
3.  **Dropout**: Dropout is a regularization technique where randomly selected neurons are ignored during training, helping to prevent overfitting.
    
4.  **BatchNorm**: Batch normalization normalizes the activations of the neurons, which can accelerate the training process and improve generalization. By placing it after dropout, we ensure that the normalization is not affected by the randomness of dropout.
    
5.  **CONV/FC**: Another convolutional or fully connected layer follows, continuing the pattern.
    

#### Scheme 2:

CONV/FC→BatchNorm→ReLU (or other activation)→Dropout→CONV/FC
##### Explanation:

1.  **CONV/FC**: Similar to Scheme 1, this is where the primary computation happens.
    
2.  **BatchNorm**: Here, batch normalization is applied immediately after the convolutional or fully connected layer. This ensures that the data fed into the activation function is normalized.
    
3.  **ReLU (or other activation)**: The activation function is applied after normalization.
    
4.  **Dropout**: Dropout is applied after the activation, serving as a regularization technique.
    
5.  **CONV/FC**: Another convolutional or fully connected layer follows.
    

#### Which Scheme to Use?

The choice between Scheme 1 and Scheme 2 often depends on empirical results and the specific problem at hand. However, Scheme 2 is more commonly adopted in recent deep learning practices. The rationale is that by normalizing immediately after the convolutional or fully connected layer, the data fed into the activation function is more consistent, leading to more stable training dynamics. Additionally, applying dropout after the activation function can be more effective as it drops the activated features.

In practice, it's beneficial to experiment with both schemes and observe which one offers better performance for your specific task.
    
Here's a simple implementation of the sequence `CONV/FC → BatchNorm → ReLU (or other activation) → Dropout → CONV/FC` in PyTorch:

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()

        # First sequence: CONV -> BatchNorm -> ReLU -> Dropout
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        # Second sequence: CONV
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully Connected layer (FC)
        self.fc = nn.Linear(64 * 28 * 28, num_classes)  # Assuming input image size is 28x28

    def forward(self, x):
        # First sequence
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second sequence
        x = self.conv2(x)

        # Flatten and pass through FC
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Instantiate the model
model = SimpleCNN(input_channels=1, num_classes=10)
print(model)
```
If a larger model is used, it is better to stick to the widely used mode archietctures and backbones that are already available such as existing Resnet based architectures or U-net base dones depending on the application. These architectures have already optimized and it is not necessary to search for another optimal layer arrangements.
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