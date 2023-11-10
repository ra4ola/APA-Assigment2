# APA-Assigment2


## 1. Intro

In this assignment, it is expected that you explore autoencoders in reconstruction and denoising tasks, and exploit an object detector network with real world data. This assignment is split in two parts: image reconstruction and object detection. This assignment is part of your continuous evaluation. You should provide a report and notebooks (.ipynb), which include your development and analysis for the requested tasks.


## 2. Dataset

In this assignment you will be using the MNIST dataset (used in the first CNN code) and a subset of the KITTI1 object dataset. The KITTI dataset was recorded in outdoor environments using an ego-vehicle equipped with different sensors, including an RGB camera. For the purpose of this work, a KITTI subset containing 500 raw RGB images (the images are not object-centric images) as well as their labels for the 3 object classes (Car, Person, and Cyclist), was prepared, and it is available to download at https://drive. google.com/file/d/1xMKIf6igwCYWMFTjsovlt04rJwgWUhbj/view. A raw RGB image may contain more than one object, where each row in the label file represents an object (bounding box) with the following structure:

```object_class_id normalized_center_x normalized_center_y normalized_width normalized_height```

It is important to note that an image can contain no objects.

## 3. Parte I: Reconstruction and denoising using Autoencoders

### 3.1 Background Materials

In this first part of the assignment, you will define a suitable network composed by encoder and de- coder architectures to process MNIST images. You will implement an Autoencoder (AE), a Variational Autoencoder (VAE) and a Denoising Autoencoder (DAE).

#### 3.1.1 Autoencoders


Implementing AEs is a simple process in PyTorch. Similarly to how a network is built, you must first define the encoder layer and decoder layer and later call them in the ”forward(self,input)” function. You must also define a loss function that measures how similar (or how distant) the decoding and the input are. Commonly, Mean Squared Error (MSE) and Binary Cross Entropy (BCE) are used as loss functions for reconstruction problems. The training code from the previous assignment can be slightly modified to suit a reconstruction task, taking into account that the loss now considers the batch images and the batch decodings. It is important to note that the last activation function in the decoder layer plays an important role in the reconstruction task, as it should map the pixel values correctly. Although it is possible to create encoder/decoder layers using linear/fully connected layers, since we are working with image data, convolutional encoder/decoder layers should be used. You are not required to make the encoder and decoder a complete mirror of each other (e.g., MaxPool2D is not fully invertible), however, for most cases, ”inverse” layers are required (Conv2d → ConvTranspose2d).

#### 3.1.2 Variational Autoencoders
VAE is a type of generative model that combines elements of AEs and probabilistic modeling, meaning that from an initial AE network, modifications can be introduced to build a VAE network. The encoder layer of the AE, outputs a deterministic representation of the feature map, also known as bottleneck. To implement a VAE, the first step will be to modify the encoder’s output layer to output the mean (μ) and the log-variance (log σ2) of a probability distribution over the latent space. The problem now becomes how to sample from the distribution while maintaining the network traversable during back propagation, as the efficient calculation of gradients during the backpropagation process is essential for training neural networks. This is known as the reparameterization trick and consists in adding a sampling layer that generates random samples taken from a normal distribution and scales it by the standard deviation (represented by the log- variance). The latent variable z is modeled as a Gaussian distribution with mean μ and standard deviation σ, and ε is a sample from a standard normal distribution $(ε ∼ N (0, 1))$. Since the VAE learns the log-variance (as it improves numerical stability) the reparameterization trick is expressed as:

$$ z = μ + e0.5(log σ2) · ε $$
