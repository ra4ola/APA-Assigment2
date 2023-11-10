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

$$ z = μ + e^{0.5(log σ^{2})} · ε $$

The input of the VAE’s decoder layer becomes the sampled latent vector. Finally, the total loss for the VAE is composed of two parts, the reconstruction loss and the KL Divergence loss. The reconstruction loss is the loss used in the AE. The KL Divergence loss is introduced to evaluate the distance between the learned distribution and a chosen prior (commonly a standard normal distribution). The KL Divergence loss is expressed as:

$$ D_{KL} = -\frac{1}{2} \sum_{i} (1 - \mu_{i}^{2} + \log(\sigma_{i}^{2}) - e^{\log(\sigma_{i}^{2})}) $$

where N is the dimensionality of the latent space.

#### 3.1.3 Denoising Autoencoders
The main goal of a DAE is to learn how to denoise the input data, which can improve its ability to extract meaningful features. For that reason, the only step required is the generation of noisy inputs:

```
def add_noise(data, noise_factor=0.5):
  noisy_data = data + noise_factor * torch.randn_like(data)
  return torch.clamp(noisy_data , 0., 1.)
```

#### 3.1.4 Evaluation Metrics
Since Autoencoders are data-driven, the evaluation of your models will be in terms of reconstruction errors and the qualitative representation of some reconstruction pairs.

### 3.2 Tasks
* Implement an AE network (the network of the first assignment can be a good starting point); You can use a fully-connected-based encoder-decoder, however a convolution-based encoder-decoder will be valued.
* Modify the encoder, add the reparametrization trick and add the VAE loss to the aforementioned network;
* Implement the DAE’s pipeline;
* Implement the aforementioned evaluation metrics to evaluate your network;
* Gather the metrics and loss curves for all training conditions.
* Use t-SNE (t-distributed Stochastic Neighbor Embedding) to visualize the bottleneck/latent space of the AE, VAE and DAE. Example:

```
latent_space = [] labels = []
  with torch.no_grad():
  for batch_idx , data in enumerate(training_loader):
    ae.eval()
    mu= ae.encode(data[0].to(device))
    latent_space.append(mu)
    labels.append(data[1])

latent_space = torch.cat(latent_space , dim=0).cpu().numpy() labels = torch.cat(labels, dim=0).cpu().numpy()
```
```
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42) latent_tsne = tsne.fit_transform(latent_space)
import matplotlib.pyplot as plt
plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='viridis') plt.colorbar()
plt.title("t-SNE Visualization of Latent Space")
plt.show()
```
* Apply transfer learning using the encoder layer of the AE, VAE and DAE. Add a classification layer to all the encoders (individual), freeze the encoder layer and only train the classification layer. Compare the classification results obtained, and how well the learned representations generalize.


## 4 Part II: Object Detection

### 4.1 Background Materials

In this second part of the assignment, you will use the object detector YOLOv52 network. YOLO is one of the most successful real-time CNN-based object detectors developed to date.

#### 4.1.1 Evaluation Metrics

Evaluating an object detection network is not the same as evaluating an object classification one. In an object detection network, the prediction results are bounding boxes within an image representing the detected objects, instead of a category label associated with an image. These predicted bounding boxes need to be evaluated by the Intersection over Union (IoU) metric, which evaluates the precision of each predicted bounding box with the ground-truth. IoU is represented by (3). The individual bounding box evaluation is the base of the object detection method evaluation metrics, designed by mean-Average Precision (mAP), which is based on Precision and Recall evaluation metrics as shown in (4), (5), (6), and (7).


$$ IoU = \frac{{\text{{area of overlap}}}}{{\text{{area of union}}}} $$ (3)

$$ Precision = TP / (TP + FP) $$ (4)

$$ Recall = TP / (TP + FN) $$ (5)

$$ F1 Score = 2 * Precision * Recall / (Precision + Recall) $$ (6)

$$ Average Precision = ∫ Precision(Recall) dRecall, from 0 to 1 $$ (7)








