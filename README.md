# Variational Autoencoder (VAE) & Conditional Variational Autoencoder (CVAE) in PyTorch

A PyTorch implementation of Variational Autoencoders (VAE) and Conditional Variational Autoencoders (CVAE) for generating and reconstructing MNIST digits. This project explores generative modeling using Deep Neural Networks, leveraging the reparameterization trick and probabilistic inference techniques.

## 📌 Features

-   **Variational Autoencoder (VAE)**: Encodes and reconstructs MNIST digits by learning a probabilistic latent space.
-   **Conditional Variational Autoencoder (CVAE)**: Extends VAE by conditioning on digit labels, enabling targeted generation.
-   **Reparameterization Trick**: Enables backpropagation through a stochastic latent space.
-   **ELBO & CELBO Loss Functions**: Implements the Evidence Lower Bound (ELBO) and Conditional ELBO (CELBO) for optimization.
-   **Training & Evaluation**: Scripts for training both models on the MNIST dataset.
-   **Visualization**: Generates and visualizes reconstructed and generated digits.

## 🛠 Technologies

This project is built using **PyTorch**, a popular deep learning framework, and relies on several key libraries:

| Module                          | Description                                                    |
| ------------------------------- | -------------------------------------------------------------- |
| **PyTorch (`torch`)**           | Provides the deep learning framework and tensor operations.    |
| **TorchVision (`torchvision`)** | Includes the MNIST dataset and image transformation utilities. |
| **Matplotlib (`matplotlib`)**   | Used for visualizing training progress and generated digits.   |
| **NumPy (`numpy`)**             | Supports numerical operations for data manipulation.           |

## 📥 Installation

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/yourusername/vae-cvae-pytorch.git
cd vae-cvae-pytorch
```

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Training and Visualization Script

```sh
python main.py
```

## 🏗 Model Architecture

### **Variational Autoencoder (VAE)**

A VAE consists of:

-   **Encoder**: Maps input images to a latent distribution (mean & log variance).
-   **Reparameterization Trick**: Samples latent variables using a differentiable trick.
-   **Decoder**: Reconstructs images from sampled latent vectors.

### **Conditional Variational Autoencoder (CVAE)**

Extends VAE by conditioning both the encoder and decoder on class labels:

-   **Conditioning Mechanism**: Labels are concatenated with inputs and latent variables.
-   **Targeted Generation**: Allows for class-specific sampling.


## 📜 License
This project is open-source under the [License](./LICENSE).

## 📩 Contact

For questions, contributions, or issues, feel free to open a GitHub issue or reach out.
