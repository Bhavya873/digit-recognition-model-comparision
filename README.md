# Digit Recognition: classical vs deep learning

This project lets you **draw your own digits** and see how two different machine learning models, a **one-vs-all Perceptron** and a **VGG-11 Convolutional Nueral Network(CNN)**, guess what digit you wrote, live using pretrained weights!

---

## Features

- 🎨 Draw digits in your browser
- ⚡ See instant predictions as you draw
- 🔄 Switch between VGG-11 and Perceptron
- 🧹 Clear and start over anytime

---

# Why & What I Learned 

I built this to see how a basic perceptron and a deep CNN handle handwritten digits. The perceptron is fast but misses details, while VGG-11 is much smarter with messy writing. It’s eye-opening to watch your predictions update in real time! I learned how techniques like convolution, pooling, batch normalization, dropout, and backpropagation allow deep networks like VGG-11 to recognize digits far more accurately than simple linear models like perceptrons as well as how these technique result in a more accurate prediction. 

---

## Setup Guide

### 1. Clone the Repository

### 2. Install Dependencies (Requires Python 3.8+)  

`pip install -r requirements.txt`

### 3. Obtain Model Weights

**Option A: Use Pretrained Models**  
Download these files and put them in your project folder:

- **VGG-11 weights**  
  https://drive.google.com/file/d/1OS3VIUdQ3uaxsNCdPkpNEp2gFPPmN-mn/view?usp=sharing  
- **Perceptron weights**  
  https://drive.google.com/file/d/16oQBicyqhpE26Vml6ncqk-hHPOTg3Xos/view?usp=sharing  

**Option B: Train Models Yourself**  
Run the Jupyter notebooks:
- `vgg11_mnist.ipynb` — trains and saves `vgg11_mnist.pth`
- `perceptron_ova_mnist.ipynb` — trains and saves `perceptron_ova_mnist.pth`

Move the `.pth` files to the project root after training.

### 4. Run the App

`python app.py`

Open the URL shown (e.g., `http://localhost:7860`) in your browser and start drawing!

---

## File Overview

- **app.py**: Gradio interface, sketchpad, real-time prediction logic.
- **cnn.py**: VGG-11 CNN architecture.
- **perceptron.py**: Perceptron model and loader.
- **vgg11_mnist.ipynb**: Train and save VGG-11 weights.
- **perceptron_ova_mnist.ipynb**: Train and save Perceptron weights.
- **requirements.txt**: Python dependencies.

---

Have fun experimenting and see the difference between classic and deep learning for yourself!


