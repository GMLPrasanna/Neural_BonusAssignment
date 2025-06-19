# Neural_BonusAssignment

**Student Name:** GUNTUR MURALI LAKSHMI PRASANNA  
**Student ID:** 700768410


---

## 🚀 Overview

This assignment is divided into two main parts:

1. **Question Answering with Transformers:**  
   A pre-trained transformer model is used to answer questions based on a provided context using Hugging Face's `pipeline`.

2. **Conditional GAN (cGAN) for MNIST Digit Generation:**  
   A conditional GAN is built from scratch using PyTorch to generate MNIST digits based on specified class labels (0–9).

---

## 🧠 Part 1: Question Answering with Transformers

### 🔧 Tools & Libraries
- `transformers` (Hugging Face)
- `torch`

### ✅ Tasks Completed

| Task | Description |
|------|-------------|
| Basic Pipeline | Used default QA pipeline to answer questions from a context |
| Custom Model | Used `deepset/roberta-base-squad2` for more accurate answers |
| Own Context | Created custom paragraph and asked two unique questions |



## 🎨 Part 2: Conditional GAN on MNIST

###🔧 Tools & Libraries
--torch, torchvision
--matplotlib for visualization

| Task          | Description                                        |
| ------------- | -------------------------------------------------- |
| Generator     | Accepts noise + label embedding to produce digits  |
| Discriminator | Accepts image + label embedding to verify validity |
| Training      | Trained on MNIST for 50 epochs with BCE loss       |
| Visualization | Displayed generated digits 0 through 9             |

