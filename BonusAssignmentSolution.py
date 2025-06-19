"""
Neural Bonus Assignment 
Student Name: GUNTUR MURALI LAKSHMI PRASANNA
Student ID: 700768410

PART 1: QUESTION ANSWERING WITH TRANSFORMERS
"""

from transformers import pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------
# Task 1: Basic Pipeline Setup
# ---------------------------------------------
print("\n=== Part 1: Basic QA Pipeline ===")
context = "Charles Babbage is considered the father of the computer. He conceptualized and invented the first mechanical computer in the early 19th century."
question = "Who is considered the father of the computer?"

qa_pipeline = pipeline("question-answering")
result1 = qa_pipeline(question=question, context=context)
print("Default model result:", result1)

# ---------------------------------------------
# Task 2: Use a Custom Pretrained Model
# ---------------------------------------------
qa_pipeline_custom = pipeline("question-answering", model="deepset/roberta-base-squad2")
result2 = qa_pipeline_custom(question=question, context=context)
print("Custom model result:", result2)

# ---------------------------------------------
# Task 3: Test on Your Own Example
# ---------------------------------------------
print("\n=== Custom Example ===")
custom_context = """The International Space Station (ISS) is a large spacecraft in orbit around Earth. 
It serves as a home where astronauts live and conduct experiments. 
NASA and other international space agencies collaborate to operate the ISS."""

q1 = "What is the ISS?"
q2 = "Who maintains the ISS?"

print("Q1:", qa_pipeline_custom(question=q1, context=custom_context))
print("Q2:", qa_pipeline_custom(question=q2, context=custom_context))

# ======================================================
# PART 2: CONDITIONAL GAN (cGAN) FOR MNIST DIGIT GENERATION
# ======================================================

"""
Short Answer 1:
Q: How does a Conditional GAN differ from a vanilla GAN?
A: A Conditional GAN accepts additional input such as labels, enabling the generator to produce outputs conditioned on the label. For example, we can generate only '3's or '7's from the MNIST dataset. This allows controlled generation.

Real-world application: Text-to-image generation (e.g., "Generate a picture of a blue car" or class-based medical imaging synthesis).

Short Answer 2:
Q: What does the discriminator learn in an image-to-image GAN?
A: It learns to determine if the generated image is real and if it corresponds correctly to the given input (i.e., the "pair"). 

Why is pairing important?
In image-to-image GANs, pairing ensures that the model learns proper mappings (e.g., from edge maps to photos) rather than just generating realistic-looking but unrelated images.
"""

# ---------------------------------------------
# Part 2: Conditional GAN
# ---------------------------------------------

# Hyperparameters
batch_size = 128
lr = 0.0002
epochs = 1  # Set to 1 for testing; increase for full training
noise_dim = 100
label_dim = 10
img_size = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim):
        super().__init__()
        self.label_embed = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, img_size*img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embed(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        img = self.model(x)
        return img.view(-1, 1, img_size, img_size)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super().__init__()
        self.label_embed = nn.Embedding(10, label_dim)
        self.model = nn.Sequential(
            nn.Linear(1 * img_size * img_size + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_embedding = self.label_embed(labels)
        x = torch.cat([img.view(img.size(0), -1), label_embedding], dim=1)
        return self.model(x)

# Initialize models
G = Generator(noise_dim, label_dim).to(device)
D = Discriminator(label_dim).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)

# Training loop
print("\n=== Training cGAN ===")
for epoch in range(epochs):
    for imgs, labels in train_loader:
        batch_size = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
        gen_imgs = G(z, gen_labels)
        g_loss = criterion(D(gen_imgs, gen_labels), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(D(imgs, labels), valid)
        fake_loss = criterion(D(gen_imgs.detach(), gen_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Visualization
def generate_digits_by_class():
    G.eval()
    with torch.no_grad():
        z = torch.randn(10, noise_dim).to(device)
        labels = torch.arange(0, 10).to(device)
        gen_imgs = G(z, labels).cpu()

        fig, axs = plt.subplots(1, 10, figsize=(15, 2))
        for i in range(10):
            axs[i].imshow(gen_imgs[i].squeeze(), cmap="gray")
            axs[i].set_title(f"{i}")
            axs[i].axis("off")
        plt.tight_layout()
        plt.savefig("sample_digits.png")
        plt.show()

generate_digits_by_class()
