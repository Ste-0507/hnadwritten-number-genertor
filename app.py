import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels = self.label_embed(labels)
        x = torch.cat((noise, labels), dim=1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

device = torch.device("cpu")
model = Generator()
model.load_state_dict(torch.load("mnist_generator.pth", map_location=device))
model.eval()

st.title("🧠 Handwritten Digit Generator (0-9)")
digit = st.selectbox("Choose a digit to generate", list(range(10)))

if st.button("Generate 5 Images"):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        images = model(z, labels)

    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i][0], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
