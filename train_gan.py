import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# ---- CONFIG ----
LATENT_DIM = 64
EPOCHS = 30          # enough for visible results on CPU
BATCH_SIZE = 64
LR = 0.0002
IMG_SIZE = 28
SAVE_DIR = "generated_images"
MODEL_DIR = "gan_model"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- DATA ----
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # normalize to [-1, 1]
])
dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- GENERATOR ----
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z).view(-1, 1, IMG_SIZE, IMG_SIZE)

# ---- DISCRIMINATOR ----
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

# ---- INIT ----
G = Generator()
D = Discriminator()
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

print("Starting GAN training on CPU (~15-25 min)...")
print("Watch the loss values — G_loss and D_loss should stay balanced\n")

for epoch in range(EPOCHS):
    for i, (imgs, _) in enumerate(dataloader):
        batch = imgs.size(0)
        real = torch.ones(batch, 1)
        fake = torch.zeros(batch, 1)

        # --- Train Discriminator ---
        opt_D.zero_grad()
        real_loss = criterion(D(imgs), real)
        z = torch.randn(batch, LATENT_DIM)
        fake_imgs = G(z).detach()
        fake_loss = criterion(D(fake_imgs), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        opt_D.step()

        # --- Train Generator ---
        opt_G.zero_grad()
        z = torch.randn(batch, LATENT_DIM)
        gen_imgs = G(z)
        g_loss = criterion(D(gen_imgs), real)  # fool discriminator
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    # Save sample images every 5 epochs
    if (epoch + 1) % 5 == 0:
        z = torch.randn(16, LATENT_DIM)
        samples = G(z)
        save_image(samples, f"{SAVE_DIR}/epoch_{epoch+1}.png", nrow=4, normalize=True)
        print(f"  >> Saved sample images to {SAVE_DIR}/epoch_{epoch+1}.png")

# Save models
torch.save(G.state_dict(), f"{MODEL_DIR}/generator.pth")
torch.save(D.state_dict(), f"{MODEL_DIR}/discriminator.pth")
print(f"\nTraining complete! Models saved to {MODEL_DIR}/")
print(f"Check {SAVE_DIR}/ folder for generated digit images")