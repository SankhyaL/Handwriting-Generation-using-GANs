"""
evaluate_gan.py — DCGAN Evaluation
Matches train_gan.py: ConvTranspose2d Generator, Conv2d Discriminator,
64x64 images, latent_dim=100, num_workers=0
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import os

# ─────────────────────────────────────────
#  CONFIG — must match train_gan.py
# ─────────────────────────────────────────
LATENT_DIM  = 100
IMG_SIZE    = 64
CHANNELS    = 1
MODEL_DIR   = "gan_model"
SAVE_DIR    = "generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
#  GENERATOR (mirrors train_gan.py exactly)
# ─────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

# ─────────────────────────────────────────
#  DISCRIMINATOR (mirrors train_gan.py exactly)
# ─────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, img):
        return self.net(img).view(-1, 1).squeeze(1)

# ─────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────
G = Generator(LATENT_DIM).to(device)
D = Discriminator().to(device)

G.load_state_dict(torch.load(f"{MODEL_DIR}/generator.pth", map_location=device, weights_only=True))
D.load_state_dict(torch.load(f"{MODEL_DIR}/discriminator.pth", map_location=device, weights_only=True))
G.eval()
D.eval()

print("=" * 55)
print("           DCGAN EVALUATION RESULTS")
print("=" * 55)

# ─────────────────────────────────────────
#  1. Generate & save 64 samples
# ─────────────────────────────────────────
z = torch.randn(64, LATENT_DIM, 1, 1, device=device)
with torch.no_grad():
    generated = G(z)
save_image(generated, f"{SAVE_DIR}/final_evaluation.png", nrow=8, normalize=True)
print(f"\n[1] Generated 64 sample images → {SAVE_DIR}/final_evaluation.png")

# ─────────────────────────────────────────
#  2. Discriminator score on generated images
# ─────────────────────────────────────────
with torch.no_grad():
    d_scores = D(generated).cpu().numpy()
avg_score = d_scores.mean()
print(f"\n[2] Discriminator Score on Generated Images:")
print(f"    Average : {avg_score:.4f}  (closer to 0.5 = well-balanced GAN)")
print(f"    Min     : {d_scores.min():.4f}  |  Max: {d_scores.max():.4f}")

# ─────────────────────────────────────────
#  3. Pixel distribution similarity (simplified FID)
#     num_workers=0 — consistent with train_gan.py
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
real_data   = datasets.MNIST("./data", train=False, download=True, transform=transform)
real_loader = torch.utils.data.DataLoader(
    real_data, batch_size=500, shuffle=True, num_workers=0
)
real_imgs, _ = next(iter(real_loader))
real_imgs    = real_imgs.to(device)

with torch.no_grad():
    z         = torch.randn(500, LATENT_DIM, 1, 1, device=device)
    fake_imgs = G(z)

real_flat = real_imgs.view(500, -1).cpu().numpy()
fake_flat = fake_imgs.view(500, -1).cpu().numpy()

real_mean, real_std = real_flat.mean(axis=0), real_flat.std(axis=0)
fake_mean, fake_std = fake_flat.mean(axis=0), fake_flat.std(axis=0)

mean_diff  = np.linalg.norm(real_mean - fake_mean)
std_diff   = np.linalg.norm(real_std  - fake_std)
approx_fid = mean_diff + std_diff

print(f"\n[3] Approximate Distribution Distance (simplified FID):")
print(f"    Score : {approx_fid:.4f}  (lower = generated closer to real)")

# ─────────────────────────────────────────
#  4. Pixel statistics
# ─────────────────────────────────────────
print(f"\n[4] Pixel Statistics Comparison:")
print(f"    Real  → Mean: {real_flat.mean():.4f}  |  Std: {real_flat.std():.4f}")
print(f"    Fake  → Mean: {fake_flat.mean():.4f}  |  Std: {fake_flat.std():.4f}")

# ─────────────────────────────────────────
#  5. Discriminator score on real images
# ─────────────────────────────────────────
with torch.no_grad():
    real_scores = D(real_imgs[:64]).cpu().numpy()
print(f"\n[5] Discriminator Score on Real Images:")
print(f"    Average : {real_scores.mean():.4f}  (should be > 0.5)")

print("\n" + "=" * 55)
print("  Evaluation complete!")
print(f"  Check {SAVE_DIR}/final_evaluation.png")
print("=" * 55)
