"""
Advanced DCGAN — Handwriting Generation
- Architecture : Deep Convolutional GAN (Radford et al., 2015)
- Dataset      : MNIST (auto-downloaded)
- Hardware     : Multi-GPU via DataParallel, falls back to CPU
- Epochs       : 100 (configurable via CLI)
- Output       : generated_images/, gan_model/, loss_curves/

Usage:
    python train_gan.py
    python train_gan.py --epochs 100 --batch_size 128 --dataset mnist
    python train_gan.py --epochs 100 --dataset fashionmnist
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────
#  CLI ARGUMENTS
# ─────────────────────────────────────────
parser = argparse.ArgumentParser(description="DCGAN Handwriting Generator")
parser.add_argument("--epochs",      type=int,   default=100)
parser.add_argument("--batch_size",  type=int,   default=128)
parser.add_argument("--lr",          type=float, default=0.0002)
parser.add_argument("--latent_dim",  type=int,   default=100)
parser.add_argument("--dataset",     type=str,   default="mnist",
                    choices=["mnist", "fashionmnist"])
parser.add_argument("--save_every",  type=int,   default=10,
                    help="Save sample images every N epochs")
args = parser.parse_args()

# ─────────────────────────────────────────
#  DIRECTORIES
# ─────────────────────────────────────────
SAVE_DIR  = "generated_images"
MODEL_DIR = "gan_model"
CURVE_DIR = "loss_curves"
for d in [SAVE_DIR, MODEL_DIR, CURVE_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────
#  DEVICE — multi-GPU if available
# ─────────────────────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus   = torch.cuda.device_count()
print(f"\n{'='*55}")
print(f"  DCGAN — {args.dataset.upper()} | {args.epochs} Epochs")
print(f"  Device : {device}  |  GPUs available: {num_gpus}")
print(f"  Batch  : {args.batch_size}  |  Latent dim: {args.latent_dim}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────
IMG_SIZE   = 64   # upsample MNIST 28→64 for proper conv layers
CHANNELS   = 1    # grayscale

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
])

DatasetClass = datasets.MNIST if args.dataset == "mnist" else datasets.FashionMNIST
dataset = DatasetClass(
    root="./data", train=True, download=True, transform=transform
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,          # parallel data loading (i9 has plenty of cores)
    pin_memory=True,        # faster GPU transfer
    drop_last=True,
)
print(f"  Dataset : {args.dataset} | {len(dataset):,} images | "
      f"{len(dataloader)} batches/epoch\n")

# ─────────────────────────────────────────
#  WEIGHT INIT  (DCGAN paper: N(0, 0.02))
# ─────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ─────────────────────────────────────────
#  GENERATOR  (latent z → 64×64 image)
#
#  Each ConvTranspose2d doubles spatial size.
#  BN + ReLU on all layers except output (Tanh).
#  Feature map sequence: z(100) → 512 → 256 → 128 → 64 → 1
# ─────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            # z → (512, 4, 4)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # → (256, 8, 8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # → (128, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # → (64, 32, 32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # → (1, 64, 64)
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()   # output in [-1, 1]
        )

    def forward(self, z):
        # z shape: (batch, latent_dim, 1, 1)
        return self.net(z)

# ─────────────────────────────────────────
#  DISCRIMINATOR  (64×64 image → real/fake)
#
#  Each Conv2d halves spatial size.
#  LeakyReLU(0.2) — no MaxPool (DCGAN guideline).
#  No BN in first layer (DCGAN guideline).
# ─────────────────────────────────────────
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (1, 64, 64) → (64, 32, 32)  — no BN on first layer
            nn.Conv2d(CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → (128, 16, 16)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # → (256, 8, 8)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # → (512, 4, 4)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # → (1, 1, 1) scalar
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.net(img).view(-1, 1).squeeze(1)

# ─────────────────────────────────────────
#  INSTANTIATE + MULTI-GPU WRAP
# ─────────────────────────────────────────
G = Generator(args.latent_dim).to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

if num_gpus > 1:
    print(f"  Using DataParallel across {num_gpus} GPUs\n")
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

# ─────────────────────────────────────────
#  LOSS + OPTIMIZERS
# ─────────────────────────────────────────
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Learning rate schedulers — decay LR by 0.5 at epoch 50 and 75
scheduler_G = optim.lr_scheduler.MultiStepLR(opt_G, milestones=[50, 75], gamma=0.5)
scheduler_D = optim.lr_scheduler.MultiStepLR(opt_D, milestones=[50, 75], gamma=0.5)

# Fixed noise to track generator progress across epochs
fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

# ─────────────────────────────────────────
#  LABEL SMOOTHING HELPERS
#  Real labels: 0.9 instead of 1.0  →  prevents D from being overconfident
# ─────────────────────────────────────────
def real_labels(size):
    return torch.full((size,), 0.9, device=device)

def fake_labels(size):
    return torch.zeros(size, device=device)

# ─────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────
g_losses, d_losses = [], []
print("Starting training...\n")

for epoch in range(1, args.epochs + 1):
    G.train()
    D.train()

    epoch_g_loss = 0.0
    epoch_d_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:>3}/{args.epochs}", leave=False)

    for real_imgs, _ in pbar:
        batch_size = real_imgs.size(0)
        real_imgs  = real_imgs.to(device)

        # ── Train Discriminator ──────────────────────
        opt_D.zero_grad()

        # Real images
        out_real = D(real_imgs)
        loss_real = criterion(out_real, real_labels(batch_size))

        # Fake images (detach so G gradients don't flow here)
        noise     = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
        fake_imgs = G(noise).detach()
        out_fake  = D(fake_imgs)
        loss_fake = criterion(out_fake, fake_labels(batch_size))

        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        opt_D.step()

        # ── Train Generator ──────────────────────────
        opt_G.zero_grad()

        noise    = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
        gen_imgs = G(noise)
        out_gen  = D(gen_imgs)
        # Generator wants D to output 1 (real) for its fakes
        g_loss   = criterion(out_gen, real_labels(batch_size))
        g_loss.backward()
        opt_G.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()

        pbar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")

    # Average losses
    avg_d = epoch_d_loss / len(dataloader)
    avg_g = epoch_g_loss / len(dataloader)
    d_losses.append(avg_d)
    g_losses.append(avg_g)

    scheduler_G.step()
    scheduler_D.step()

    print(f"Epoch {epoch:>3}/{args.epochs}  |  D_loss: {avg_d:.4f}  |  G_loss: {avg_g:.4f}  |  "
          f"LR: {scheduler_G.get_last_lr()[0]:.6f}")

    # ── Save generated samples ───────────────────────
    if epoch % args.save_every == 0 or epoch == 1 or epoch == args.epochs:
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise)
        save_image(samples, f"{SAVE_DIR}/epoch_{epoch:03d}.png",
                   nrow=8, normalize=True)
        print(f"  >> Saved sample grid → {SAVE_DIR}/epoch_{epoch:03d}.png")
        G.train()

    # ── Save checkpoints every 25 epochs ─────────────
    if epoch % 25 == 0:
        g_state = G.module.state_dict() if num_gpus > 1 else G.state_dict()
        d_state = D.module.state_dict() if num_gpus > 1 else D.state_dict()
        torch.save(g_state, f"{MODEL_DIR}/generator_ep{epoch}.pth")
        torch.save(d_state, f"{MODEL_DIR}/discriminator_ep{epoch}.pth")
        print(f"  >> Checkpoint saved at epoch {epoch}")

# ─────────────────────────────────────────
#  FINAL SAVE
# ─────────────────────────────────────────
g_final = G.module.state_dict() if num_gpus > 1 else G.state_dict()
d_final = D.module.state_dict() if num_gpus > 1 else D.state_dict()
torch.save(g_final, f"{MODEL_DIR}/generator.pth")
torch.save(d_final, f"{MODEL_DIR}/discriminator.pth")
print(f"\nFinal models saved → {MODEL_DIR}/")

# ─────────────────────────────────────────
#  LOSS CURVE PLOT
# ─────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(range(1, args.epochs + 1), g_losses, label="Generator Loss",     color="#a78bfa", linewidth=2)
plt.plot(range(1, args.epochs + 1), d_losses, label="Discriminator Loss",  color="#f472b6", linewidth=2)
plt.axvline(x=50, color="#666", linestyle="--", alpha=0.5, label="LR decay @ ep 50")
plt.axvline(x=75, color="#888", linestyle="--", alpha=0.5, label="LR decay @ ep 75")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"DCGAN Training Loss — {args.dataset.upper()} — {args.epochs} Epochs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{CURVE_DIR}/loss_curve.png", dpi=150)
print(f"Loss curve saved → {CURVE_DIR}/loss_curve.png")

print(f"\n{'='*55}")
print("  Training complete!")
print(f"  Generated images : {SAVE_DIR}/")
print(f"  Models           : {MODEL_DIR}/")
print(f"  Loss curves      : {CURVE_DIR}/")
print(f"{'='*55}\n")
