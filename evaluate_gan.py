import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy import linalg
import numpy as np

LATENT_DIM = 64
IMG_SIZE = 28

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, IMG_SIZE * IMG_SIZE), nn.Tanh()
        )
    def forward(self, z):
        return self.model(z).view(-1, 1, IMG_SIZE, IMG_SIZE)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(IMG_SIZE * IMG_SIZE, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

# Load models
G = Generator()
D = Discriminator()
G.load_state_dict(torch.load("gan_model/generator.pth"))
D.load_state_dict(torch.load("gan_model/discriminator.pth"))
G.eval()
D.eval()

print("=" * 55)
print("           GAN EVALUATION RESULTS")
print("=" * 55)

# 1. Generate samples and save
z = torch.randn(64, LATENT_DIM)
with torch.no_grad():
    generated = G(z)
save_image(generated, "generated_images/final_evaluation.png", nrow=8, normalize=True)
print(f"\n[1] Generated 64 sample images → generated_images/final_evaluation.png")

# 2. Discriminator score on generated images
with torch.no_grad():
    d_scores = D(generated).squeeze().numpy()
avg_score = d_scores.mean()
print(f"\n[2] Discriminator Score on Generated Images:")
print(f"    Average: {avg_score:.4f} (closer to 0.5 = better trained GAN)")
print(f"    Min: {d_scores.min():.4f} | Max: {d_scores.max():.4f}")

# 3. Simple FID-like score (pixel distribution similarity)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
real_data = datasets.MNIST("./data", train=False, download=True, transform=transform)
real_loader = torch.utils.data.DataLoader(real_data, batch_size=500, shuffle=True)
real_imgs, _ = next(iter(real_loader))

with torch.no_grad():
    z = torch.randn(500, LATENT_DIM)
    fake_imgs = G(z)

real_flat = real_imgs.view(500, -1).numpy()
fake_flat = fake_imgs.view(500, -1).numpy()

real_mean, real_std = real_flat.mean(axis=0), real_flat.std(axis=0)
fake_mean, fake_std = fake_flat.mean(axis=0), fake_flat.std(axis=0)

mean_diff = np.linalg.norm(real_mean - fake_mean)
std_diff  = np.linalg.norm(real_std  - fake_std)
approx_fid = mean_diff + std_diff

print(f"\n[3] Approximate Distribution Distance (simplified FID):")
print(f"    Score: {approx_fid:.4f} (lower = generated images closer to real)")

# 4. Pixel stats comparison
print(f"\n[4] Pixel Statistics Comparison:")
print(f"    Real  → Mean: {real_flat.mean():.4f} | Std: {real_flat.std():.4f}")
print(f"    Fake  → Mean: {fake_flat.mean():.4f} | Std: {fake_flat.std():.4f}")

print("\n" + "=" * 55)
print("Evaluation complete! Show 'final_evaluation.png' to faculty.")
print("=" * 55)