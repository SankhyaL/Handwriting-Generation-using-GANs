"""
app.py — Flask demo for DCGAN Handwriting Generator
Matches train_gan.py: ConvTranspose2d Generator, latent_dim=100, 64x64 images
"""

from flask import Flask, render_template_string, request, jsonify
import torch
import torch.nn as nn
from torchvision.utils import save_image
import io, base64

# ─────────────────────────────────────────
#  CONFIG — must match train_gan.py
# ─────────────────────────────────────────
LATENT_DIM = 100
IMG_SIZE   = 64
CHANNELS   = 1

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
#  LOAD MODEL
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(LATENT_DIM).to(device)
G.load_state_dict(torch.load("gan_model/generator.pth", map_location=device, weights_only=True))
G.eval()
print(f"Generator loaded on {device}")

# ─────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Handwriting GAN Demo</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f0f13; color: #e8e8f0; font-family: 'Inter', sans-serif;
           min-height: 100vh; display: flex; flex-direction: column;
           align-items: center; padding: 40px 20px; }
    h1 { font-family: 'Space Mono', monospace; font-size: 1.6rem;
         color: #a78bfa; letter-spacing: -0.5px; margin-bottom: 6px; }
    .subtitle { color: #666; font-size: 0.85rem; margin-bottom: 40px; }
    .card { background: #1a1a24; border: 1px solid #2a2a38; border-radius: 16px;
            padding: 32px; width: 100%; max-width: 600px; }
    .label { font-size: 0.78rem; color: #888; text-transform: uppercase;
             letter-spacing: 1px; margin-bottom: 10px; }
    input[type=range] { width: 100%; accent-color: #a78bfa; margin-bottom: 6px; }
    .range-val { font-family: 'Space Mono', monospace; color: #a78bfa; font-size: 0.9rem; }
    button { width: 100%; margin-top: 24px; padding: 14px; background: #a78bfa;
             color: #0f0f13; border: none; border-radius: 10px; font-size: 1rem;
             font-weight: 600; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #c4b5fd; }
    button:disabled { background: #444; color: #888; cursor: not-allowed; }
    #output { margin-top: 28px; text-align: center; }
    #output img { width: 100%; max-width: 512px; border-radius: 10px;
                  border: 1px solid #2a2a38; image-rendering: pixelated; }
    .loading { color: #a78bfa; font-family: 'Space Mono', monospace;
               font-size: 0.9rem; margin-top: 20px; }
    .metrics { margin-top: 16px; display: flex; gap: 12px;
               justify-content: center; flex-wrap: wrap; }
    .metric { background: #12121a; border: 1px solid #2a2a38; border-radius: 8px;
              padding: 10px 18px; text-align: center; }
    .metric .val { font-family: 'Space Mono', monospace; color: #a78bfa; font-size: 1.1rem; }
    .metric .key { font-size: 0.72rem; color: #666; margin-top: 3px; }
  </style>
</head>
<body>
  <h1>✍️ Handwriting DCGAN</h1>
  <p class="subtitle">Deep Convolutional GAN · Trained on MNIST · 64×64 output</p>
  <div class="card">
    <div class="label">Number of digits to generate</div>
    <input type="range" min="4" max="64" value="16" step="4" id="count"
           oninput="document.getElementById('cv').innerText=this.value">
    <div class="range-val"><span id="cv">16</span> digits</div>
    <button id="btn" onclick="generate()">Generate Handwriting</button>
    <div id="output"></div>
  </div>

  <script>
    async function generate() {
      const btn   = document.getElementById('btn');
      const count = document.getElementById('count').value;
      btn.disabled = true;
      btn.innerText = 'Generating...';
      document.getElementById('output').innerHTML =
        '<p class="loading">Running DCGAN inference...</p>';

      const res  = await fetch('/generate?n=' + count);
      const data = await res.json();

      document.getElementById('output').innerHTML = `
        <img src="data:image/png;base64,${data.image}" />
        <div class="metrics">
          <div class="metric">
            <div class="val">${data.avg_d_score}</div>
            <div class="key">Avg D-Score</div>
          </div>
          <div class="metric">
            <div class="val">${data.count}</div>
            <div class="key">Images</div>
          </div>
          <div class="metric">
            <div class="val">${data.latent_dim}</div>
            <div class="key">Latent Dim</div>
          </div>
          <div class="metric">
            <div class="val">${data.img_size}px</div>
            <div class="key">Resolution</div>
          </div>
        </div>`;

      btn.disabled = false;
      btn.innerText = 'Generate Again';
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate")
def generate():
    n = int(request.args.get("n", 16))
    n = min(max(n, 4), 64)

    z = torch.randn(n, LATENT_DIM, 1, 1, device=device)
    with torch.no_grad():
        imgs = G(z)

    nrow = int(n ** 0.5)
    buf  = io.BytesIO()
    save_image(imgs, buf, nrow=nrow, normalize=True, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    return jsonify({
        "image":       img_b64,
        "count":       n,
        "latent_dim":  LATENT_DIM,
        "img_size":    IMG_SIZE,
        "avg_d_score": "N/A",   # D not loaded in app (keeps it lightweight)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001, use_reloader=False)
