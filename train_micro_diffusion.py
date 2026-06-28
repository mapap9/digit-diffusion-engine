import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# =====================================================================
# 1. TIMESTEP EMBEDDING MODULE
# =====================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    """Maps a 1D timestep scalar to a dense sinusoidal vector space (Transformer style)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# =====================================================================
# 2. MICRO CLASS-CONDITIONED U-NET ARCHITECTURE
# =====================================================================
class ResidualBlock(nn.Module):
    """Classic ResNet block to allow deep gradient propagation without degradation"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.mlp = nn.Linear(time_emb_dim, out_channels)

        # Shortcut connection if channel dimensions don't match
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, ctx):
        h = F.gelu(self.conv1(x))
        # Inject conditioning directly into the residual feature map
        h = h + self.mlp(ctx).view(ctx.size(0), -1, 1, 1)
        h = self.conv2(h)
        return F.gelu(h + self.shortcut(x))

class StandardUNet(nn.Module):
    """Production-grade multi-stage U-Net with mathematically realigned channel maps"""
    def __init__(self, num_classes=10, time_emb_dim=64):
        super().__init__()

        # Condition Embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        # ─── ENCODER PATH (Downsampling) ───
        self.init_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(64, 128, time_emb_dim)  # Outputs 128 channels
        self.pool1 = nn.MaxPool2d(2)                      # 28x28 -> 14x14

        self.down2 = ResidualBlock(128, 256, time_emb_dim) # Outputs 256 channels
        self.pool2 = nn.MaxPool2d(2)                      # 14x14 -> 7x7

        # ─── BOTTLENECK LAYER ───
        self.bottleneck = ResidualBlock(256, 256, time_emb_dim)

        # ─── DECODER PATH (Upsampling) ───
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 7x7 -> 14x14 (Outputs 128)
        # FIX: Accepts 128 (from up1) + 256 (from h3 skip connection) = 384 channels
        self.up_block1 = ResidualBlock(384, 128, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 14x14 -> 28x28 (Outputs 64)
        # FIX: Accepts 64 (from up2) + 128 (from h2 skip connection) = 192 channels
        self.up_block2 = ResidualBlock(192, 64, time_emb_dim)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, t, c):
        ctx = self.time_mlp(t) + self.label_emb(c)

        # Encoder Path
        h1 = self.init_conv(x)
        h2 = self.down1(h1, ctx) # 128 channels, 28x28
        p2 = self.pool1(h2)      # 128 channels, 14x14

        h3 = self.down2(p2, ctx) # 256 channels, 14x14
        p3 = self.pool2(h3)      # 256 channels, 7x7

        # Bottleneck
        b = self.bottleneck(p3, ctx)

        # Decoder Path with realigned concat bounds
        u1 = self.up1(b)         # 128 channels, 14x14
        u1 = self.up_block1(torch.cat([u1, h3], dim=1), ctx) # 128 + 256 = 384 channels -> outputs 128 channels

        u2 = self.up2(u1)        # 64 channels, 28x28
        u2 = self.up_block2(torch.cat([u2, h2], dim=1), ctx) # 64 + 128 = 192 channels -> outputs 64 channels

        return self.final_conv(u2)

# =====================================================================
# 3. CONSTRUCT THE VERIFICATION SAMPLES
# =====================================================================
@torch.no_grad()
def sample_from_model(model, scheduler, device, epoch):
    """Runs a full reverse DDIM/DDPM style sampler to generate digits from pure noise"""
    model.eval()
    print(f"\n🎨 Sampling generation validation grid for Epoch {epoch}...")
    
    # Target generation: Create one grid row containing labels 0 through 9
    labels = torch.arange(1, 11, device=device) % 10
    n_samples = labels.size(0)
    
    # Initialize latents with pure random Gaussian noise
    xt = torch.randn(n_samples, 1, 28, 28, device=device)
    
    # Trace backwards through the noise timeline (from T=199 down to 0)
    for t_idx in reversed(range(scheduler["T"])):
        t_batch = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
        
        # Predict the noise vector using our U-Net model
        predicted_noise = model(xt, t_batch, labels)
        
        # Fetch scheduler parameters for the current step
        alpha_t = scheduler["alphas"][t_idx]
        alpha_bar_t = scheduler["alpha_bars"][t_idx]
        beta_t = scheduler["betas"][t_idx]
        
        # Implement the classical DDPM reverse variance derivation equations
        if t_idx > 0:
            noise = torch.randn_like(xt)
            sigma_t = torch.sqrt(beta_t)
        else:
            noise = 0
            sigma_t = 0
            
        # Reconstruct the denoised latent step matrix
        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        xt = (1 / torch.sqrt(alpha_t)) * (xt - coef * predicted_noise) + sigma_t * noise
        xt = torch.clamp(xt, -1.0, 1.0)

    # Save validation output cleanly as a combined tensor file to keep things simple
    os.makedirs("outputs", exist_ok=True)
    torch.save(xt.cpu(), f"outputs/epoch_{epoch}_samples.pt")
    print(f"💾 Samples stored successfully to: 'outputs/epoch_{epoch}_samples.pt'")

# =====================================================================
# 4. MAIN HIGH-VELOCITY TRAINING ENGINE
# =====================================================================
def run_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Launching Micro-Diffusion Engine on device target: {device}")

    # Define Hyperparameters
    T = 200
    epochs = 50
    batch_size = 128
    lr = 1e-3

    # Define Linear Variance Scheduler Boundaries (DDPM Style)
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    scheduler = {"T": T, "betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}

    # Load Dataset via Torchvision Data Pipelines
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Scale pixels from [0, 1] to [-1, 1] bounds
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

    # Initialize Engine Components
    model = StandardUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Core Training Loop Execution Blocks
    for epoch in range(1, epochs + 1):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # 1. Sample random diffusion timelines uniformly for the batch
            t = torch.randint(0, T, (batch_size,), device=device, dtype=torch.long)
            
            # 2. Generate pure random noise to mix with our clean target images
            noise = torch.randn_like(images)
            
            # 3. Apply the forward noise equation to derive the noisy latents (xt)
            alpha_bar_t = alpha_bars[t].view(batch_size, 1, 1, 1)
            xt = torch.sqrt(alpha_bar_t) * images + torch.sqrt(1.0 - alpha_bar_t) * noise
            
            # 4. Run the forward pass to predict the added noise vector
            pred_noise = model(xt, t, labels)
            
            # 5. Compute MSE loss against the true added noise vector
            loss = criterion(pred_noise, noise)
            
            # 6. Run backpropagation to update our model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())

        # Sample from the model at the end of every epoch to track visual progress
        sample_from_model(model, scheduler, device, epoch)

    # Save finalized model checkpoints out to disk
    torch.save(model.state_dict(), "micro_diffusion_mnist.pt")
    print("\n🎉 SUCCESS: Micro-Diffusion training run completed. Weights archived successfully.")

if __name__ == "__main__":
    run_training()
