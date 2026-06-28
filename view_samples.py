import torch
import matplotlib.pyplot as plt
import os

def plot_epoch_samples(epoch_num=10):
    filepath = f"outputs/epoch_{epoch_num}_samples.pt"
    
    if not os.path.exists(filepath):
        print(f"❌ Error: Could not find validation file '{filepath}'. Check your outputs/ folder.")
        return

    print(f"🎨 Loading generated latents from Epoch {epoch_num}...")
    # Load the raw tensor block [10, 1, 28, 28]
    samples = torch.load(filepath, map_location="cpu" if not torch.cuda.is_available() else None)
    
    # Reverse the training normalization: pixel = (pixel * 0.5) + 0.5
    # This brings the math back from [-1, 1] bounds to standard [0, 1] image scales
    samples = (samples * 0.5) + 0.5
    samples = torch.clamp(samples, 0.0, 1.0)
    
    # Set up the matplotlib grid canvas
    fig, axes = plt.subplots(1, 10, figsize=(15, 2.5))
    fig.suptitle(f"Micro-Diffusion Generation (Epoch {epoch_num})", fontsize=14, fontweight='bold', y=1.05)

    for i in range(10):
        # Extract the 2D pixel grid for each generated digit
        img = samples[i].squeeze().numpy()
        
        # Calculate matching label mapping (labels generated were 1 to 10 % 10 -> 1,2,3,4,5,6,7,8,9,0)
        target_digit = (i + 1) % 10
        
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {target_digit}", fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    output_image_path = f"outputs/generation_grid_epoch_{epoch_num}.png"
    plt.savefig(output_image_path, bbox_inches="tight", dpi=150)
    print(f"🎉 SUCCESS: Visualization grid compiled and saved to: '{output_image_path}'")
    plt.show()

if __name__ == "__main__":
    # Change the epoch integer here if you want to inspect earlier snapshots (e.g., epoch 1 vs epoch 10)
    plot_epoch_samples(epoch_num=50)
