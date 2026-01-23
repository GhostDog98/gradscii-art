import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import os
import shutil

# Configuration
CHAR_WIDTH = 12
CHAR_HEIGHT = 24
GRID_WIDTH = 42
GRID_HEIGHT = 21
IMAGE_WIDTH = CHAR_WIDTH * GRID_WIDTH  # 504
IMAGE_HEIGHT = CHAR_HEIGHT * GRID_HEIGHT  # 504

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Metal) device")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")

# Printable ASCII characters (space to ~)
CHARS = ''.join(chr(i) for i in range(32, 127))
NUM_CHARS = len(CHARS)

print(f"Using {NUM_CHARS} characters: {CHARS[:20]}...")


def create_char_bitmaps():
    """Create a lookup table of character bitmaps using Monaco font."""
    print("Creating character bitmap LUT...")

    # Try to load Monaco font
    try:
        font = ImageFont.truetype("Monaco.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", 18)
        except:
            # Fallback to default monospace
            print("Warning: Monaco not found, using default font")
            font = ImageFont.load_default()

    # Render each character to a bitmap
    bitmaps = []
    for char in CHARS:
        # Create image for single character
        img = Image.new('L', (CHAR_WIDTH, CHAR_HEIGHT), 255)  # White background
        draw = ImageDraw.Draw(img)

        # Draw character in black
        draw.text((0, 0), char, font=font, fill=0)

        # Convert to numpy array and normalize to [0, 1]
        bitmap = np.array(img).astype(np.float32) / 255.0
        bitmaps.append(bitmap)

    # Stack into tensor: (NUM_CHARS, CHAR_HEIGHT, CHAR_WIDTH)
    bitmaps_tensor = torch.tensor(np.stack(bitmaps), dtype=torch.float32).to(DEVICE)
    print(f"Character bitmaps shape: {bitmaps_tensor.shape}")

    return bitmaps_tensor


def load_target_image(image_path):
    """Load and preprocess target image."""
    img = Image.open(image_path).convert('L')

    # Resize to match our grid dimensions
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

    # Convert to tensor and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).to(DEVICE)

    print(f"Target image shape: {img_tensor.shape}")
    return img_tensor


def render_ascii(logits, char_bitmaps):
    """
    Render ASCII art using soft character selection (vectorized).

    Args:
        logits: (GRID_HEIGHT, GRID_WIDTH, NUM_CHARS) - unnormalized scores
        char_bitmaps: (NUM_CHARS, CHAR_HEIGHT, CHAR_WIDTH) - character bitmaps

    Returns:
        rendered: (IMAGE_HEIGHT, IMAGE_WIDTH) - rendered image
    """
    # Apply softmax to get character weights
    weights = torch.softmax(logits, dim=-1)  # (GRID_HEIGHT, GRID_WIDTH, NUM_CHARS)

    # Vectorized rendering using einsum
    # weights: (GRID_HEIGHT, GRID_WIDTH, NUM_CHARS)
    # char_bitmaps: (NUM_CHARS, CHAR_HEIGHT, CHAR_WIDTH)
    # Result: (GRID_HEIGHT, GRID_WIDTH, CHAR_HEIGHT, CHAR_WIDTH)
    rendered_grid = torch.einsum('ijk,khw->ijhw', weights, char_bitmaps)

    # Reshape to final image by interleaving the grid
    # (GRID_HEIGHT, GRID_WIDTH, CHAR_HEIGHT, CHAR_WIDTH) -> (IMAGE_HEIGHT, IMAGE_WIDTH)
    rendered = rendered_grid.permute(0, 2, 1, 3).contiguous()  # (GRID_HEIGHT, CHAR_HEIGHT, GRID_WIDTH, CHAR_WIDTH)
    rendered = rendered.view(IMAGE_HEIGHT, IMAGE_WIDTH)

    return rendered


def train(target_image, char_bitmaps, num_iterations=1000, lr=0.01, save_interval=100, warmup_iterations=50):
    """Train ASCII art using gradient descent with cosine learning rate schedule."""

    # Clear and create steps directory
    if os.path.exists("steps"):
        shutil.rmtree("steps")
    os.makedirs("steps")

    # Initialize logits randomly
    logits = nn.Parameter(
        torch.randn(GRID_HEIGHT, GRID_WIDTH, NUM_CHARS, device=DEVICE) * 0.01
    )

    # Optimizer
    optimizer = optim.AdamW([logits], lr=lr)

    # Cosine annealing scheduler with warmup
    def get_lr_multiplier(iteration):
        if iteration < warmup_iterations:
            # Linear warmup
            return iteration / warmup_iterations
        else:
            # Cosine annealing
            progress = (iteration - warmup_iterations) / (num_iterations - warmup_iterations)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    # Loss function
    criterion = nn.MSELoss()

    print(f"\nTraining for {num_iterations} iterations with warmup={warmup_iterations}...")

    pbar = tqdm(range(num_iterations))
    for iteration in pbar:
        optimizer.zero_grad()

        # Render current ASCII art
        rendered = render_ascii(logits, char_bitmaps)

        # Compute loss
        loss = criterion(rendered, target_image)

        # Backprop
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'lr': f'{current_lr:.6f}'})

        # Save intermediate results
        if iteration % save_interval == 0 or iteration == num_iterations - 1:
            save_result(
                logits,
                char_bitmaps,
                output_path=f"steps/i_iter_{iteration:04d}.png",
                text_path=f"steps/t_iter_{iteration:04d}.txt"
            )

    return logits


def save_result(logits, char_bitmaps, output_path="output.png", text_path="output.txt"):
    """Save the final ASCII art as image and text."""
    # Get discrete character selection for text file
    char_indices = torch.argmax(logits, dim=-1)  # (GRID_HEIGHT, GRID_WIDTH) - keep on device

    # Save as text file
    char_indices_cpu = char_indices.cpu()
    with open(text_path, 'w') as f:
        for i in range(GRID_HEIGHT):
            line = ''.join(CHARS[char_indices_cpu[i, j].item()] for j in range(GRID_WIDTH))
            f.write(line + '\n')

    # Render with soft selection (logits will be softmaxed inside render_ascii)
    rendered = render_ascii(logits, char_bitmaps)

    # Save as image
    img_array = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save(output_path)


def test_char_bitmaps():
    """Test character bitmap creation and save visualizations."""
    print("Testing character bitmap creation...")
    char_bitmaps = create_char_bitmaps()

    # Save a visualization of all characters
    num_chars = char_bitmaps.shape[0]
    chars_per_row = 16
    num_rows = (num_chars + chars_per_row - 1) // chars_per_row

    # Create a grid showing all characters
    grid_height = num_rows * CHAR_HEIGHT
    grid_width = chars_per_row * CHAR_WIDTH
    grid = np.ones((grid_height, grid_width), dtype=np.float32)  # White background

    for idx, char in enumerate(CHARS):
        row = idx // chars_per_row
        col = idx % chars_per_row

        y_start = row * CHAR_HEIGHT
        y_end = (row + 1) * CHAR_HEIGHT
        x_start = col * CHAR_WIDTH
        x_end = (col + 1) * CHAR_WIDTH

        grid[y_start:y_end, x_start:x_end] = char_bitmaps[idx].cpu().numpy()

    # Save grid
    grid_img = Image.fromarray((grid * 255).astype(np.uint8), mode='L')
    grid_img.save("char_grid.png")
    print(f"Saved character grid to char_grid.png")
    print(f"Grid dimensions: {grid_width}x{grid_height}")
    print(f"Character dimensions: {CHAR_WIDTH}x{CHAR_HEIGHT}")

    # Also save individual character examples
    test_chars = "AaBb@#01 "
    for char in test_chars:
        if char in CHARS:
            idx = CHARS.index(char)
            bitmap = char_bitmaps[idx].cpu().numpy()
            char_img = Image.fromarray((bitmap * 255).astype(np.uint8), mode='L')
            safe_name = char if char != ' ' else 'space'
            char_img.save(f"char_{safe_name}.png")
            print(f"Saved char_{safe_name}.png - shape: {bitmap.shape}, min: {bitmap.min():.2f}, max: {bitmap.max():.2f}")


if __name__ == "__main__":
    import sys

    # Test mode
    # test_char_bitmaps()
    # exit()

    # Training mode (disabled for now)
    if len(sys.argv) < 2:
        print("Usage: python train.py <input_image>")
        sys.exit(1)

    input_image_path = sys.argv[1]

    # Create character bitmaps
    char_bitmaps = create_char_bitmaps()

    # Load target image
    target_image = load_target_image(input_image_path)

    # Train
    logits = train(target_image, char_bitmaps, num_iterations=10000, lr=0.01, warmup_iterations=500)

    # Save final results
    save_result(logits, char_bitmaps, output_path="output.png", text_path="output.txt")

    print("\nDone!")
