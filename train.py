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
ROW_GAP = 6  # Gap between rows (receipt printer spacing. Use 0 for discord, 6 for receipt printer)
IMAGE_WIDTH = CHAR_WIDTH * GRID_WIDTH  # 504
IMAGE_HEIGHT = CHAR_HEIGHT * GRID_HEIGHT + ROW_GAP * (GRID_HEIGHT - 1)  # 504 + 120 = 624

# Character encoding (cp437 for receipt printers, ascii for standard text)
ENCODING = 'cp437'

# Ban certain characters (block characters that feel like cheating)
BANNED_CHARS = ['`', '\\'] # ['░', '▒', '▓', '█', '▄', '▌', '▐', '▀', '■']

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

# Character set based on encoding
if ENCODING == 'cp437':
    CHARS = ''.join(bytes([i]).decode('cp437') for i in range(32, 256))
else:
    # Standard 7-bit ASCII
    CHARS = ''.join(chr(i) for i in range(32, 127))

CHARS = ''.join(c for c in CHARS if c not in BANNED_CHARS)

NUM_CHARS = len(CHARS)

print(f"Using {NUM_CHARS} characters ({ENCODING}): {CHARS}")


def create_char_bitmaps():
    """Create a lookup table of character bitmaps with font fallback."""
    print("Creating character bitmap LUT...")

    # Try to load printer font (bitArray-A2.ttf) for 7-bit ASCII
    printer_font, printer_y_offset = None, None
    try:
        printer_font, printer_y_offset = ImageFont.truetype("./fonts/bitArray-A2.ttf", 24), 4
        # printer_font, printer_y_offset = ImageFont.truetype("./fonts/gg mono.ttf", 18), 0 # for discord
        print("Loaded printer font: bitArray-A2.ttf (24pt)")
    except:
        print("Printer font not found, using fallback for all characters")

    # Load fallback font (Menlo for extended ASCII)
    fallback_font = None
    fallback_paths = [
        # "./fonts/SourceCodePro-VariableFont_wght.ttf", # for discord
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
    ]
    for path in fallback_paths:
        try:
            fallback_font = ImageFont.truetype(path, 18)
            print(f"Loaded fallback font: {path} (18pt)")
            break
        except:
            continue

    if fallback_font is None:
        print("Warning: No fallback font found, using default")
        fallback_font = ImageFont.load_default()

    # Render each character to a bitmap
    bitmaps = []
    printer_count = 0
    fallback_count = 0

    for idx, char in enumerate(CHARS):
        # Use printer font for 7-bit ASCII, fallback for extended
        char_code = ord(char)

        if printer_font is not None and char_code < 127:
            # Use printer font with Y offset 4
            font = printer_font
            y_offset = printer_y_offset
            printer_count += 1
        else:
            # Use fallback font with Y offset 0
            font = fallback_font
            y_offset = 0
            fallback_count += 1

        # Create image for single character
        img = Image.new('L', (CHAR_WIDTH, CHAR_HEIGHT), 255)  # White background
        draw = ImageDraw.Draw(img)
        draw.text((0, y_offset), char, font=font, fill=0)

        # Convert to numpy array and normalize to [0, 1]
        bitmap = np.array(img).astype(np.float32) / 255.0
        bitmaps.append(bitmap)

    # Stack into tensor: (NUM_CHARS, CHAR_HEIGHT, CHAR_WIDTH)
    bitmaps_tensor = torch.tensor(np.stack(bitmaps), dtype=torch.float32).to(DEVICE)
    print(f"Character bitmaps shape: {bitmaps_tensor.shape}")
    print(f"Using printer font: {printer_count} chars, fallback font: {fallback_count} chars")

    return bitmaps_tensor


def load_target_image(image_path):
    """Load and preprocess target image."""
    img = Image.open(image_path).convert('L')

    # Resize to content dimensions (without gap space)
    content_height = CHAR_HEIGHT * GRID_HEIGHT  # 504
    img = img.resize((IMAGE_WIDTH, content_height), Image.LANCZOS)

    # Convert to array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0

    # Pad to match IMAGE_HEIGHT if we have row gaps (split padding top and bottom)
    if IMAGE_HEIGHT > content_height:
        padding = IMAGE_HEIGHT - content_height
        pad_top = padding // 2
        pad_bottom = padding - pad_top
        img_array = np.pad(img_array, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=1.0)  # White padding

    # Convert to tensor
    img_tensor = torch.tensor(img_array, dtype=torch.float32, device=DEVICE)

    print(f"Target image shape: {img_tensor.shape}")
    return img_tensor


def render_ascii(logits, char_bitmaps, temperature=1.0, use_gumbel=False):
    """
    Render ASCII art using soft character selection (vectorized).

    Args:
        logits: (GRID_HEIGHT, GRID_WIDTH, NUM_CHARS) - unnormalized scores
        char_bitmaps: (NUM_CHARS, CHAR_HEIGHT, CHAR_WIDTH) - character bitmaps
        temperature: Temperature for softmax (lower = more discrete)
        use_gumbel: Whether to add Gumbel noise

    Returns:
        rendered: (IMAGE_HEIGHT, IMAGE_WIDTH) - rendered image with row gaps
    """
    # Apply Gumbel noise if requested
    if use_gumbel and logits.requires_grad:  # Only during training
        # Sample Gumbel noise: g = -log(-log(u)) where u ~ Uniform(0,1)
        # Ensure random sampling happens on device (MPS/CUDA)
        u = torch.rand_like(logits, device=logits.device)
        gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        logits_with_noise = logits + gumbel_noise
    else:
        logits_with_noise = logits

    # Apply temperature-scaled softmax to get character weights
    weights = torch.softmax(logits_with_noise / temperature, dim=-1)  # (GRID_HEIGHT, GRID_WIDTH, NUM_CHARS)

    # Vectorized rendering using einsum
    # weights: (GRID_HEIGHT, GRID_WIDTH, NUM_CHARS)
    # char_bitmaps: (NUM_CHARS, CHAR_HEIGHT, CHAR_WIDTH)
    # Result: (GRID_HEIGHT, GRID_WIDTH, CHAR_HEIGHT, CHAR_WIDTH)
    rendered_grid = torch.einsum('ijk,khw->ijhw', weights, char_bitmaps)

    # Reshape to image with row gaps
    # (GRID_HEIGHT, GRID_WIDTH, CHAR_HEIGHT, CHAR_WIDTH) -> (GRID_HEIGHT, CHAR_HEIGHT, GRID_WIDTH, CHAR_WIDTH)
    rendered_grid = rendered_grid.permute(0, 2, 1, 3).contiguous()
    # -> (GRID_HEIGHT, CHAR_HEIGHT, IMAGE_WIDTH)
    rendered_grid = rendered_grid.view(GRID_HEIGHT, CHAR_HEIGHT, IMAGE_WIDTH)

    if ROW_GAP > 0:
        # Create output with gaps (white = 1.0)
        rendered = torch.ones((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32, device=DEVICE)

        # Place each row with gaps
        for i in range(GRID_HEIGHT):
            y_start = i * (CHAR_HEIGHT + ROW_GAP)
            y_end = y_start + CHAR_HEIGHT
            rendered[y_start:y_end, :] = rendered_grid[i]
    else:
        # No gaps, just reshape
        rendered = rendered_grid.view(IMAGE_HEIGHT, IMAGE_WIDTH)

    return rendered


def train(target_image, char_bitmaps, num_iterations=1000, lr=0.01, save_interval=100, warmup_iterations=50, diversity_weight=0.01,
          use_gumbel=True, temp_start=1.0, temp_end=0.01, protect_whitespace=True):
    """Train ASCII art using gradient descent with cosine learning rate schedule, diversity loss, and Gumbel-softmax."""

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

    print(f"\nTraining for {num_iterations} iterations with warmup={warmup_iterations}")
    print(f"Gumbel-softmax: {use_gumbel}, Temperature: {temp_start} -> {temp_end}")

    pbar = tqdm(range(num_iterations))
    for iteration in pbar:
        optimizer.zero_grad()

        # Compute current temperature based on learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if iteration < warmup_iterations:
            temperature = temp_start
        else:
            lr_ratio = current_lr / lr  # Ratio of current LR to initial LR
            # lr_ratio_curved = ((1 - lr_ratio) ** 2) # Square the LR ratio so we stay high temp for longer
            lr_ratio_curved = ((1 - lr_ratio))
            temperature = temp_start + (temp_end - temp_start) * lr_ratio_curved

        # Render current ASCII art with Gumbel-softmax
        rendered = render_ascii(logits, char_bitmaps, temperature=temperature, use_gumbel=use_gumbel)

        # Compute reconstruction loss
        recon_loss = criterion(rendered, target_image)

        if diversity_weight != 0.0:
            # Compute diversity loss (entropy of character usage, excluding whitespace)
            weights = torch.softmax(logits, dim=-1)  # (GRID_HEIGHT, GRID_WIDTH, NUM_CHARS)
            char_usage = weights.mean(dim=[0, 1])  # (NUM_CHARS,) - average usage of each character

            # Exclude whitespace from diversity calculation
            space_idx = CHARS.index(' ') if ' ' in CHARS else -1
            if space_idx >= 0 and protect_whitespace:
                # Mask out space character
                mask = torch.ones(NUM_CHARS, device=DEVICE)
                mask[space_idx] = 0
                char_usage_masked = char_usage * mask
                # Renormalize (only among non-space characters)
                char_usage_masked = char_usage_masked / (char_usage_masked.sum() + 1e-10)
            else:
                char_usage_masked = char_usage

            # Entropy: -sum(p * log(p)) - higher entropy = more diverse
            entropy = -(char_usage_masked * torch.log(char_usage_masked + 1e-10)).sum()
            # We want to maximize entropy, so subtract it (or add negative)
            diversity_loss = -entropy
        else:
            diversity_loss = torch.tensor(0.0).to(DEVICE)

        # Total loss
        loss = recon_loss + diversity_weight * diversity_loss

        # Backprop
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update progress bar
        pbar.set_postfix({
            'recon': f'{recon_loss.item():.4f}',
            'div': f'{diversity_loss.item():.4f}',
            'lr': f'{current_lr:.4f}',
            'temp': f'{temperature:.4f}'
        })

        # Save intermediate results
        if iteration % save_interval == 0 or iteration == num_iterations - 1:
            save_result(
                logits,
                char_bitmaps,
                output_path=f"steps/i_iter_{iteration:04d}.png",
                text_path=f"steps/t_iter_{iteration:04d}.txt",
                temperature=temperature
            )

    return logits


def save_result(logits, char_bitmaps, output_path="output.png", text_path="output.txt", utf8_path="", temperature=0.1):
    """Save the final ASCII art as image and text."""
    # Get discrete character selection for text file
    char_indices = torch.argmax(logits, dim=-1)  # (GRID_HEIGHT, GRID_WIDTH) - keep on device

    # Save as text file with specified encoding
    char_indices_cpu = char_indices.cpu()
    with open(text_path, 'w', encoding=ENCODING) as f:
        for i in range(GRID_HEIGHT):
            line = ''.join(CHARS[char_indices_cpu[i, j].item()] for j in range(GRID_WIDTH))
            f.write(line + '\n')

    if utf8_path:
        with open(text_path, 'r', encoding=ENCODING) as f1:
            with open(utf8_path, 'w', encoding='utf-8') as f2:
                f2.write(f1.read())

    # Render with soft selection (no Gumbel noise for deterministic output)
    rendered = render_ascii(logits, char_bitmaps, temperature=temperature, use_gumbel=False)

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
    logits = train(target_image, char_bitmaps, num_iterations=10000, lr=0.01, warmup_iterations=1000, diversity_weight=0.01,
                   use_gumbel=True, temp_start=1.0, temp_end=0.1, protect_whitespace=False)

    # Save final results (use low temperature for sharp output)
    save_result(logits, char_bitmaps, output_path="output.png", text_path="output.txt", utf8_path="output.utf8.txt", temperature=0.01)

    print("\nDone!")
