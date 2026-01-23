import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import os
import shutil
import argparse

# Default Configuration (can be overridden via command line arguments)
CHAR_WIDTH = 12
CHAR_HEIGHT = 24
GRID_WIDTH = 42
GRID_HEIGHT = 21
ROW_GAP = 6
IMAGE_WIDTH = CHAR_WIDTH * GRID_WIDTH
IMAGE_HEIGHT = CHAR_HEIGHT * GRID_HEIGHT + ROW_GAP * (GRID_HEIGHT - 1)

ENCODING = 'cp437'
BANNED_CHARS = ['`', '\\']

PRINTER_FONT = "./fonts/bitArray-A2.ttf"
PRINTER_FONT_SIZE = 24
PRINTER_Y_OFFSET = 4
FALLBACK_FONTS = ["/System/Library/Fonts/Supplemental/Menlo.ttc", "/System/Library/Fonts/Monaco.dfont"]
FALLBACK_FONT_SIZE = 18

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

# Character set (initialized in main after parsing args)
CHARS = ''
NUM_CHARS = 0


def create_char_bitmaps():
    """Create a lookup table of character bitmaps with font fallback."""
    print("Creating character bitmap LUT...")

    # Try to load printer font for 7-bit ASCII
    printer_font, printer_y_offset = None, None
    try:
        printer_font = ImageFont.truetype(PRINTER_FONT, PRINTER_FONT_SIZE)
        printer_y_offset = PRINTER_Y_OFFSET
        print(f"Loaded printer font: {PRINTER_FONT} ({PRINTER_FONT_SIZE}pt)")
    except:
        print("Printer font not found, using fallback for all characters")

    # Load fallback font for extended ASCII
    fallback_font = None
    for path in FALLBACK_FONTS:
        try:
            fallback_font = ImageFont.truetype(path, FALLBACK_FONT_SIZE)
            print(f"Loaded fallback font: {path} ({FALLBACK_FONT_SIZE}pt)")
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ASCII art using gradient descent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  epson    Receipt printer (CP437, bitArray-A2 font, 6px row gap)
  discord  Discord display (ASCII, gg mono font, no row gap)

Examples:
  python train.py image.jpg
  python train.py image.jpg --preset discord
  python train.py image.jpg --iterations 20000 --diversity-weight 0.05
        """
    )

    # Required arguments
    parser.add_argument('input_image', help='Input image path')

    # Preset configuration
    parser.add_argument('--preset', choices=['epson', 'discord'],
                       help='Use preset configuration (epson=receipt printer, discord=Discord)')

    # Grid configuration
    parser.add_argument('--char-width', type=int, default=12,
                       help='Character width in pixels (default: 12)')
    parser.add_argument('--char-height', type=int, default=24,
                       help='Character height in pixels (default: 24)')
    parser.add_argument('--grid-width', type=int, default=42,
                       help='Number of characters per row (default: 42)')
    parser.add_argument('--grid-height', type=int, default=21,
                       help='Number of character rows (default: 21)')
    parser.add_argument('--row-gap', type=int, default=6,
                       help='Gap between rows in pixels (default: 6 for receipt printer, 0 for Discord)')

    # Character set configuration
    parser.add_argument('--encoding', choices=['cp437', 'ascii'], default='cp437',
                       help='Character encoding (cp437 for receipt printers, ascii for standard text)')
    parser.add_argument('--ban-chars', type=str, default='',
                       help='Characters to ban from charset (default: "")')
    parser.add_argument('--ban-blocks', action='store_true',
                       help='Ban block characters: ░▒▓█▄▌▐▀■')

    # Font configuration
    parser.add_argument('--printer-font', type=str, default='./fonts/bitArray-A2.ttf',
                       help='Path to printer font for 7-bit ASCII (default: bitArray-A2.ttf)')
    parser.add_argument('--printer-font-size', type=int, default=24,
                       help='Printer font size in points (default: 24)')
    parser.add_argument('--printer-y-offset', type=int, default=4,
                       help='Y offset for printer font rendering (default: 4)')
    parser.add_argument('--fallback-font', type=str,
                       default='/System/Library/Fonts/Supplemental/Menlo.ttc',
                       help='Path to fallback font for extended ASCII')
    parser.add_argument('--fallback-font-size', type=int, default=18,
                       help='Fallback font size in points (default: 18)')

    # Training hyperparameters
    parser.add_argument('--iterations', type=int, default=10000,
                       help='Number of training iterations (default: 10000)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--warmup', type=int, default=1000,
                       help='Number of warmup iterations for learning rate schedule (default: 1000)')
    parser.add_argument('--diversity-weight', type=float, default=0.01,
                       help='Weight for diversity loss encouraging varied character usage (default: 0.01, set to 0 to disable)')
    parser.add_argument('--penalize-whitespace', action='store_true',
                       help='Include whitespace in diversity penalty (default: whitespace is protected)')

    # Gumbel-softmax parameters
    parser.add_argument('--no-gumbel', action='store_true',
                       help='Disable Gumbel-softmax (use plain softmax)')
    parser.add_argument('--temp-start', type=float, default=1.0,
                       help='Starting temperature for Gumbel-softmax (default: 1.0, higher = more exploration)')
    parser.add_argument('--temp-end', type=float, default=0.1,
                       help='Ending temperature for Gumbel-softmax (default: 0.1, lower = more discrete)')
    parser.add_argument('--save-temp', type=float, default=0.01,
                       help='Temperature for final output rendering (default: 0.01)')

    # Output configuration
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save intermediate results every N iterations (default: 100)')
    parser.add_argument('--output', type=str, default='output.png',
                       help='Output image path (default: output.png)')
    parser.add_argument('--output-text', type=str, default='output.txt',
                       help='Output text file path (default: output.txt)')
    parser.add_argument('--output-utf8', type=str, default='output.utf8.txt',
                       help='Output UTF-8 text file path (default: output.utf8.txt)')

    # Test mode
    parser.add_argument('--test-chars', action='store_true',
                       help='Test character rendering and exit')

    args = parser.parse_args()

    # Apply presets
    if args.preset == 'epson':
        args.encoding = 'cp437'
        args.printer_font = './fonts/bitArray-A2.ttf'
        args.printer_font_size = 24
        args.printer_y_offset = 4
        args.row_gap = 6
        args.ban_chars = ''
    elif args.preset == 'discord':
        args.encoding = 'cp437'
        args.printer_font = './fonts/gg mono.ttf'
        args.printer_font_size = 18
        args.printer_y_offset = 0
        args.row_gap = 0
        args.fallback_font = './fonts/SourceCodePro-VariableFont_wght.ttf'
        args.ban_chars = '`\\'

    # Add block characters to ban list if requested
    if args.ban_blocks:
        args.ban_chars += '░▒▓█▄▌▐▀■'

    return args


if __name__ == "__main__":
    args = parse_args()

    # Update global configuration from args
    CHAR_WIDTH = args.char_width
    CHAR_HEIGHT = args.char_height
    GRID_WIDTH = args.grid_width
    GRID_HEIGHT = args.grid_height
    ROW_GAP = args.row_gap
    IMAGE_WIDTH = CHAR_WIDTH * GRID_WIDTH
    IMAGE_HEIGHT = CHAR_HEIGHT * GRID_HEIGHT + ROW_GAP * (GRID_HEIGHT - 1)

    ENCODING = args.encoding
    BANNED_CHARS = list(args.ban_chars)

    PRINTER_FONT = args.printer_font
    PRINTER_FONT_SIZE = args.printer_font_size
    PRINTER_Y_OFFSET = args.printer_y_offset
    FALLBACK_FONTS = [args.fallback_font]
    FALLBACK_FONT_SIZE = args.fallback_font_size

    # Rebuild character set with new configuration
    if ENCODING == 'cp437':
        CHARS = ''.join(bytes([i]).decode('cp437') for i in range(32, 256))
    else:
        CHARS = ''.join(chr(i) for i in range(32, 127))
    CHARS = ''.join(c for c in CHARS if c not in BANNED_CHARS)
    NUM_CHARS = len(CHARS)

    # Print all configuration
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Preset:           {args.preset or 'None'}")
    print(f"Input Image:      {args.input_image if not args.test_chars else 'N/A (test mode)'}")
    print()
    print("Grid Configuration:")
    print(f"  Grid Size:      {GRID_WIDTH}x{GRID_HEIGHT} characters")
    print(f"  Character Size: {CHAR_WIDTH}x{CHAR_HEIGHT} pixels")
    print(f"  Row Gap:        {ROW_GAP} pixels")
    print(f"  Image Size:     {IMAGE_WIDTH}x{IMAGE_HEIGHT} pixels")
    print()
    print("Character Set:")
    print(f"  Encoding:       {ENCODING}")
    print(f"  Total Chars:    {NUM_CHARS}")
    print(f"  Banned:         {repr(args.ban_chars) if args.ban_chars else 'None'}")
    print(f"  Included:       {CHARS}")
    print()
    print("Fonts:")
    print(f"  Printer Font:   {PRINTER_FONT} ({PRINTER_FONT_SIZE}pt, y-offset={PRINTER_Y_OFFSET})")
    print(f"  Fallback Font:  {args.fallback_font} ({FALLBACK_FONT_SIZE}pt)")
    print()
    print("Training Hyperparameters:")
    print(f"  Iterations:     {args.iterations}")
    print(f"  Learning Rate:  {args.lr}")
    print(f"  Warmup:         {args.warmup} iterations")
    print(f"  Diversity:      {args.diversity_weight} (whitespace {'protected' if not args.penalize_whitespace else ' '})")
    print(f"  Gumbel-softmax: {'Enabled' if not args.no_gumbel else 'Disabled'}")
    if not args.no_gumbel:
        print(f"    Temperature:  {args.temp_start} → {args.temp_end}")
        print(f"    Save Temp:    {args.save_temp}")
    print()
    print("Output:")
    print(f"  Save Interval:  every {args.save_interval} iterations")
    print(f"  Output Image:   {args.output}")
    print(f"  Output Text:    {args.output_text}")
    print(f"  Output UTF-8:   {args.output_utf8}")
    print("=" * 70)
    print()

    # Test mode
    if args.test_chars:
        test_char_bitmaps()
        exit()

    # Training mode
    char_bitmaps = create_char_bitmaps()
    target_image = load_target_image(args.input_image)

    logits = train(
        target_image, char_bitmaps,
        num_iterations=args.iterations,
        lr=args.lr,
        save_interval=args.save_interval,
        warmup_iterations=args.warmup,
        diversity_weight=args.diversity_weight,
        use_gumbel=not args.no_gumbel,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        protect_whitespace=not args.penalize_whitespace
    )

    save_result(
        logits, char_bitmaps,
        output_path=args.output,
        text_path=args.output_text,
        utf8_path=args.output_utf8,
        temperature=args.save_temp
    )

    print("\nDone!")
