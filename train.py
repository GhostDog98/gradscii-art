import torch
import torch.nn as nn
import torch.nn.functional as F
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
WARP_INTERP_CACHE = None  # Initialized after IMAGE_HEIGHT/WIDTH are set

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


def plot_curve_ascii(curve, width=32, height=16):
    """
    Plot a tone curve as ASCII art using sub-character resolution.

    Args:
        curve: numpy array of curve values (0-1)
        width: plot width in characters
        height: plot height in characters
    """
    # Characters for sub-row detail (6 levels from bottom to top within a row)
    chars = "_.-^`'"

    # Sample curve at width points
    x_indices = np.linspace(0, len(curve) - 1, width).astype(int)
    y_values = curve[x_indices]

    # Build plot from top to bottom (row 0 = top = y=1.0)
    lines = []
    for row in range(height):
        line = []
        # Row represents y range [row_min, row_max]
        row_max = 1.0 - (row / height)
        row_min = 1.0 - ((row + 1) / height)

        for col in range(width):
            y = y_values[col]

            if y >= row_max:
                # Above this row
                line.append(' ')
            elif y < row_min:
                # Below this row
                line.append(' ')
            else:
                # Within this row - use sub-character detail
                # Position within row (0=bottom, 1=top)
                pos_in_row = (y - row_min) / (row_max - row_min)
                char_idx = int(pos_in_row * len(chars))
                char_idx = min(char_idx, len(chars) - 1)
                line.append(chars[char_idx])

        lines.append(''.join(line))

    # Print with border
    print("\n  Learned Contrast Curve:")
    print("  +" + "-" * width + "+")
    for line in lines:
        print("  |" + line + "|")
    print("  +" + "-" * width + "+")
    print("  0" + " " * (width // 2 - 1) + "input" + " " * (width // 2 - 4) + "1")


def optimize_contrast_curve(image, num_bins=256):
    """
    Apply histogram equalization to maximize contrast.

    Args:
        image: (H, W) tensor with values in [0, 1]
        num_bins: number of histogram bins

    Returns:
        contrast_adjusted: (H, W) tensor with adjusted contrast
        curve: the equalization curve (CDF) for visualization
    """
    print(f"\nApplying histogram equalization...")

    # Compute histogram and CDF (classical histogram equalization)
    img_np = image.cpu().numpy().flatten()
    hist, bins = np.histogram(img_np, num_bins, [0, 1])
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]

    # Apply equalization via interpolation
    equalized_np = np.interp(img_np, bins[:-1], cdf).reshape(image.shape)
    equalized_image = torch.from_numpy(equalized_np).float().to(DEVICE)

    print(f"Histogram equalization complete.")
    print(f"  Input brightness range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"  Output brightness range: [{equalized_image.min().item():.3f}, {equalized_image.max().item():.3f}]")

    return equalized_image, cdf


def precompute_warp_interpolation_structure(H, W):
    """Precompute fixed interpolation structure for control point warping (only depends on grid, not warp values)."""
    # Create coordinate grids for output pixels
    y_out = torch.arange(H, device=DEVICE, dtype=torch.float32).view(-1, 1)
    x_out = torch.arange(W, device=DEVICE, dtype=torch.float32).view(1, -1)

    # Map pixel coordinates to control grid coordinates
    if ROW_GAP > 0:
        char_y = y_out / (CHAR_HEIGHT + ROW_GAP)
        char_x = x_out / CHAR_WIDTH
    else:
        char_y = y_out / CHAR_HEIGHT
        char_x = x_out / CHAR_WIDTH

    # Clamp and get control point indices
    char_y_clamped = torch.clamp(char_y, 0, GRID_HEIGHT)
    char_x_clamped = torch.clamp(char_x, 0, GRID_WIDTH)

    cy0 = torch.floor(char_y_clamped).long()
    cy1 = torch.clamp(cy0 + 1, 0, GRID_HEIGHT)
    cx0 = torch.floor(char_x_clamped).long()
    cx1 = torch.clamp(cx0 + 1, 0, GRID_WIDTH)

    # Interpolation weights
    wy1 = char_y_clamped - cy0.float()
    wy0 = 1.0 - wy1
    wx1 = char_x_clamped - cx0.float()
    wx0 = 1.0 - wx1

    # Centers for scaling
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0

    return {
        'cy0': cy0, 'cy1': cy1, 'cx0': cx0, 'cx1': cx1,
        'wy0': wy0, 'wy1': wy1, 'wx0': wx0, 'wx1': wx1,
        'y_out': y_out, 'x_out': x_out,
        'center_y': center_y, 'center_x': center_x
    }


def apply_spatially_varying_transform(image, tx_global, ty_global, warp_tx, warp_ty, scale_x, scale_y):
    """
    Apply spatially-varying transformation using precomputed global WARP_INTERP_CACHE.

    Args:
        image: (H, W) tensor
        tx_global, ty_global: scalar global translation
        warp_tx, warp_ty: (GRID_HEIGHT+1, GRID_WIDTH+1) local warp offsets
        scale_x, scale_y: scalar scale factors
    """
    H, W = image.shape

    # Unpack global cached values
    cy0, cy1, cx0, cx1 = WARP_INTERP_CACHE['cy0'], WARP_INTERP_CACHE['cy1'], WARP_INTERP_CACHE['cx0'], WARP_INTERP_CACHE['cx1']
    wy0, wy1, wx0, wx1 = WARP_INTERP_CACHE['wy0'], WARP_INTERP_CACHE['wy1'], WARP_INTERP_CACHE['wx0'], WARP_INTERP_CACHE['wx1']
    y_out, x_out = WARP_INTERP_CACHE['y_out'], WARP_INTERP_CACHE['x_out']
    center_y, center_x = WARP_INTERP_CACHE['center_y'], WARP_INTERP_CACHE['center_x']

    # Bilinearly interpolate local warp offsets (this is the only dynamic part)
    tx_warp_interp = (
        warp_tx[cy0, cx0] * wy0 * wx0 +
        warp_tx[cy0, cx1] * wy0 * wx1 +
        warp_tx[cy1, cx0] * wy1 * wx0 +
        warp_tx[cy1, cx1] * wy1 * wx1
    )
    ty_warp_interp = (
        warp_ty[cy0, cx0] * wy0 * wx0 +
        warp_ty[cy0, cx1] * wy0 * wx1 +
        warp_ty[cy1, cx0] * wy1 * wx0 +
        warp_ty[cy1, cx1] * wy1 * wx1
    )

    # Apply inverse transformation to find source coordinates
    # Order: (1) scale from center, (2) global translate, (3) local warp
    y_coords = (y_out - center_y) / scale_y + center_y - ty_global - ty_warp_interp
    x_coords = (x_out - center_x) / scale_x + center_x - tx_global - tx_warp_interp

    # Get integer coordinates for 4 neighbors
    y0 = torch.floor(y_coords).long()
    y1 = y0 + 1
    x0 = torch.floor(x_coords).long()
    x1 = x0 + 1

    # Compute interpolation weights
    wy1_interp = y_coords - y0.float()
    wy0_interp = 1.0 - wy1_interp
    wx1_interp = x_coords - x0.float()
    wx0_interp = 1.0 - wx1_interp

    # Create masks for valid coordinates (within bounds)
    valid_y0 = (y0 >= 0) & (y0 < H)
    valid_y1 = (y1 >= 0) & (y1 < H)
    valid_x0 = (x0 >= 0) & (x0 < W)
    valid_x1 = (x1 >= 0) & (x1 < W)

    # Clamp coordinates for safe indexing
    y0_safe = torch.clamp(y0, 0, H - 1)
    y1_safe = torch.clamp(y1, 0, H - 1)
    x0_safe = torch.clamp(x0, 0, W - 1)
    x1_safe = torch.clamp(x1, 0, W - 1)

    # Gather 4 neighbors with validity masks (white padding for out of bounds)
    val_00 = torch.where(valid_y0 & valid_x0, image[y0_safe, x0_safe], torch.ones(1, device=DEVICE))
    val_01 = torch.where(valid_y0 & valid_x1, image[y0_safe, x1_safe], torch.ones(1, device=DEVICE))
    val_10 = torch.where(valid_y1 & valid_x0, image[y1_safe, x0_safe], torch.ones(1, device=DEVICE))
    val_11 = torch.where(valid_y1 & valid_x1, image[y1_safe, x1_safe], torch.ones(1, device=DEVICE))

    # Bilinear interpolation
    transformed = (
        val_00 * wy0_interp * wx0_interp +
        val_01 * wy0_interp * wx1_interp +
        val_10 * wy1_interp * wx0_interp +
        val_11 * wy1_interp * wx1_interp
    )

    return transformed


def apply_transform(image, tx, ty, scale_x, scale_y):
    """
    Apply spatial transformation (translation + scaling) using manual bilinear interpolation.
    MPS doesn't support grid_sample backward, so we implement it manually.

    Args:
        image: (H, W) tensor
        tx: horizontal translation in pixels (positive = shift right)
        ty: vertical translation in pixels (positive = shift down)
        scale_x: horizontal scale factor (1.0 = no scaling, <1.0 = downscale)
        scale_y: vertical scale factor (1.0 = no scaling, <1.0 = downscale)

    Returns:
        transformed: (H, W) transformed image
    """
    H, W = image.shape

    # Centers for scaling
    center_y = (H - 1) / 2.0
    center_x = (W - 1) / 2.0

    # Create coordinate grids for output pixels
    y_out = torch.arange(H, device=DEVICE, dtype=torch.float32).view(-1, 1)
    x_out = torch.arange(W, device=DEVICE, dtype=torch.float32).view(1, -1)

    # Apply inverse transformation to find source coordinates
    # For each output pixel, compute where to sample from in the input
    # Scale from center, then translate
    y_coords = (y_out - center_y) / scale_y + center_y - ty
    x_coords = (x_out - center_x) / scale_x + center_x - tx

    # Get integer coordinates for 4 neighbors
    y0 = torch.floor(y_coords).long()
    y1 = y0 + 1
    x0 = torch.floor(x_coords).long()
    x1 = x0 + 1

    # Compute interpolation weights
    wy1 = y_coords - y0.float()
    wy0 = 1.0 - wy1
    wx1 = x_coords - x0.float()
    wx0 = 1.0 - wx1

    # Create masks for valid coordinates (within bounds)
    valid_y0 = (y0 >= 0) & (y0 < H)
    valid_y1 = (y1 >= 0) & (y1 < H)
    valid_x0 = (x0 >= 0) & (x0 < W)
    valid_x1 = (x1 >= 0) & (x1 < W)

    # Clamp coordinates for safe indexing (but track validity separately)
    y0_safe = torch.clamp(y0, 0, H - 1)
    y1_safe = torch.clamp(y1, 0, H - 1)
    x0_safe = torch.clamp(x0, 0, W - 1)
    x1_safe = torch.clamp(x1, 0, W - 1)

    # Gather 4 neighbors with validity masks
    # If coordinate is out of bounds, use white (1.0) instead
    val_00 = torch.where(valid_y0 & valid_x0, image[y0_safe, x0_safe], torch.ones_like(image[0, 0]))
    val_01 = torch.where(valid_y0 & valid_x1, image[y0_safe, x1_safe], torch.ones_like(image[0, 0]))
    val_10 = torch.where(valid_y1 & valid_x0, image[y1_safe, x0_safe], torch.ones_like(image[0, 0]))
    val_11 = torch.where(valid_y1 & valid_x1, image[y1_safe, x1_safe], torch.ones_like(image[0, 0]))

    # Bilinear interpolation
    transformed = (
        val_00 * wy0 * wx0 +
        val_01 * wy0 * wx1 +
        val_10 * wy1 * wx0 +
        val_11 * wy1 * wx1
    )

    return transformed


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
          use_gumbel=True, temp_start=1.0, temp_end=0.01, protect_whitespace=True, multiscale_weight=0.0, multiscale_kernel=4,
          optimize_alignment=False, alignment_lr=0.01, warp_reg_weight=0.01, dark_mode=False):
    """Train ASCII art using gradient descent with cosine learning rate schedule, diversity loss, multiscale perceptual loss, learnable spatial alignment, and Gumbel-softmax."""

    # Clear and create steps directory
    if os.path.exists("steps"):
        shutil.rmtree("steps")
    os.makedirs("steps")

    # Initialize logits randomly
    logits = nn.Parameter(
        torch.randn(GRID_HEIGHT, GRID_WIDTH, NUM_CHARS, device=DEVICE) * 0.01
    )

    # Learnable spatial transformation: global shift + per-control-point warping
    if optimize_alignment:
        # Global translation (shifts entire image)
        translation_x = nn.Parameter(torch.zeros(1, device=DEVICE))
        translation_y = nn.Parameter(torch.zeros(1, device=DEVICE))

        # Control points at corners of character cells: (GRID_HEIGHT+1, GRID_WIDTH+1)
        # Local warping on top of global shift
        control_tx = nn.Parameter(torch.zeros(GRID_HEIGHT + 1, GRID_WIDTH + 1, device=DEVICE))
        control_ty = nn.Parameter(torch.zeros(GRID_HEIGHT + 1, GRID_WIDTH + 1, device=DEVICE))

        # Global scale (same for entire image)
        scale_x_param = nn.Parameter(torch.zeros(1, device=DEVICE))  # sigmoid -> [0.9, 1.2]
        scale_y_param = nn.Parameter(torch.zeros(1, device=DEVICE))

        optimizer = optim.AdamW([
            {'params': [logits], 'lr': lr},
            {'params': [translation_x, translation_y, control_tx, control_ty, scale_x_param, scale_y_param], 'lr': alignment_lr}
        ])
    else:
        translation_x = None
        translation_y = None
        control_tx = None
        control_ty = None
        scale_x_param = None
        scale_y_param = None
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
    if optimize_alignment:
        print(f"Spatial alignment: Global shift + deformation field ({GRID_HEIGHT+1}x{GRID_WIDTH+1} = {(GRID_HEIGHT+1)*(GRID_WIDTH+1)} control points)")
        print(f"  Global translation ±{CHAR_WIDTH/2:.1f}px H/V + per-control-point warp ±{CHAR_WIDTH/2:.1f}px H/V, scale 0.9-1.25x")

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

        # Apply learnable spatial transformation to target if enabled
        if optimize_alignment:
            # Global translation (shifts entire image)
            tx_base = (CHAR_WIDTH / 2) * torch.tanh(translation_x)
            ty_base = (CHAR_HEIGHT / 2) * torch.tanh(translation_y)

            # Local control point warping (bounded per control point)
            tx_warp = (CHAR_WIDTH / 2) * torch.tanh(control_tx)  # (GRID_HEIGHT+1, GRID_WIDTH+1)
            ty_warp = (CHAR_HEIGHT / 2) * torch.tanh(control_ty)

            # Map unconstrained scale params to [0.9, 1.2] via sigmoid
            sx = 0.9 + 0.3 * torch.sigmoid(scale_x_param)
            sy = 0.9 + 0.3 * torch.sigmoid(scale_y_param)

            # Apply spatially-varying transformation: scale -> global translate -> local warp
            target_shifted = apply_spatially_varying_transform(target_image, tx_base, ty_base, tx_warp, ty_warp, sx, sy)
        else:
            target_shifted = target_image

        # Invert for dark mode (white text on black background)
        if dark_mode:
            rendered_cmp = 1.0 - rendered
            target_cmp = target_shifted #1.0 - target_shifted
        else:
            rendered_cmp = rendered
            target_cmp = target_shifted

        # Compute reconstruction loss
        recon_loss = criterion(rendered_cmp, target_cmp)

        # Compute multiscale perceptual loss (dithering effect)
        if multiscale_weight != 0.0:
            # Add batch and channel dimensions for pooling
            rendered_4d = rendered_cmp.unsqueeze(0).unsqueeze(0)
            target_4d = target_cmp.unsqueeze(0).unsqueeze(0)

            # Downsample both rendered and target with overlapping patches
            # Use stride = kernel_size // 2 for 50% overlap
            stride = max(1, multiscale_kernel // 2)
            rendered_small = F.avg_pool2d(rendered_4d, kernel_size=multiscale_kernel, stride=stride).squeeze()
            target_small = F.avg_pool2d(target_4d, kernel_size=multiscale_kernel, stride=stride).squeeze()

            # Loss on downsampled version
            multiscale_loss = criterion(rendered_small, target_small)
        else:
            multiscale_loss = torch.tensor(0.0).to(DEVICE)

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

        # Warp regularization: penalize non-uniform warping (spatial gradients)
        if optimize_alignment and warp_reg_weight != 0.0:
            # Penalize differences between neighboring control points (Total Variation)
            # This allows uniform shifts but penalizes distortion
            dx_tx = (tx_warp[:, 1:] - tx_warp[:, :-1]) ** 2  # horizontal differences in tx
            dy_tx = (tx_warp[1:, :] - tx_warp[:-1, :]) ** 2  # vertical differences in tx
            dx_ty = (ty_warp[:, 1:] - ty_warp[:, :-1]) ** 2  # horizontal differences in ty
            dy_ty = (ty_warp[1:, :] - ty_warp[:-1, :]) ** 2  # vertical differences in ty
            warp_reg_loss = dx_tx.mean() + dy_tx.mean() + dx_ty.mean() + dy_ty.mean()
        else:
            warp_reg_loss = torch.tensor(0.0).to(DEVICE)

        # Total loss
        loss = recon_loss + multiscale_weight * multiscale_loss + diversity_weight * diversity_loss + warp_reg_weight * warp_reg_loss

        # Backprop
        loss.backward()

        optimizer.step()
        scheduler.step()

        # Update progress bar
        postfix = {
            'recon': f'{recon_loss.item():.4f}',
            'lr': f'{current_lr:.4f}',
            'temp': f'{temperature:.4f}'
        }
        if multiscale_weight != 0.0:
            postfix['ms'] = f'{multiscale_loss.item():.4f}'
        if diversity_weight != 0.0:
            postfix['div'] = f'{diversity_loss.item():.4f}'
        if optimize_alignment and warp_reg_weight != 0.0:
            postfix['w_reg'] = f'{warp_reg_loss.item():.4f}'
        if optimize_alignment:
            # Show global base translation and mean warp
            postfix['tx_base'] = f'{tx_base.item():.1f}'
            postfix['ty_base'] = f'{ty_base.item():.1f}'
            postfix['warp'] = f'{tx_warp.abs().mean().item():.1f}'
            postfix['sx'] = f'{sx.item():.3f}'
            postfix['sy'] = f'{sy.item():.3f}'
        pbar.set_postfix(postfix)

        # Save intermediate results
        if iteration % save_interval == 0 or iteration == num_iterations - 1:
            save_result(
                logits,
                char_bitmaps,
                output_path=f"steps/i_iter_{iteration:04d}.png",
                text_path=f"steps/t_iter_{iteration:04d}.txt",
                temperature=temperature,
                target_image=target_shifted if optimize_alignment else None,
                warp_params={'tx_warp': tx_warp, 'ty_warp': ty_warp, 'tx_base': tx_base.item(), 'ty_base': ty_base.item()} if optimize_alignment else None,
                dark_mode=dark_mode
            )

    if optimize_alignment:
        # Return global base translation and warp field
        return logits, tx_base.item(), ty_base.item(), target_shifted, tx_warp, ty_warp
    else:
        return logits


def save_result(logits, char_bitmaps, output_path="output.png", text_path="output.txt", utf8_path="", temperature=0.1, target_image=None, warp_params=None, dark_mode=False):
    """Save the final ASCII art as image and text.

    Args:
        warp_params: Optional dict with keys 'tx_warp', 'ty_warp', 'tx_base', 'ty_base' for visualizing deformation field
        dark_mode: If True, invert colors for display (white text on black background)
    """
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

    # Invert for dark mode display
    if dark_mode:
        rendered = 1.0 - rendered

    # Save as image
    img_array = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)

    # If target image provided, show side by side
    if target_image is not None:
        target_display = target_image
        # if dark_mode:
        #     target_display = 1.0 - target_display
        target_array = (target_display.detach().cpu().numpy() * 255).astype(np.uint8)

        # Draw warp control points on target image if provided
        if warp_params is not None:
            from PIL import ImageDraw
            # Convert to RGB to draw colored arrows
            target_img = Image.fromarray(target_array, mode='L').convert('RGB')
            draw = ImageDraw.Draw(target_img)

            tx_warp = warp_params['tx_warp'].detach().cpu().numpy()  # (GRID_HEIGHT+1, GRID_WIDTH+1)
            ty_warp = warp_params['ty_warp'].detach().cpu().numpy()
            tx_base = warp_params['tx_base']
            ty_base = warp_params['ty_base']

            # Draw control points and displacement vectors
            for i in range(GRID_HEIGHT + 1):
                for j in range(GRID_WIDTH + 1):
                    # Control point position in image coordinates
                    if ROW_GAP > 0:
                        y_pos = i * (CHAR_HEIGHT + ROW_GAP)
                        x_pos = j * CHAR_WIDTH
                    else:
                        y_pos = i * CHAR_HEIGHT
                        x_pos = j * CHAR_WIDTH

                    # Warp displacement at this control point
                    dx = tx_warp[i, j]
                    dy = ty_warp[i, j]

                    # Draw control point as a circle
                    draw.circle([x_pos+dx, y_pos+dy], radius=2, fill=(255, 0, 0))

            target_array = np.array(target_img)
        else:
            # Convert grayscale target to RGB for consistency
            target_array = np.stack([target_array]*3, axis=-1)

        # Convert rendered to RGB too
        img_array_rgb = np.stack([img_array]*3, axis=-1)

        # Horizontally concatenate: rendered | target
        img_array = np.hstack([img_array_rgb, target_array])

    img = Image.fromarray(img_array if target_image is not None else img_array, mode='RGB' if target_image is not None else 'L')
    img.save(output_path)


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
    parser.add_argument('--multiscale-weight', type=float, default=0.5,
                       help='Weight for multiscale perceptual loss (dithering effect) - optimizes for how it looks when downsampled (default: 0.5, try 0.0-1.0)')
    parser.add_argument('--multiscale-kernel', type=int, default=4,
                       help='Downsampling kernel size for multiscale loss (default: 4, simulates viewing distance)')
    parser.add_argument('--optimize-alignment', action='store_true', default=True,
                       help='Learn global spatial translation, scaling, and warp matrix to align image with character grid. Warning: slow')
    parser.add_argument('--alignment-lr', type=float, default=0.1,
                       help='Learning rate for spatial alignment (default: 0.1)')
    parser.add_argument('--warp-reg-weight', type=float, default=0.005,
                       help='Regularization weight for penalizing strong warping (default: 0.005, 0 to disable). Lower values will warp harder')

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
    parser.add_argument('--dark-mode', action='store_true',
                       help='Invert colors for dark mode (white text on black background)')
    parser.add_argument('--optimize-contrast', action='store_true', default=True,
                       help='Optimize tone curve to maximize histogram entropy (fixes poor contrast). (Default: true)')
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

    # Initialize warp interpolation cache for spatial alignment
    WARP_INTERP_CACHE = precompute_warp_interpolation_structure(IMAGE_HEIGHT, IMAGE_WIDTH)

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
    print(f"Input Image:      {args.input_image}")
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
    print(f"  Dark mode:      {args.dark_mode}")
    print(f"  Contrast opt:   {args.optimize_contrast}")
    print()
    print("Training Hyperparameters:")
    print(f"  Iterations:     {args.iterations}")
    print(f"  Learning Rate:  {args.lr}")
    print(f"  Warmup:         {args.warmup} iterations")
    print(f"  Diversity:      {args.diversity_weight} (whitespace {'protected' if not args.penalize_whitespace else 'included'})")
    print(f"  Multiscale:     {args.multiscale_weight} (kernel={args.multiscale_kernel})")
    print(f"  Alignment:      {'Enabled' if args.optimize_alignment else 'Disabled'}" + (f" (lr={args.alignment_lr})" if args.optimize_alignment else ""))
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

    # Training mode
    char_bitmaps = create_char_bitmaps()
    target_image = load_target_image(args.input_image)

    # Optimize contrast curve if requested
    if args.optimize_contrast:
        target_image, contrast_curve = optimize_contrast_curve(target_image)

        # Plot the curve as ASCII art
        plot_curve_ascii(contrast_curve)

    result = train(
        target_image, char_bitmaps,
        num_iterations=args.iterations,
        lr=args.lr,
        save_interval=args.save_interval,
        warmup_iterations=args.warmup,
        diversity_weight=args.diversity_weight,
        use_gumbel=not args.no_gumbel,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        protect_whitespace=not args.penalize_whitespace,
        multiscale_weight=args.multiscale_weight,
        multiscale_kernel=args.multiscale_kernel,
        optimize_alignment=args.optimize_alignment,
        alignment_lr=args.alignment_lr,
        warp_reg_weight=args.warp_reg_weight,
        dark_mode=args.dark_mode
    )

    if args.optimize_alignment:
        logits, tx_base, ty_base, target_shifted, tx_warp, ty_warp = result
        print(f"\nLearned global translation: x={tx_base:.2f}px, y={ty_base:.2f}px (+ per-control-point warping)")
        warp_params = {'tx_warp': tx_warp, 'ty_warp': ty_warp, 'tx_base': tx_base, 'ty_base': ty_base}
    else:
        logits = result
        target_shifted = None
        warp_params = None

    save_result(
        logits, char_bitmaps,
        output_path=args.output,
        text_path=args.output_text,
        utf8_path=args.output_utf8,
        temperature=args.save_temp,
        target_image=target_shifted,
        warp_params=warp_params,
        dark_mode=args.dark_mode
    )

    print("\nDone!")
