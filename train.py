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
FALLBACK_FONTS = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"]
FALLBACK_FONT_SIZE = 24

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
            # Use fallback font with Y offset -5
            font = fallback_font
            y_offset = -5
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


def load_target_image(image_path, keep_rgb=False):
    """Load and preprocess target image.

    Args:
        image_path: path to image
        keep_rgb: if True, return RGB tensor instead of grayscale
    """
    img = Image.open(image_path)

    if keep_rgb:
        img = img.convert('RGB')
    else:
        img = img.convert('L')

    # Resize to content dimensions (without gap space)
    content_height = CHAR_HEIGHT * GRID_HEIGHT  # 504
    img = img.resize((IMAGE_WIDTH, content_height), Image.LANCZOS)

    # Convert to array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0

    if keep_rgb:
        # Pad RGB image: (H, W, 3)
        if IMAGE_HEIGHT > content_height:
            padding = IMAGE_HEIGHT - content_height
            pad_top = padding // 2
            pad_bottom = padding - pad_top
            img_array = np.pad(img_array, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=1.0)
    else:
        # Pad grayscale image: (H, W)
        if IMAGE_HEIGHT > content_height:
            padding = IMAGE_HEIGHT - content_height
            pad_top = padding // 2
            pad_bottom = padding - pad_top
            img_array = np.pad(img_array, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=1.0)  # White padding

    # Convert to tensor
    img_tensor = torch.tensor(img_array, dtype=torch.float32, device=DEVICE)

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
    print("\n  Learned Contrast Curve (Center Control Point):")
    print("  +" + "-" * width + "+")
    for line in lines:
        print("  |" + line + "|")
    print("  +" + "-" * width + "+")
    print("  0" + " " * (width // 2 - 1) + "input" + " " * (width // 2 - 4) + "1")


def optimize_rgb_curves(rgb_image, iterations=250, lr=0.01):
    """
    Learn a neural network to map RGB→LAB→grayscale for maximum separability.

    Uses a small MLP: LAB (3D) -> hidden layers -> grayscale (1D)
    
    Main idea: we need to lose information somehow. We lose information by sacrificing distinguishibility
    of nearby colors. Push similar colors together to make more room in the space for more colors total.

    Args:
        rgb_image: (H, W, 3) tensor with values in [0, 1]
        iterations: optimization iterations
        lr: learning rate

    Returns:
        gray_image: (H, W) grayscale tensor
        model: the learned neural network
    """
    print(f"\nLearning LAB→grayscale mapping for maximum separability ({iterations} iterations)...")
    print(f"  Architecture: LAB(3) -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid")

    H, W, C = rgb_image.shape
    assert C == 3, "Input must be RGB"

    # Convert entire image to LAB upfront
    rgb_flat = rgb_image.reshape(-1, 3)  # (H*W, 3)

    # Linearize sRGB
    mask = rgb_flat > 0.04045
    rgb_linear = torch.where(mask,
                             ((rgb_flat + 0.055) / 1.055) ** 2.4,
                             rgb_flat / 12.92)

    # RGB to XYZ
    rgb_to_xyz = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=DEVICE, dtype=rgb_flat.dtype)
    xyz = rgb_linear @ rgb_to_xyz.T

    # XYZ to LAB
    xyz_n = torch.tensor([0.95047, 1.0, 1.08883], device=DEVICE, dtype=xyz.dtype)
    xyz_norm = xyz / xyz_n
    delta = 6.0 / 29.0
    mask_f = xyz_norm > delta ** 3
    f_xyz = torch.where(mask_f,
                       xyz_norm ** (1.0 / 3.0),
                       xyz_norm / (3.0 * delta ** 2) + 4.0 / 29.0)

    L = 116.0 * f_xyz[:, 1] - 16.0
    a = 500.0 * (f_xyz[:, 0] - f_xyz[:, 1])
    b = 200.0 * (f_xyz[:, 1] - f_xyz[:, 2])
    lab_full = torch.stack([L, a, b], dim=-1)  # (H*W, 3)

    # Normalize LAB to reasonable range for neural network
    # L: [0, 100], a: [-128, 127], b: [-128, 127]
    lab_normalized = lab_full.clone()
    lab_normalized[:, 0] = lab_normalized[:, 0] / 100.0  # L to [0, 1]
    lab_normalized[:, 1] = (lab_normalized[:, 1] + 128.0) / 255.0  # a to [0, 1]
    lab_normalized[:, 2] = (lab_normalized[:, 2] + 128.0) / 255.0  # b to [0, 1]

    # Downsample LAB for loss computation (low-frequency focus)
    downsample_factor = 4
    lab_image = lab_normalized.reshape(H, W, 3)
    lab_for_loss = lab_image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    lab_downsampled = F.avg_pool2d(lab_for_loss, kernel_size=downsample_factor, stride=downsample_factor)
    lab_downsampled = lab_downsampled.squeeze(0).permute(1, 2, 0)  # (H', W', 3)
    H_down, W_down, _ = lab_downsampled.shape

    # Build simple MLP with GELU activations (prevent dead neurons)
    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.GELU(),
        nn.Linear(16, 8),
        nn.GELU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Cosine learning rate schedule with warmup
    warmup_iterations = int(0.1 * iterations)  # 10% warmup

    for i in range(iterations):
        # Adjust learning rate
        if i < warmup_iterations:
            # Linear warmup
            lr_mult = i / warmup_iterations
        else:
            # Cosine annealing
            progress = (i - warmup_iterations) / (iterations - warmup_iterations)
            lr_mult = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * lr_mult

        optimizer.zero_grad()

        # Forward pass on full image
        gray_flat = model(lab_normalized).squeeze(-1)  # (H*W,)
        gray_image = gray_flat.reshape(H, W)

        # Downsample grayscale for loss
        gray_for_loss = gray_image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        gray_downsampled = F.avg_pool2d(gray_for_loss, kernel_size=downsample_factor, stride=downsample_factor)
        gray_downsampled = gray_downsampled.squeeze()  # (H', W')

        # Sample random pixels for distance preservation
        num_pixels_down = H_down * W_down
        num_samples = min(10000, num_pixels_down)
        indices = torch.randperm(num_pixels_down, device=DEVICE)[:num_samples]

        lab_flat = lab_downsampled.reshape(num_pixels_down, 3)
        lab_samples = lab_flat[indices]  # (num_samples, 3)
        gray_samples = gray_downsampled.reshape(num_pixels_down)[indices]  # (num_samples,)

        # Compute pairwise distances (subsample for efficiency)
        num_pairs = min(1000, num_samples)
        pair_indices = torch.randperm(num_samples, device=DEVICE)[:num_pairs]

        lab_subset = lab_samples[pair_indices]  # (num_pairs, 3)
        gray_subset = gray_samples[pair_indices]  # (num_pairs,)

        # Pairwise LAB distances
        lab_dist = torch.cdist(lab_subset, lab_subset, p=2)  # (num_pairs, num_pairs)

        # Pairwise grayscale distances
        gray_dist = torch.cdist(gray_subset.unsqueeze(-1), gray_subset.unsqueeze(-1), p=2).squeeze(-1)

        # Distance-preserving loss via cross-entropy over similarity distributions
        temperature = 0.1
        lab_similarities = F.softmax(-lab_dist / temperature, dim=-1)
        gray_similarities = F.log_softmax(-gray_dist / temperature, dim=-1)

        # Contrastive clustering loss: create gaps between color clusters
        # For similar LAB colors: pull grayscale values together (cluster)
        # For different LAB colors: push grayscale values apart (create gaps)

        lab_dist_flat = lab_dist.flatten()
        gray_dist_flat = gray_dist.flatten()

        # Define similarity threshold - pairs closer than this should cluster
        similarity_threshold = torch.quantile(lab_dist_flat, 0.2)  # Bottom 20% = similar
        dissimilarity_threshold = torch.quantile(lab_dist_flat, 0.5)  # Top 50% = different

        similar_mask = lab_dist_flat < similarity_threshold
        different_mask = lab_dist_flat > dissimilarity_threshold

        # Pull similar colors together in grayscale (minimize their distance)
        if similar_mask.sum() > 0:
            cluster_loss = gray_dist_flat[similar_mask].mean()
        else:
            cluster_loss = torch.tensor(0.0, device=DEVICE)

        # Push different colors apart in grayscale with a margin
        margin = 0.25  # Minimum grayscale separation for different colors
        if different_mask.sum() > 0:
            # Hinge loss: penalize if grayscale distance < margin
            separation_loss = torch.clamp(margin - gray_dist_flat[different_mask], min=0.0).mean()
        else:
            separation_loss = torch.tensor(0.0, device=DEVICE)

        # Luminance preservation: roughly match L channel (weak constraint)
        # keep white as white, keep black as black
        luminance_input = lab_normalized[:, 0]
        luminance_loss = ((gray_flat - luminance_input) ** 2).mean()

        # Total loss
        loss = cluster_loss + 1*separation_loss + 5*luminance_loss

        loss.backward()
        optimizer.step()

        if i % 50 == 0 or i == iterations - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Iteration {i}/{iterations}: cluster={cluster_loss.item():.4f}, separate={separation_loss.item():.4f}, luma={luminance_loss.item():.4f}, lr={current_lr:.6f}")

    # Generate final grayscale
    with torch.no_grad():
        gray_final = model(lab_normalized).squeeze(-1).reshape(H, W)

    print(f"LAB→grayscale learning complete.")
    print(f"  Output brightness range: [{gray_final.min().item():.3f}, {gray_final.max().item():.3f}]")

    # Save result
    gray_img_array = (gray_final.cpu().numpy() * 255).astype(np.uint8)
    gray_img = Image.fromarray(gray_img_array, mode='L')
    gray_img.save("rgb_curves_output.png")
    print(f"  Saved grayscale result to: rgb_curves_output.png")

    return gray_final, model


def optimize_contrast_curve_field(image, num_bins=64, iterations=200, lr=0.05, smoothness_weight=0.1):
    """
    Optimize a spatially-varying tone curve field to maximize local entropy.
    Each control point has its own tone curve, interpolated across the image.

    Args:
        image: (H, W) tensor with values in [0, 1]
        num_bins: number of bins per curve (kept smaller than 256 for performance)
        iterations: optimization iterations
        lr: learning rate
        smoothness_weight: regularization weight for curve smoothness between neighbors.
                           Lower regularization leads to a strong effect like an anime artist who went too hard with the shading.

    Returns:
        contrast_adjusted: (H, W) tensor with adjusted contrast
        center_curve: curve at center control point for visualization
    """
    print(f"\nOptimizing spatially-varying tone curve field ({iterations} iterations)...")
    print(f"  Control grid: ({GRID_HEIGHT+1} × {GRID_WIDTH+1}) = {(GRID_HEIGHT+1)*(GRID_WIDTH+1)} control points")
    print(f"  Bins per curve: {num_bins}")

    H, W = image.shape

    # Initialize curve increments for each control point
    # Shape: (GRID_HEIGHT+1, GRID_WIDTH+1, num_bins)
    # Start with uniform increments (identity curves everywhere)
    curve_increments = nn.Parameter(
        torch.ones(GRID_HEIGHT + 1, GRID_WIDTH + 1, num_bins, device=DEVICE) / num_bins
    )

    optimizer = optim.Adam([curve_increments], lr=lr)

    # Precompute pixel-to-control-point interpolation (reuse warp cache structure)
    cy0, cy1, cx0, cx1 = WARP_INTERP_CACHE['cy0'], WARP_INTERP_CACHE['cy1'], WARP_INTERP_CACHE['cx0'], WARP_INTERP_CACHE['cx1']
    wy0, wy1, wx0, wx1 = WARP_INTERP_CACHE['wy0'], WARP_INTERP_CACHE['wy1'], WARP_INTERP_CACHE['wx0'], WARP_INTERP_CACHE['wx1']

    for i in range(iterations):
        optimizer.zero_grad()

        # Build curves for each control point via cumsum
        positive_increments = F.softplus(curve_increments)  # (H+1, W+1, bins)
        normalized_increments = positive_increments / positive_increments.sum(dim=-1, keepdim=True)
        curves = torch.cumsum(normalized_increments, dim=-1)  # (H+1, W+1, bins)
        # Prepend 0 to each curve
        curves = torch.cat([torch.zeros(GRID_HEIGHT+1, GRID_WIDTH+1, 1, device=DEVICE), curves], dim=-1)  # (H+1, W+1, bins+1)

        # For each pixel, interpolate curve parameters from 4 nearest control points
        # Then apply the interpolated curve
        img_flat = image.flatten()

        # Get curves at 4 corners for each pixel
        curves_00 = curves[cy0, cx0]  # (H, W, bins+1)
        curves_01 = curves[cy0, cx1]
        curves_10 = curves[cy1, cx0]
        curves_11 = curves[cy1, cx1]

        # Flatten spatial dimensions
        curves_00 = curves_00.reshape(-1, num_bins + 1)  # (H*W, bins+1)
        curves_01 = curves_01.reshape(-1, num_bins + 1)
        curves_10 = curves_10.reshape(-1, num_bins + 1)
        curves_11 = curves_11.reshape(-1, num_bins + 1)

        # Bilinearly interpolate curve parameters
        w00 = (wy0 * wx0).flatten().unsqueeze(-1)  # (H*W, 1)
        w01 = (wy0 * wx1).flatten().unsqueeze(-1)
        w10 = (wy1 * wx0).flatten().unsqueeze(-1)
        w11 = (wy1 * wx1).flatten().unsqueeze(-1)

        interp_curves = curves_00 * w00 + curves_01 * w01 + curves_10 * w10 + curves_11 * w11  # (H*W, bins+1)

        # Apply interpolated curve to each pixel
        # Map pixel value to curve via interpolation
        bin_indices = (img_flat * num_bins).clamp(0, num_bins)
        idx0 = torch.floor(bin_indices).long()
        idx1 = torch.clamp(idx0 + 1, 0, num_bins)
        weight1 = bin_indices - idx0.float()
        weight0 = 1.0 - weight1

        # Index into interpolated curves (vectorized)
        adjusted_flat = interp_curves[torch.arange(len(img_flat), device=DEVICE), idx0] * weight0 + \
                        interp_curves[torch.arange(len(img_flat), device=DEVICE), idx1] * weight1

        adjusted = adjusted_flat.reshape(image.shape)

        # Compute histogram and entropy
        # Use soft binning for differentiability
        bin_centers = torch.linspace(0, 1, num_bins, device=DEVICE)
        distances = torch.abs(adjusted_flat.unsqueeze(1) - bin_centers.unsqueeze(0))
        sigma = 0.02
        weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))
        histogram = weights.sum(dim=0)
        histogram = histogram / histogram.sum()

        entropy = -(histogram * torch.log(histogram + 1e-10)).sum()

        # Smoothness regularization: penalize differences between neighboring control points' curves
        # Horizontal differences
        dx_increments = curve_increments[:, 1:, :] - curve_increments[:, :-1, :]
        # Vertical differences
        dy_increments = curve_increments[1:, :, :] - curve_increments[:-1, :, :]

        smoothness_loss = (dx_increments ** 2).mean() + (dy_increments ** 2).mean()

        # Total loss: maximize entropy (minimize -entropy) + smoothness regularization
        loss = -entropy + smoothness_weight * smoothness_loss

        loss.backward()
        optimizer.step()

        if i % 50 == 0 or i == iterations - 1:
            print(f"  Iteration {i}/{iterations}: entropy={entropy.item():.4f}, smoothness={smoothness_loss.item():.4f}")

    # Apply final curves
    with torch.no_grad():
        positive_increments = F.softplus(curve_increments)
        normalized_increments = positive_increments / positive_increments.sum(dim=-1, keepdim=True)
        curves_final = torch.cumsum(normalized_increments, dim=-1)
        curves_final = torch.cat([torch.zeros(GRID_HEIGHT+1, GRID_WIDTH+1, 1, device=DEVICE), curves_final], dim=-1)

        img_flat = image.flatten()

        curves_00 = curves_final[cy0, cx0]
        curves_01 = curves_final[cy0, cx1]
        curves_10 = curves_final[cy1, cx0]
        curves_11 = curves_final[cy1, cx1]

        # Flatten spatial dimensions
        curves_00 = curves_00.reshape(-1, num_bins + 1)
        curves_01 = curves_01.reshape(-1, num_bins + 1)
        curves_10 = curves_10.reshape(-1, num_bins + 1)
        curves_11 = curves_11.reshape(-1, num_bins + 1)

        w00 = (wy0 * wx0).flatten().unsqueeze(-1)
        w01 = (wy0 * wx1).flatten().unsqueeze(-1)
        w10 = (wy1 * wx0).flatten().unsqueeze(-1)
        w11 = (wy1 * wx1).flatten().unsqueeze(-1)

        interp_curves = curves_00 * w00 + curves_01 * w01 + curves_10 * w10 + curves_11 * w11

        bin_indices = (img_flat * num_bins).clamp(0, num_bins)
        idx0 = torch.floor(bin_indices).long()
        idx1 = torch.clamp(idx0 + 1, 0, num_bins)
        weight1 = bin_indices - idx0.float()
        weight0 = 1.0 - weight1

        adjusted_flat = interp_curves[torch.arange(len(img_flat), device=DEVICE), idx0] * weight0 + \
                        interp_curves[torch.arange(len(img_flat), device=DEVICE), idx1] * weight1
        adjusted_image = adjusted_flat.reshape(image.shape)

        # Get center curve for visualization
        center_y = GRID_HEIGHT // 2
        center_x = GRID_WIDTH // 2
        center_curve = curves_final[center_y, center_x].cpu().numpy()

    print(f"Spatially-varying contrast optimization complete.")
    print(f"  Input brightness range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"  Output brightness range: [{adjusted_image.min().item():.3f}, {adjusted_image.max().item():.3f}]")

    # Visualize non-linearity map
    print_curve_nonlinearity_map(curves_final, width=32, height=16)

    return adjusted_image, center_curve


def print_curve_nonlinearity_map(curves, width=32, height=16):
    """
    Display a 32×16 ASCII map showing how non-linear each control point's curve is.

    Args:
        curves: (GRID_HEIGHT+1, GRID_WIDTH+1, num_bins+1) tensor of curves
        width: display width in characters
        height: display height in characters
    """
    # Compute non-linearity for each control point
    # Non-linearity = RMS deviation from identity curve (y=x)
    num_bins = curves.shape[-1]
    identity = torch.linspace(0, 1, num_bins, device=curves.device)
    deviations = (curves - identity) ** 2  # (H+1, W+1, bins)
    nonlinearity = torch.sqrt(deviations.mean(dim=-1))  # (H+1, W+1)

    # Convert to numpy
    nonlinearity_np = nonlinearity.cpu().numpy()

    # Resample to display size
    from scipy.ndimage import zoom
    scale_y = height / nonlinearity_np.shape[0]
    scale_x = width / nonlinearity_np.shape[1]
    resampled = zoom(nonlinearity_np, (scale_y, scale_x), order=1)

    # Map to ASCII characters (low to high non-linearity)
    chars = ' .-:=+*#@'
    min_val = resampled.min()
    max_val = resampled.max()

    print("\n  Curve Non-linearity Map (deviation from identity):")
    print("  +" + "-" * width + "+")

    for row in range(height):
        line = []
        for col in range(width):
            val = resampled[row, col]
            # Normalize to [0, 1]
            if max_val > min_val:
                normalized = (val - min_val) / (max_val - min_val)
            else:
                normalized = 0
            # Map to character
            char_idx = int(normalized * (len(chars) - 1))
            char_idx = min(char_idx, len(chars) - 1)
            line.append(chars[char_idx])
        print("  |" + ''.join(line) + "|")

    print("  +" + "-" * width + "+")
    print(f"  Range: {min_val:.3f} (linear) to {max_val:.3f} (very non-linear)")


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
    Always expects batched inputs.

    Args:
        image: (B, H, W) tensor
        tx_global, ty_global: (B,) global translation
        warp_tx, warp_ty: (B, GRID_HEIGHT+1, GRID_WIDTH+1) local warp offsets
        scale_x, scale_y: (B,) scale factors
    """
    B, H, W = image.shape

    # Unpack global cached values
    cy0, cy1, cx0, cx1 = WARP_INTERP_CACHE['cy0'], WARP_INTERP_CACHE['cy1'], WARP_INTERP_CACHE['cx0'], WARP_INTERP_CACHE['cx1']
    wy0, wy1, wx0, wx1 = WARP_INTERP_CACHE['wy0'], WARP_INTERP_CACHE['wy1'], WARP_INTERP_CACHE['wx0'], WARP_INTERP_CACHE['wx1']
    y_out, x_out = WARP_INTERP_CACHE['y_out'], WARP_INTERP_CACHE['x_out']
    center_y, center_x = WARP_INTERP_CACHE['center_y'], WARP_INTERP_CACHE['center_x']

    # Bilinearly interpolate local warp offsets (batched)
    # warp_tx/ty: (B, GRID_HEIGHT+1, GRID_WIDTH+1)
    # cy0, cx0, etc: (H, W)
    # Result: (B, H, W)
    tx_warp_interp = (
        warp_tx[:, cy0, cx0] * wy0 * wx0 +
        warp_tx[:, cy0, cx1] * wy0 * wx1 +
        warp_tx[:, cy1, cx0] * wy1 * wx0 +
        warp_tx[:, cy1, cx1] * wy1 * wx1
    )
    ty_warp_interp = (
        warp_ty[:, cy0, cx0] * wy0 * wx0 +
        warp_ty[:, cy0, cx1] * wy0 * wx1 +
        warp_ty[:, cy1, cx0] * wy1 * wx0 +
        warp_ty[:, cy1, cx1] * wy1 * wx1
    )

    # Reshape scalars for broadcasting: (B,) -> (B, 1, 1)
    tx_global = tx_global.view(B, 1, 1)
    ty_global = ty_global.view(B, 1, 1)
    scale_x = scale_x.view(B, 1, 1)
    scale_y = scale_y.view(B, 1, 1)

    # Apply inverse transformation to find source coordinates
    # Order: (1) scale from center, (2) global translate, (3) local warp
    y_coords = (y_out - center_y) / scale_y + center_y - ty_global - ty_warp_interp  # (B, H, W)
    x_coords = (x_out - center_x) / scale_x + center_x - tx_global - tx_warp_interp  # (B, H, W)

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

    # Create batch indices for advanced indexing
    batch_idx = torch.arange(B, device=DEVICE).view(B, 1, 1).expand(B, H, W)

    # Gather 4 neighbors with validity masks (white padding for out of bounds)
    val_00 = torch.where(valid_y0 & valid_x0, image[batch_idx, y0_safe, x0_safe], torch.ones(1, device=DEVICE))
    val_01 = torch.where(valid_y0 & valid_x1, image[batch_idx, y0_safe, x1_safe], torch.ones(1, device=DEVICE))
    val_10 = torch.where(valid_y1 & valid_x0, image[batch_idx, y1_safe, x0_safe], torch.ones(1, device=DEVICE))
    val_11 = torch.where(valid_y1 & valid_x1, image[batch_idx, y1_safe, x1_safe], torch.ones(1, device=DEVICE))

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
    Always expects batched input.

    Args:
        logits: (B, GRID_HEIGHT, GRID_WIDTH, NUM_CHARS) - unnormalized scores
        char_bitmaps: (NUM_CHARS, CHAR_HEIGHT, CHAR_WIDTH) - character bitmaps
        temperature: Temperature for softmax (lower = more discrete)
        use_gumbel: Whether to add Gumbel noise

    Returns:
        rendered: (B, IMAGE_HEIGHT, IMAGE_WIDTH) - rendered images with row gaps
    """
    B = logits.shape[0]

    # Apply Gumbel noise if requested
    if use_gumbel and logits.requires_grad:  # Only during training
        # Sample Gumbel noise: g = -log(-log(u)) where u ~ Uniform(0,1)
        u = torch.rand_like(logits, device=logits.device)
        gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
        logits_with_noise = logits + gumbel_noise
    else:
        logits_with_noise = logits

    # Apply temperature-scaled softmax to get character weights
    weights = torch.softmax(logits_with_noise / temperature, dim=-1)  # (B, GRID_HEIGHT, GRID_WIDTH, NUM_CHARS)

    # Vectorized rendering using einsum
    # weights: (B, GRID_HEIGHT, GRID_WIDTH, NUM_CHARS)
    # char_bitmaps: (NUM_CHARS, CHAR_HEIGHT, CHAR_WIDTH)
    # Result: (B, GRID_HEIGHT, GRID_WIDTH, CHAR_HEIGHT, CHAR_WIDTH)
    rendered_grid = torch.einsum('bijk,khw->bijhw', weights, char_bitmaps)

    # Reshape to image with row gaps
    # (B, GRID_HEIGHT, GRID_WIDTH, CHAR_HEIGHT, CHAR_WIDTH) -> (B, GRID_HEIGHT, CHAR_HEIGHT, GRID_WIDTH, CHAR_WIDTH)
    rendered_grid = rendered_grid.permute(0, 1, 3, 2, 4).contiguous()
    # -> (B, GRID_HEIGHT, CHAR_HEIGHT, IMAGE_WIDTH)
    rendered_grid = rendered_grid.view(B, GRID_HEIGHT, CHAR_HEIGHT, IMAGE_WIDTH)

    if ROW_GAP > 0:
        # Create output with gaps (white = 1.0)
        rendered = torch.ones((B, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32, device=DEVICE)

        # Place each row with gaps
        for i in range(GRID_HEIGHT):
            y_start = i * (CHAR_HEIGHT + ROW_GAP)
            y_end = y_start + CHAR_HEIGHT
            rendered[:, y_start:y_end, :] = rendered_grid[:, i]
    else:
        # No gaps, just reshape
        rendered = rendered_grid.view(B, IMAGE_HEIGHT, IMAGE_WIDTH)

    return rendered


def train(target_image, char_bitmaps, num_iterations=1000, lr=0.01, save_interval=100, warmup_iterations=50, diversity_weight=0.01,
          use_gumbel=True, temp_start=1.0, temp_end=0.01, protect_whitespace=True, multiscale_weight=0.0, multiscale_kernel=4,
          optimize_alignment=False, alignment_lr=0.01, warp_reg_weight=0.01, dark_mode=False,
          prev_alignment_params=None, temporal_weight=0.0):
    """Train ASCII art using gradient descent with cosine learning rate schedule, diversity loss, multiscale perceptual loss, learnable spatial alignment, and Gumbel-softmax.
    Always expects batched input where batch dim = temporal/frame dim.
    prev_alignment_params: Optional tuple of (tx_base, ty_base, tx_warp, ty_warp, sx, sy) from previous batch for temporal continuity."""

    # Clear and create steps directory
    if os.path.exists("steps"):
        shutil.rmtree("steps")
    os.makedirs("steps")

    # Get batch size (number of frames)
    B = target_image.shape[0]

    # Initialize logits randomly
    logits = nn.Parameter(
        torch.randn(B, GRID_HEIGHT, GRID_WIDTH, NUM_CHARS, device=DEVICE) * 0.01
    )

    # Learnable spatial transformation: global shift + per-control-point warping
    if optimize_alignment:
        # Global translation (shifts entire image) - batched for each frame
        translation_x = nn.Parameter(torch.zeros(B, device=DEVICE))
        translation_y = nn.Parameter(torch.zeros(B, device=DEVICE))

        # Control points at corners of character cells: (B, GRID_HEIGHT+1, GRID_WIDTH+1)
        # Local warping on top of global shift
        control_tx = nn.Parameter(torch.zeros(B, GRID_HEIGHT + 1, GRID_WIDTH + 1, device=DEVICE))
        control_ty = nn.Parameter(torch.zeros(B, GRID_HEIGHT + 1, GRID_WIDTH + 1, device=DEVICE))

        # Global scale (same for entire image) - batched for each frame
        scale_x_param = nn.Parameter(torch.zeros(B, device=DEVICE))  # sigmoid -> [0.9, 1.2]
        scale_y_param = nn.Parameter(torch.zeros(B, device=DEVICE))

        optimizer = optim.AdamW([
            {'params': [logits], 'lr': lr},
            {'params': [translation_x, translation_y, scale_x_param, scale_y_param, control_tx, control_ty], 'lr': alignment_lr}
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
        print(f"  Global translation ±{CHAR_WIDTH/2:.1f}px H/V + per-control-point warp ±{CHAR_WIDTH/2:.1f}px H/V, scale 0.9-1.2x")

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
            # Add channel dimension for pooling (already batched: B, H, W)
            rendered_4d = rendered_cmp.unsqueeze(1)  # (B, 1, H, W)
            target_4d = target_cmp.unsqueeze(1)

            # Downsample both rendered and target with overlapping patches
            # Use stride = kernel_size // 2 for 50% overlap
            stride = max(1, multiscale_kernel // 2)
            rendered_small = F.avg_pool2d(rendered_4d, kernel_size=multiscale_kernel, stride=stride).squeeze(1)
            target_small = F.avg_pool2d(target_4d, kernel_size=multiscale_kernel, stride=stride).squeeze(1)

            # Loss on downsampled version
            multiscale_loss = criterion(rendered_small, target_small)
        else:
            multiscale_loss = torch.tensor(0.0).to(DEVICE)

        if diversity_weight != 0.0:
            # Compute diversity loss (entropy of character usage, excluding whitespace)
            weights = torch.softmax(logits, dim=-1)  # (B, GRID_HEIGHT, GRID_WIDTH, NUM_CHARS)
            char_usage = weights.mean(dim=[0, 1, 2])  # (NUM_CHARS,) - average usage across batch and spatial dims

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
            # tx_warp/ty_warp are (B, GRID_HEIGHT+1, GRID_WIDTH+1)
            dx_tx = (tx_warp[:, :, 1:] - tx_warp[:, :, :-1]) ** 2  # horizontal differences
            dy_tx = (tx_warp[:, 1:, :] - tx_warp[:, :-1, :]) ** 2  # vertical differences
            dx_ty = (ty_warp[:, :, 1:] - ty_warp[:, :, :-1]) ** 2
            dy_ty = (ty_warp[:, 1:, :] - ty_warp[:, :-1, :]) ** 2
            warp_reg_loss = dx_tx.mean() + dy_tx.mean() + dx_ty.mean() + dy_ty.mean()
        else:
            warp_reg_loss = torch.tensor(0.0).to(DEVICE)

        # Gentle regularization on global alignment parameters (prefer identity transform)
        if optimize_alignment:
            alignment_reg_weight = 0.001
            alignment_reg_loss = ((translation_x ** 2).mean() + (translation_y ** 2).mean() +
                                 (scale_x_param ** 2).mean() + (scale_y_param ** 2).mean() +
                                 (tx_warp ** 2).mean() + (ty_warp ** 2).mean())
        else:
            alignment_reg_loss = torch.tensor(0.0).to(DEVICE)

        # Temporal regularization: alignment params should be similar between adjacent frames
        if optimize_alignment and temporal_weight > 0:
            temporal_loss = torch.tensor(0.0, device=DEVICE)

            # Within-batch: consecutive frames
            if B > 1:
                temporal_loss = (
                    ((tx_base[1:] - tx_base[:-1]) ** 2).mean() +
                    ((ty_base[1:] - ty_base[:-1]) ** 2).mean() +
                    ((tx_warp[1:] - tx_warp[:-1]) ** 2).mean() +
                    ((ty_warp[1:] - ty_warp[:-1]) ** 2).mean() +
                    ((sx[1:] - sx[:-1]) ** 2).mean() +
                    ((sy[1:] - sy[:-1]) ** 2).mean()
                )

            # Cross-batch: first frame of this batch to last frame of prev batch
            if prev_alignment_params is not None:
                prev_tx_base, prev_ty_base, prev_tx_warp, prev_ty_warp, prev_sx, prev_sy = prev_alignment_params
                cross_batch_loss = (
                    ((tx_base[0] - prev_tx_base) ** 2).mean() +
                    ((ty_base[0] - prev_ty_base) ** 2).mean() +
                    ((tx_warp[0] - prev_tx_warp) ** 2).mean() +
                    ((ty_warp[0] - prev_ty_warp) ** 2).mean() +
                    ((sx[0] - prev_sx) ** 2).mean() +
                    ((sy[0] - prev_sy) ** 2).mean()
                )
                temporal_loss = temporal_loss + cross_batch_loss
        else:
            temporal_loss = torch.tensor(0.0).to(DEVICE)

        # Total loss
        loss = (recon_loss
            + multiscale_weight * multiscale_loss
            + diversity_weight * diversity_loss
            + warp_reg_weight * warp_reg_loss
            + alignment_reg_loss
            + temporal_weight * temporal_loss)

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
        if temporal_weight > 0 and B > 1:
            postfix['temp_reg'] = f'{temporal_loss.item():.4f}'
        if optimize_alignment:
            # Show global base translation and mean warp (averaged across batch)
            postfix['tx_base'] = f'{tx_base.mean().item():.1f}'
            postfix['ty_base'] = f'{ty_base.mean().item():.1f}'
            postfix['warp'] = f'{tx_warp.abs().mean().item():.1f}'
            postfix['sx'] = f'{sx.mean().item():.3f}'
            postfix['sy'] = f'{sy.mean().item():.3f}'
        pbar.set_postfix(postfix)

        # Save intermediate results - loop through each frame in batch
        if save_interval > 0:
            if iteration % save_interval == 0 or iteration == num_iterations - 1:
                for b in range(B):
                    save_result(
                        logits[b],
                        char_bitmaps,
                        output_path=f"steps/i_iter_{iteration:04d}_f{b:04d}.png",
                        text_path=f"steps/t_iter_{iteration:04d}_f{b:04d}.txt",
                        temperature=temperature,
                        target_image=target_shifted[b] if optimize_alignment else None,
                        warp_params={'tx_warp': tx_warp[b], 'ty_warp': ty_warp[b], 'tx_base': tx_base[b].item(), 'ty_base': ty_base[b].item()} if optimize_alignment else None,
                        dark_mode=dark_mode
                    )

    # Return batched results and all alignment params
    if optimize_alignment:
        # Return all alignment params for all frames: (B, ...)
        alignment_params = {
            'tx_base': tx_base.detach(),  # (B,)
            'ty_base': ty_base.detach(),  # (B,)
            'tx_warp': tx_warp.detach(),  # (B, GRID_HEIGHT+1, GRID_WIDTH+1)
            'ty_warp': ty_warp.detach(),  # (B, GRID_HEIGHT+1, GRID_WIDTH+1)
            'sx': sx.detach(),  # (B,)
            'sy': sy.detach(),  # (B,)
            'target_shifted': target_shifted.detach()  # (B, H, W)
        }
        return logits, alignment_params
    else:
        return logits, None


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
    # render_ascii expects batched input, so add batch dim
    rendered = render_ascii(logits.unsqueeze(0), char_bitmaps, temperature=temperature, use_gumbel=False)
    rendered = rendered.squeeze(0)  # Remove batch dim

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

    # Input arguments (one of input_image or --animation required)
    parser.add_argument('input_image', nargs='?', help='Input image path (not needed for animation mode)')

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
                       default='/System/Library/Fonts/Menlo.ttc',
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
    parser.add_argument('--optimize-alignment', action='store_true',
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
    parser.add_argument('--optimize-contrast', action='store_true',
                       help='Optimize tone curve to maximize histogram entropy (fixes poor contrast). Higher values lead to more shading. (Default: true)')
    parser.add_argument('--optimize-rgb-to-gray', action='store_true',
                       help='Learn model to map RGB to grayscale to maximize color separation (for color images). Important for images where information is mostly chrominance')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save intermediate results every N iterations (default: 100)')
    parser.add_argument('--output', type=str, default='output.png',
                       help='Output image path (default: output.png)')
    parser.add_argument('--output-text', type=str, default='output.txt',
                       help='Output text file path (default: output.txt)')
    parser.add_argument('--output-utf8', type=str, default='output.utf8.txt',
                       help='Output UTF-8 text file path (default: output.utf8.txt)')
    parser.add_argument('--animation', type=str,
                       help='Path to text file containing list of frame paths for animation mode')
    parser.add_argument('--animation-temporal-weight', type=float, default=1.0,
                       help='Weight for temporal regularization (alignment params similar to previous frame)')
    parser.add_argument('--animation-batch-size', type=int, default=None,
                       help='Process animation in batches of this size (default: process all frames at once)')

    args = parser.parse_args()

    # Validate input: need either input_image or --animation
    if not args.input_image and not args.animation:
        parser.error("Either input_image or --animation must be provided")
    if args.input_image and args.animation:
        parser.error("Cannot specify both input_image and --animation")

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

    # Helper functions
    def preprocess_image(image_path, optimize_rgb, optimize_contrast, plot_contrast=False):
        """Load and preprocess image with optional RGB and contrast optimization."""
        if optimize_rgb:
            target_image_rgb = load_target_image(image_path, keep_rgb=True)
            target_image, rgb_model = optimize_rgb_curves(target_image_rgb)
        else:
            target_image = load_target_image(image_path, keep_rgb=False)

        print(f"Target image shape: {target_image.shape}")

        if optimize_contrast:
            target_image, contrast_curve = optimize_contrast_curve_field(target_image)
            if plot_contrast:
                plot_curve_ascii(contrast_curve)

        return target_image

    def make_train_kwargs(args, prev_alignment_params=None, temporal_weight=0.0):
        """Construct kwargs dict for train() based on args."""
        return {
            'num_iterations': args.iterations,
            'lr': args.lr,
            'save_interval': args.save_interval,
            'warmup_iterations': args.warmup,
            'diversity_weight': args.diversity_weight,
            'use_gumbel': not args.no_gumbel,
            'temp_start': args.temp_start,
            'temp_end': args.temp_end,
            'protect_whitespace': not args.penalize_whitespace,
            'multiscale_weight': args.multiscale_weight,
            'multiscale_kernel': args.multiscale_kernel,
            'optimize_alignment': args.optimize_alignment,
            'alignment_lr': args.alignment_lr,
            'warp_reg_weight': args.warp_reg_weight,
            'dark_mode': args.dark_mode,
            'prev_alignment_params': prev_alignment_params,
            'temporal_weight': temporal_weight
        }

    # Training mode
    char_bitmaps = create_char_bitmaps()

    # Construct base training kwargs (same for all frames in animation)
    base_train_kwargs = make_train_kwargs(args)

    # Animation mode or single image mode
    if args.animation:
        # Animation mode: process frames in batches
        with open(args.animation, 'r') as f:
            frame_paths = [line.strip() for line in f if line.strip()]

        total_frames = len(frame_paths)
        batch_size = args.animation_batch_size if args.animation_batch_size else total_frames
        num_batches = (total_frames + batch_size - 1) // batch_size

        print(f"\nAnimation mode: processing {total_frames} frames")
        print(f"Batch size: {batch_size} (splitting into {num_batches} batch(es))")
        print(f"Temporal regularization weight: {args.animation_temporal_weight}")
        print("Note: RGB/contrast optimization disabled")

        # Create frames directory
        os.makedirs("frames", exist_ok=True)

        # Process each batch
        prev_alignment_params = None
        frame_idx = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_frames)
            batch_paths = frame_paths[start_idx:end_idx]

            print(f"\n{'='*70}")
            print(f"Batch {batch_idx + 1}/{num_batches}: frames {start_idx}-{end_idx - 1} ({len(batch_paths)} frames)")
            print(f"{'='*70}")

            # Load batch frames into batched tensor
            print("Loading batch frames...")
            frame_list = []
            for frame_path in tqdm(batch_paths):
                frame = load_target_image(frame_path, keep_rgb=False)
                frame_list.append(frame)

            # Stack into batch: (B, H, W)
            target_images = torch.stack(frame_list, dim=0)
            print(f"Loaded {target_images.shape[0]} frames, shape: {target_images.shape}")

            # Train on batch with temporal regularization
            train_kwargs = base_train_kwargs.copy()
            train_kwargs['temporal_weight'] = args.animation_temporal_weight
            train_kwargs['prev_alignment_params'] = prev_alignment_params

            logits, alignment_params = train(target_images, char_bitmaps, **train_kwargs)

            # Save each frame from the batch
            print("\nSaving batch frames...")
            for b in range(logits.shape[0]):
                # Extract alignment params for this frame if available
                if alignment_params is not None:
                    warp_params = {
                        'tx_warp': alignment_params['tx_warp'][b],
                        'ty_warp': alignment_params['ty_warp'][b],
                        'tx_base': alignment_params['tx_base'][b].item(),
                        'ty_base': alignment_params['ty_base'][b].item()
                    }
                    target_shifted = alignment_params['target_shifted'][b]
                else:
                    warp_params = None
                    target_shifted = None

                save_result(
                    logits[b], char_bitmaps,
                    output_path=f"frames/{frame_idx:04d}.png",
                    text_path=f"frames/enc_{frame_idx:04d}.txt",
                    utf8_path=f"frames/{frame_idx:04d}.txt",
                    temperature=args.save_temp,
                    target_image=target_shifted,
                    warp_params=warp_params,
                    dark_mode=args.dark_mode
                )
                frame_idx += 1

            # Extract alignment params from last frame of batch for next batch
            if alignment_params is not None and batch_idx < num_batches - 1:
                prev_alignment_params = (
                    alignment_params['tx_base'][-1],
                    alignment_params['ty_base'][-1],
                    alignment_params['tx_warp'][-1],
                    alignment_params['ty_warp'][-1],
                    alignment_params['sx'][-1],
                    alignment_params['sy'][-1]
                )
            else:
                prev_alignment_params = None

        print(f"\n{'='*70}")
        print(f"Animation complete! {total_frames} frames saved to frames/")
        print(f"{'='*70}")

    else:
        # Single image mode with pre-optimization
        target_image = preprocess_image(
            args.input_image,
            optimize_rgb=args.optimize_rgb_to_gray,
            optimize_contrast=args.optimize_contrast,
            plot_contrast=True
        )

        # Add batch dimension for single image (B=1)
        target_image = target_image.unsqueeze(0)  # (1, H, W)

        logits, alignment_params = train(target_image, char_bitmaps, **base_train_kwargs)

        # Extract single frame from batch
        logits = logits[0]

        if alignment_params is not None:
            warp_params = {
                'tx_warp': alignment_params['tx_warp'][0],
                'ty_warp': alignment_params['ty_warp'][0],
                'tx_base': alignment_params['tx_base'][0].item(),
                'ty_base': alignment_params['ty_base'][0].item()
            }
            target_shifted = alignment_params['target_shifted'][0]
        else:
            warp_params = None
            target_shifted = None

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
