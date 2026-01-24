#!/usr/bin/env python3
"""
Test script to render all CP437 characters (32-254) in a repeating grid.
Useful for verifying font rendering and character support.
"""

from train import *

def test_all_chars_grid():
    """Test rendering by filling grid with all CP437 chars 32-254 in repeating loop."""
    print("Testing all characters grid...")
    char_bitmaps = create_char_bitmaps()

    # Create character indices for full grid (32-254 repeating)
    char_indices = []
    char_range = list(range(32, 255))  # CP437 characters 32-254

    for i in range(GRID_HEIGHT):
        row_indices = []
        for j in range(GRID_WIDTH):
            pos = i * GRID_WIDTH + j
            # Loop through char_range
            char_code = char_range[pos % len(char_range)]
            # Map char code to CHARS index
            char = bytes([char_code]).decode('cp437', errors='replace')
            if char in CHARS:
                char_idx = CHARS.index(char)
            else:
                char_idx = 0  # Fallback to first character
            row_indices.append(char_idx)
        char_indices.append(row_indices)

    # Convert to tensor
    char_indices_tensor = torch.tensor(char_indices, dtype=torch.long, device=DEVICE)

    # Create one-hot encoding for discrete rendering
    one_hot = torch.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_CHARS), device=DEVICE)
    one_hot.scatter_(-1, char_indices_tensor.unsqueeze(-1), 1.0)

    # Render (render_ascii expects batched input, so add batch dim)
    rendered = render_ascii(one_hot.unsqueeze(0), char_bitmaps, temperature=0.01, use_gumbel=False)
    rendered = rendered.squeeze(0)  # Remove batch dim

    # Save
    img_array = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    img.save("test_all_chars.png")
    print(f"Saved test_all_chars.png ({GRID_WIDTH}x{GRID_HEIGHT} grid, {IMAGE_WIDTH}x{IMAGE_HEIGHT}px)")


if __name__ == "__main__":
    test_all_chars_grid()
