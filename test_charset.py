from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Test which characters we can render
CHAR_WIDTH = 12
CHAR_HEIGHT = 24

FONT_SIZE = 24
FONT_YOFFSET = 4

font = None
font_paths = [
    "./fonts/bitArray-A2.ttf"
    # "/System/Library/Fonts/Supplemental/Menlo.ttc",
]

for path in font_paths:
    try:
        font = ImageFont.truetype(path, FONT_SIZE)
        print(f"Using font: {path}")
        break
    except:
        continue

if font is None:
    font = ImageFont.load_default()
    print("Using default PIL font (limited charset)")

# Try rendering extended ASCII (128-255) for CP437
test_chars = []
failed_chars = []

for i in range(32, 256):
    try:
        # Decode byte as CP437 to get proper character
        char = bytes([i]).decode('cp437')
        img = Image.new('L', (CHAR_WIDTH, CHAR_HEIGHT), 255)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=font, fill=0)

        # Check if anything was drawn (not all white)
        arr = np.array(img)
        if arr.min() < 255:  # Something was drawn
            test_chars.append(i)
        else:
            failed_chars.append(i)
    except Exception as e:
        failed_chars.append(i)
        print(f"Failed to render {i}: {e}")

print(f"\nSuccessfully rendered: {len(test_chars)} characters")
print(f"Failed/empty: {len(failed_chars)} characters")

print(f"\nTotal 7-bit ASCII (32-126): {len([c for c in test_chars if c < 127])}")
print(f"Total 8-bit extended (127-255): {len([c for c in test_chars if c >= 127])}")

if failed_chars:
    print(f"\nFirst fail at: {failed_chars[0]} ('{bytes([failed_chars[0]]).decode('cp437')}')")
    print(f"\nFailed range examples (first 20): {failed_chars[:20]}")

# Show what we got in extended range
extended = [c for c in test_chars if c >= 127]
if extended:
    print(f"\nExtended characters we CAN render: {extended}")
    print(f"As CP437 chars: {[bytes([c]).decode('cp437') for c in extended]}")
else:
    print("\nNo extended ASCII characters rendered!")

# Create a visual grid
chars_per_row = 16
num_rows = (256 - 32 + chars_per_row - 1) // chars_per_row

grid_height = num_rows * CHAR_HEIGHT
grid_width = chars_per_row * CHAR_WIDTH
grid = np.ones((grid_height, grid_width), dtype=np.float32) * 255  # White background

for i in range(32, 256):
    idx = i - 32
    row = idx // chars_per_row
    col = idx % chars_per_row

    try:
        # Decode byte as CP437
        char = bytes([i]).decode('cp437')
        img = Image.new('L', (CHAR_WIDTH, CHAR_HEIGHT), 255)
        draw = ImageDraw.Draw(img)
        draw.text((0, FONT_YOFFSET), char, font=font, fill=0)

        y_start = row * CHAR_HEIGHT
        y_end = (row + 1) * CHAR_HEIGHT
        x_start = col * CHAR_WIDTH
        x_end = (col + 1) * CHAR_WIDTH

        grid[y_start:y_end, x_start:x_end] = np.array(img)
    except:
        pass

grid_img = Image.fromarray(grid.astype(np.uint8), mode='L')
grid_img.save("charset_test.png")
print(f"\nSaved charset visualization to charset_test.png")
