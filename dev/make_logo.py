"""Generate dev/nanochat_rl.png in the style of dev/nanochat.png.

Strategy: render text small (so each glyph stroke is just a few pixels),
threshold to 1-bit, dilate for chunky weight, then composite a multi-step
halftone drop-shadow extending down-right. Final image is nearest-neighbor
upscaled so individual pixels stay crisp.
"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter

TEXT = "nanochat|RL"
FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"

# Small native pixel grid -> chunky pixels after upscale.
PIXEL_W, PIXEL_H = 236, 54
SCALE = 5                       # final 1180 x 270

BG = (38, 38, 38)
FG = (215, 215, 215)

# Multi-step halftone shadow (offset_x, offset_y, checker_phase).
# Each subsequent layer extends further down-right with same stipple.
SHADOW_LAYERS = [
    (1, 1, 0),
    (2, 2, 1),
    (3, 3, 0),
]


def render_text_mask(size, text, font_path, extra_letter_spacing=2):
    """Render with explicit per-character spacing to leave room for dilation."""
    w, h = size
    font_size = h
    while font_size > 4:
        font = ImageFont.truetype(font_path, font_size)
        widths = [font.getbbox(c)[2] - font.getbbox(c)[0] for c in text]
        total = sum(widths) + extra_letter_spacing * (len(text) - 1)
        bbox_h = font.getbbox(text)
        th = bbox_h[3] - bbox_h[1]
        if total <= w - 2 and th <= h - 1:
            break
        font_size -= 1
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    bbox_h = font.getbbox(text)
    y = (h - (bbox_h[3] - bbox_h[1])) // 2 - bbox_h[1]
    x = (w - total) // 2
    for c, cw in zip(text, widths):
        cb = font.getbbox(c)
        draw.text((x - cb[0], y), c, fill=255, font=font)
        x += cw + extra_letter_spacing
    return img


def shift(img, dx, dy):
    out = Image.new("L", img.size, 0)
    out.paste(img, (dx, dy))
    return out


def main():
    mask = render_text_mask((PIXEL_W, PIXEL_H), TEXT, FONT_PATH)
    mask = mask.point(lambda v: 255 if v > 100 else 0, mode="L")

    base = Image.new("RGB", (PIXEL_W, PIXEL_H), BG)
    base_px = base.load()
    mask_px = mask.load()

    # Build halftone shadow layers (each: pixels in shifted-mask AND not in
    # original mask AND on the layer's checker phase). Earlier layers paint
    # first so later (further) layers are visible only where they extend
    # beyond earlier ones.
    shadow_pixels = {}  # (x,y) -> True
    for dx, dy, phase in SHADOW_LAYERS:
        sh = shift(mask, dx, dy).load()
        for y in range(PIXEL_H):
            for x in range(PIXEL_W):
                if sh[x, y] and not mask_px[x, y] and (x + y) % 2 == phase:
                    shadow_pixels.setdefault((x, y), True)

    for (x, y) in shadow_pixels:
        base_px[x, y] = FG
    for y in range(PIXEL_H):
        for x in range(PIXEL_W):
            if mask_px[x, y]:
                base_px[x, y] = FG

    out = base.resize((PIXEL_W * SCALE, PIXEL_H * SCALE), Image.NEAREST)
    out.save("dev/nanochat_rl.png", optimize=True)
    print(f"wrote dev/nanochat_rl.png ({out.size[0]}x{out.size[1]})")


if __name__ == "__main__":
    main()
