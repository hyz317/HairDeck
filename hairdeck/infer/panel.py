"""The preprocessing panel of the image pipeline: the input with the background dimmed, the hair and face masks and the
crop box, next to the model input."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def prep_panel(p, crop, height=448):
    """Left: the input with the background (outside the person matte) dimmed, hair in magenta, face in green and the
    crop box in orange (the canvas is padded when the box leaves the image). Right: the 448 px model input. Drawn at
    twice the panel resolution (large uploads are resized first), then downscaled."""
    h, w = p['rgb'].shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if p['box'] is None else p['box']
    pl, pt = max(0.0, -x0), max(0.0, -y0)
    pr, pb = max(0.0, x1 - w), max(0.0, y1 - h)
    pad = 0.02 * max(w, h) + 2
    s = min(1.0, 2 * height / (h + pt + pb + 2 * pad))  # working scale (original px -> working px)
    ws, hs = max(1, round(w * s)), max(1, round(h * s))
    rgb = np.asarray(Image.fromarray(p['rgb']).resize((ws, hs), Image.Resampling.BILINEAR), np.float64)
    a = np.asarray(Image.fromarray(p['person']).resize((ws, hs), Image.Resampling.BILINEAR), np.float64)[..., None] / 255
    img = rgb * (0.3 + 0.7 * a) + np.array([40, 40, 70.0]) * 0.7 * (1 - a)
    for m, col in ((p['hair'], (255, 0, 200)), (p['face'], (0, 210, 90))):
        mm = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((ws, hs), Image.Resampling.NEAREST)) > 127
        img[mm] = 0.55 * img[mm] + 0.45 * np.array(col, np.float64)
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    canvas = Image.new('RGB', (round((w + pl + pr + 2 * pad) * s), round((h + pt + pb + 2 * pad) * s)), (225, 225, 225))
    ox, oy = round((pl + pad) * s), round((pt + pad) * s)
    canvas.paste(img, (ox, oy))
    if p['box'] is not None:
        lw = max(3, round(0.006 * max(ws, hs)))
        ImageDraw.Draw(canvas).rectangle([ox + x0 * s, oy + y0 * s, ox + x1 * s, oy + y1 * s], outline=(255, 140, 0), width=lw)
    k = height / canvas.height
    canvas = canvas.resize((max(1, round(canvas.width * k)), height), Image.Resampling.LANCZOS)
    right = Image.fromarray(crop) if crop is not None else Image.new('RGB', (height, height), (240, 240, 240))
    if right.size != (height, height):
        right = right.resize((height, height), Image.Resampling.LANCZOS)
    f = _font(15)
    items = (((60, 60, 90), 'background'), ((255, 0, 200), 'hair'), ((0, 210, 90), 'face'), ((255, 140, 0), 'crop'),
             (None, 'right: cropped input' if crop is not None else 'right: no crop'))
    probe = ImageDraw.Draw(Image.new('RGB', (8, 8)))
    leg_w = sum((19 if c else 0) + int(probe.textlength(t, font=f)) + 16 for c, t in items) + 4
    gap, leg = 12, 30
    out = Image.new('RGB', (max(canvas.width + gap + height, leg_w), height + leg), (255, 255, 255))
    out.paste(canvas, (0, 0))
    out.paste(right, (canvas.width + gap, 0))
    d = ImageDraw.Draw(out)
    x = 4
    for col, txt in items:
        if col:
            d.rectangle([x, height + 9, x + 14, height + 23], fill=col)
            x += 19
        d.text((x, height + 7), txt, fill=(30, 30, 30), font=f)
        x += int(d.textlength(txt, font=f)) + 16
    return out
