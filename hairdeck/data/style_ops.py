"""Style operations of the widened appearance augmentation (hairdeck.data.imgdata.style_crop).

Every operation maps the character composited on white (H, W, 3 uint8, at the render's native resolution) to a styled
image of the same size; ctx carries the alpha matte (float, 0..1), the crop side in pixels and a seed. No operation
moves pixels: each is a per-pixel map or a local, translation-invariant filter, so the crop box and the hair label of
the render stay valid. numpy / OpenCV / PIL only.

Colour stage (one per sample):
  gray       luminance with a random gamma and contrast
  mono_tone  grayscale quantised to white / light screentone / dark screentone / black (45 deg dot screen, period
             6-8 px) plus thin ink lines (manga)
  sketch     uncoloured line drawing: XDoG (Winnemoeller et al. 2012) on the CLAHE-equalised L channel of the lightly
             smoothed image (sigma 0.8-1.0 px, k 1.6, tau 0.98, eps 0, phi 150); CLAHE so that light hair still gets
             its strand lines
  hue_full   hue rotation by an angle drawn over the full circle (float HSV)
  chan       RGB channel permutation, Lab (a, b) rotation with a chroma scale, or per-channel gain / gamma (colour cast)
  flat       flat cel colouring: edge-preserving smoothing, k-means to 6-8 colours (Lab), mode filter; no gradients
  painterly  Kuwahara filter (radius 9-11 px, two passes) plus a little saturation: smears strand detail
Then: thick_ink (Canny edges of a smoothed image plus the silhouette, dilated to 4-6 px, multiplied on as near-black
ink), rim_light (a lit band inside the silhouette along the edge facing a random light direction) and shade (a smooth
light-to-shadow ramp across the character), recomposite (back under the original alpha matte, white outside) and lowres
(the whole crop down to a few hundred pixels and back).
"""
import cv2
import numpy as np

cv2.setNumThreads(1)  # one thread per DataLoader worker (the forked workers inherit this)


def u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- colour stage
def gray(rgb, ctx, rng):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    g = g ** float(rng.uniform(0.7, 1.4))
    c = float(rng.uniform(0.85, 1.3))
    g = 0.5 + (g - 0.5) * c
    return u8(np.repeat(g[..., None], 3, -1) * 255)


def hue_full(rgb, ctx, rng):
    deg = float(rng.uniform(-180, 180))
    hsv = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2HSV)  # float: H in degrees [0, 360)
    hsv[..., 0] = (hsv[..., 0] + deg) % 360
    return u8(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255)


PERMS = ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def chan(rgb, ctx, rng):
    k = int(rng.integers(3))
    if k == 0:  # channel permutation
        return np.ascontiguousarray(rgb[..., list(PERMS[int(rng.integers(len(PERMS)))])])
    if k == 1:  # Lab chroma rotation + scale
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        th, s = np.radians(float(rng.uniform(0, 360))), float(rng.uniform(0.5, 1.6))
        a, b = lab[..., 1].copy(), lab[..., 2].copy()
        lab[..., 1] = s * (np.cos(th) * a - np.sin(th) * b)
        lab[..., 2] = s * (np.sin(th) * a + np.cos(th) * b)
        return u8(np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255)
    x = rgb.astype(np.float32) / 255  # colour cast: per-channel gain and gamma
    gain = rng.uniform(0.75, 1.25, 3).astype(np.float32)
    gam = rng.uniform(0.8, 1.25, 3).astype(np.float32)
    return u8(np.clip(x ** gam * gain, 0, 1) * 255)


def flat(rgb, ctx, rng):
    k = int(rng.integers(6, 9))
    sm = rgb
    for _ in range(2):
        sm = cv2.bilateralFilter(sm, 9, 40, 7)
    lab = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = ctx['alpha'] > 0.5
    data = lab[m]
    sub = data[rng.choice(len(data), min(len(data), 30000), replace=False)]
    cv2.setRNGSeed(int(ctx['seed']))
    _, _, centers = cv2.kmeans(sub, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2), 4, cv2.KMEANS_PP_CENTERS)
    d = ((lab[..., None, :] - centers[None, None]) ** 2).sum(-1)  # (H, W, k)
    onehot = np.eye(k, dtype=np.float32)[d.argmin(-1)]
    votes = np.stack([cv2.boxFilter(onehot[..., j], -1, (9, 9)) for j in range(k)], -1)  # mode filter: clean regions
    lab_out = centers[votes.argmax(-1)]
    out = cv2.cvtColor(u8(lab_out), cv2.COLOR_LAB2RGB)
    return np.where(m[..., None], out, rgb)


def _kuwahara(img, r):
    """Classic 4-quadrant Kuwahara: each pixel takes the mean colour of the (r+1)^2 quadrant with the lowest luminance
    variance."""
    x = img.astype(np.float32)
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    k = (r + 1, r + 1)
    means, vars_ = [], []
    for anchor in ((r, r), (0, r), (r, 0), (0, 0)):  # windows ending / starting at the pixel
        f = lambda a: cv2.boxFilter(a, -1, k, anchor=anchor, borderType=cv2.BORDER_REFLECT)  # noqa: E731
        means.append(f(x))
        vars_.append(f(lum * lum) - f(lum) ** 2)
    best = np.argmin(np.stack(vars_, 0), 0)
    out = np.zeros_like(x)
    for q in range(4):
        out[best == q] = means[q][best == q]
    return out


def painterly(rgb, ctx, rng):
    r = int(rng.integers(9, 12))
    x = _kuwahara(rgb, r)
    x = _kuwahara(x, max(r - 3, 4))
    hsv = cv2.cvtColor(np.clip(x / 255, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 1)
    return u8(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) * 255)


def _edges(rgb, alpha, lo, hi):
    sm = cv2.bilateralFilter(rgb, 9, 50, 7)
    e = cv2.Canny(cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY), lo, hi) > 0
    sil = (alpha > 0.5).astype(np.uint8)
    return e | (cv2.morphologyEx(sil, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)


def mono_tone(rgb, ctx, rng):
    h, w = rgb.shape[:2]
    sm = cv2.bilateralFilter(rgb, 9, 50, 7)
    g = cv2.cvtColor(sm, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    period = float(rng.uniform(6, 8))
    th = np.radians(45 + float(rng.uniform(-8, 8)))
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    u, v = (X * np.cos(th) + Y * np.sin(th)) / period, (-X * np.sin(th) + Y * np.cos(th)) / period
    s = (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v)) / 2  # 1 at dot centres
    m = ctx['alpha'] > 0.5
    q = np.quantile(g[m], [0.12, 0.45, 0.75]) if m.any() else np.array([0.2, 0.5, 0.75])
    cover = np.select([g <= q[0], g <= q[1], g <= q[2]], [1.0, 0.55, 0.25], 0.0)  # black / dark tone / light tone / white
    # a dot screen with ink coverage c: ink where s > threshold(c); thresholds from the empirical distribution of s
    ss = np.sort(s.ravel()[::7])
    thr = np.interp(1 - cover, np.linspace(0, 1, len(ss)), ss)
    ink = (s > thr) | (cover >= 1.0)
    ink &= cover > 0
    e = cv2.dilate(_edges(rgb, ctx['alpha'], 60, 140).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    out = np.where(ink | e, 20, 250).astype(np.float32)
    return u8(np.repeat(out[..., None], 3, -1))


def sketch(rgb, ctx, rng):
    sigma = float(rng.uniform(0.8, 1.0))
    L = cv2.cvtColor(cv2.bilateralFilter(rgb, 7, 30, 5), cv2.COLOR_RGB2LAB)[..., 0]
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(L).astype(np.float32) / 255
    D = cv2.GaussianBlur(g, (0, 0), sigma) - 0.98 * cv2.GaussianBlur(g, (0, 0), 1.6 * sigma)
    out = np.clip(np.where(D >= 0, 1.0, 1 + np.tanh(150 * D)), 0, 1) ** 1.5
    return u8(np.repeat(out[..., None], 3, -1) * 255)


COLOUR_OPS = dict(gray=gray, mono_tone=mono_tone, sketch=sketch, hue_full=hue_full, chan=chan, flat=flat,
                  painterly=painterly)
LINE_OPS = ('mono_tone', 'sketch')  # carry their own lines: no ink overlay and no lighting after them


# ---------------------------------------------------------------- ink and lighting
def thick_ink(rgb, ctx, rng):
    t = int(rng.integers(4, 7))
    e = _edges(rgb, ctx['alpha'], 40, 110).astype(np.uint8)
    ink = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))).astype(np.float32)
    ink = cv2.GaussianBlur(ink, (0, 0), 0.7)
    out = rgb.astype(np.float32) * (1 - 0.92 * ink[..., None]) + 10 * ink[..., None]
    return u8(out)


def _light_colour(rng):
    k = int(rng.integers(4))
    if k == 0:
        return np.array([255, 255, 255], np.float32)
    if k == 1:
        return np.array([255, 236, 200], np.float32)  # warm
    if k == 2:
        return np.array([205, 228, 255], np.float32)  # cool
    hsv = np.array([[[rng.uniform(0, 360), rng.uniform(0.3, 0.7), 1.0]]], np.float32)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] * 255


def rim_light(rgb, ctx, rng):
    """A lit band inside the silhouette along the edge that faces a random light direction (white, warm, cool or a
    random hue)."""
    a = ctx['alpha'].astype(np.float32)
    h, w = a.shape
    th = float(rng.uniform(0, 2 * np.pi))
    width = float(rng.uniform(0.01, 0.03)) * ctx['side']
    M = np.float32([[1, 0, -np.cos(th) * width], [0, 1, -np.sin(th) * width]])
    rim = np.clip(a - cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0), 0, 1)  # inside, light side outside
    rim = cv2.GaussianBlur(rim, (0, 0), max(0.3 * width, 0.8))
    k = float(rng.uniform(0.6, 0.95)) * rim[..., None]
    x = rgb.astype(np.float32)
    return u8(x + (_light_colour(rng) - x) * k)


def shade(rgb, ctx, rng):
    """A smooth light-to-shadow ramp across the character (random direction, strength, optional cool shadow tint)."""
    a = ctx['alpha']
    h, w = a.shape
    th = float(rng.uniform(0, 2 * np.pi))
    ys, xs = np.nonzero(a > 0.5)
    if len(xs) == 0:
        return rgb
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    p = X * np.cos(th) + Y * np.sin(th)
    pm = xs * np.cos(th) + ys * np.sin(th)
    t = np.clip((p - pm.min()) / max(pm.max() - pm.min(), 1.0), 0, 1)
    c0 = float(rng.uniform(0.25, 0.75))  # where the ramp is centred
    sharp = float(rng.uniform(3, 12))
    t = 1 / (1 + np.exp(-sharp * (t - c0)))  # 0 lit side, 1 shadow side
    s = float(rng.uniform(0.2, 0.55))
    f = 1 - s * t
    tint = rng.uniform(0.85, 1.0, 3).astype(np.float32) if rng.random() < 0.5 else np.ones(3, np.float32)
    tint[2] = 1.0  # cool (blue kept) shadow
    out = rgb.astype(np.float32) * (f[..., None] * (1 - t[..., None] + t[..., None] * tint))
    lift = float(rng.uniform(0.0, 0.12))  # slight highlight on the lit side
    out = out + (255 - out) * lift * (1 - t[..., None])
    return u8(out)


# ---------------------------------------------------------------- back under the matte, low resolution
def unpremix(styled, alpha):
    """RGB to store under the original alpha so that compositing on white gives back `styled` (clipped where alpha is
    small): c = (styled - (1 - a) * 255) / a."""
    a = alpha[..., None].astype(np.float32)
    c = (styled.astype(np.float32) - (1 - a) * 255) / np.maximum(a, 1e-3)
    return np.where(a > 0.02, u8(c), styled)


def recomposite(styled, alpha):
    """The styled image under the original alpha matte (unpremix), composited back on white."""
    c = unpremix(styled, alpha).astype(np.float32)
    a = alpha[..., None].astype(np.float32)
    return u8(c * a + 255 * (1 - a))


def lowres(rgb, side, rng, lo, hi):
    """The crop down to a side of lo..hi px (area / bilinear / nearest) and back (bicubic / bilinear)."""
    h, w = rgb.shape[:2]
    s = float(rng.uniform(lo, hi)) / side
    down = (cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST)[int(rng.choice(3, p=[0.6, 0.3, 0.1]))]
    up = (cv2.INTER_CUBIC, cv2.INTER_LINEAR)[int(rng.integers(2))]
    small = cv2.resize(rgb, (max(8, round(w * s)), max(8, round(h * s))), interpolation=down)
    return cv2.resize(small, (w, h), interpolation=up)
