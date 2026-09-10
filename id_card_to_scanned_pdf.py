#!/usr/bin/env python3
"""
Convert ID card photos into PDF(s) that look like a flatbed-scanned document
(deskewed, high-contrast/grayscale, paper-white page background, light scan
grain).

Usage:
    # Combine explicit files into one PDF
    python id_card_to_scanned_pdf.py front.jpg back.jpg -o id_card_scan.pdf
    python id_card_to_scanned_pdf.py front.jpg back.jpg --same-page
    python id_card_to_scanned_pdf.py *.jpg --grayscale --page-size letter
    python id_card_to_scanned_pdf.py front.jpg --scan-dpi 75  # softer/blurrier, low-DPI look

    # Batch mode: pick every image in a folder and write one PDF per image,
    # each PDF named after its source image (front.jpg -> front.pdf)
    python id_card_to_scanned_pdf.py ./id_cards
    python id_card_to_scanned_pdf.py ./id_cards --output-dir ./scanned_pdfs
"""

import argparse
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

PAGE_SIZES_MM = {
    "a4": (210, 297),
    "letter": (215.9, 279.4),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Resolution above which a card is treated as "sharp" when simulating a
# lower-DPI scan; --scan-dpi is expressed relative to this.
SHARP_REFERENCE_DPI = 300.0

# Shared tint both shadow effects blend toward, so the card's own drop
# shadow and the page-wide shading read as the same light source rather
# than two visually distinct effects.
SHADOW_COLOR = (40, 40, 40)

# Drop shadow behind the tilted card, randomized within these ranges (sized
# relative to the card's own width) so each card reads as a physical object
# placed on the page rather than a flat rotated cutout, with natural
# card-to-card variation instead of one identical shadow every time.
SHADOW_OFFSET_FRACTION_RANGE = (0.006, 0.018)
SHADOW_BLUR_FRACTION_RANGE = (0.015, 0.03)
SHADOW_OPACITY_RANGE = (50, 120)  # 0-255

# The shadow silhouette is a distorted, non-uniformly scaled, noisy version
# of the card's own outline rather than an exact rectangular copy of it, so
# it reads as an irregular soft shadow instead of a duplicated card shape.
# SHADOW_MAX_SPREAD_FRACTION caps how far that noise can push the shadow
# beyond the card's edge, so it stays a shadow hugging the card rather than
# a gradient sweeping across the page.
SHADOW_SCALE_RANGE = (0.95, 1.1)  # independent random x/y scale of the silhouette
SHADOW_NOISE_GRID = 14  # coarse grid resolution the noise is generated at
SHADOW_NOISE_AMPLITUDE = 70  # 0-255
SHADOW_MAX_SPREAD_FRACTION = 0.05
# All shadow-mask math (including the MaxFilter dilation) runs on a copy of
# the card downscaled to at most this width, keeping shadow generation fast
# regardless of the source photo's resolution.
SHADOW_MASK_WORK_SIZE = 400

# A separate, broader shading pass covering the whole page: soft, uneven
# patches of darkness (like an unevenly lit scanner bed or dirty glass),
# with the darkness amount varying randomly both across the page and from
# one page to the next.
PAGE_SHADING_BLOB_GRID = (7, 10)  # coarse (cols, rows) grid the blobs are generated at
PAGE_SHADING_BLUR_FRACTION = 0.06  # blob softness, relative to the page's longer side
PAGE_SHADING_INTENSITY_RANGE = (10, 70)  # 0-255 max darkness, randomized per page


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect camera orientation
    return img.convert("RGB")


def add_scan_grain(img: Image.Image, amount: float = 6.0) -> Image.Image:
    """Add subtle Gaussian noise so the page doesn't look like a raw digital photo."""
    arr = np.asarray(img).astype(np.int16)
    # One noise value per pixel, broadcast across channels, instead of one
    # per channel: ~3x less random-number generation at full photo
    # resolution, with no visible difference in the result.
    noise = np.random.normal(0, amount, arr.shape[:2] + (1,))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def deskew(img: Image.Image, angle: float) -> Image.Image:
    """Rotate the card, leaving the exposed corners transparent (RGBA) so the
    caller can composite a shadow and let the page background show through,
    instead of a flat white-filled cutout."""
    rgba = img.convert("RGBA")
    if angle == 0:
        return rgba
    return rgba.rotate(
        angle, expand=True, fillcolor=(0, 0, 0, 0), resample=Image.BICUBIC
    )


def random_tilt_angle(min_degrees: float, max_degrees: float) -> float:
    """Pick a random tilt magnitude in [min_degrees, max_degrees] and a random
    left/right direction. Returns 0 if the range is disabled (max <= 0)."""
    if max_degrees <= 0:
        return 0.0
    magnitude = random.uniform(min_degrees, max_degrees)
    return magnitude if random.random() < 0.5 else -magnitude


def simulate_low_dpi(img: Image.Image, scan_dpi: float, reference_dpi: float = SHARP_REFERENCE_DPI) -> Image.Image:
    """Soften the card the way a real low-resolution scan would: downsample
    to the apparent scan_dpi, then upsample back, so detail is genuinely
    lost rather than just blurred over."""
    if not scan_dpi or scan_dpi >= reference_dpi:
        return img
    scale = max(scan_dpi / reference_dpi, 0.05)
    small_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    small = img.resize(small_size, Image.BILINEAR)
    return small.resize(img.size, Image.BILINEAR)


def apply_scan_look(
    img: Image.Image,
    grayscale: bool = False,
    tilt_min: float = 3.0,
    tilt_max: float = 8.0,
    contrast: float = 1.25,
    brightness: float = 1.08,
    grain: float = 5.0,
    scan_dpi: float = 100.0,
) -> Image.Image:
    """Process a single ID card photo to resemble a scanned card. Returns an
    RGBA image already rotated by the chosen tilt angle, with transparent
    corners where the rotation exposed the page behind it."""
    processed = img.convert("RGB")

    if grayscale:
        processed = ImageOps.grayscale(processed).convert("RGB")

    processed = ImageEnhance.Contrast(processed).enhance(contrast)
    processed = ImageEnhance.Brightness(processed).enhance(brightness)

    if scan_dpi:
        processed = simulate_low_dpi(processed, scan_dpi)

    if grain:
        processed = add_scan_grain(processed, amount=grain)

    angle = random_tilt_angle(tilt_min, tilt_max)
    return deskew(processed, angle)


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def make_irregular_shadow_mask(alpha: Image.Image) -> Image.Image:
    """Turn the card's (possibly tilted) alpha silhouette into an irregular
    blob: scale it non-uniformly on x/y, then perturb it with coarse random
    noise, so the shadow reads as an uneven soft shape rather than a clean
    duplicate of the card's rectangle.

    All of this runs on a small downscaled copy of the mask (MaxFilter's
    rank filter is O(kernel_area) per pixel, so running it at full photo
    resolution with a wide kernel is extremely slow); the result is
    upsampled back to the card's actual size at the end."""
    work_w = min(alpha.width, SHADOW_MASK_WORK_SIZE)
    work_h = max(1, round(alpha.height * work_w / alpha.width))
    small_alpha = alpha.resize((work_w, work_h), Image.BILINEAR)

    scale_x = random.uniform(*SHADOW_SCALE_RANGE)
    scale_y = random.uniform(*SHADOW_SCALE_RANGE)
    scaled_w = max(1, int(work_w * scale_x))
    scaled_h = max(1, int(work_h * scale_y))
    scaled = small_alpha.resize((scaled_w, scaled_h), Image.BILINEAR)

    canvas = Image.new("L", (work_w, work_h), 0)
    canvas.paste(scaled, ((work_w - scaled_w) // 2, (work_h - scaled_h) // 2))

    grid = max(4, SHADOW_NOISE_GRID)
    small = canvas.resize((grid, grid), Image.BILINEAR)
    arr = np.asarray(small).astype(np.int16)
    noise = np.random.uniform(-SHADOW_NOISE_AMPLITUDE, SHADOW_NOISE_AMPLITUDE, arr.shape)
    noisy_small = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), mode="L")
    noisy = noisy_small.resize((work_w, work_h), Image.BILINEAR)

    # Bound the noisy shape to a modestly dilated version of the card's own
    # silhouette, so it can't sweep far past the card's edge.
    margin = max(3, int(work_w * SHADOW_MAX_SPREAD_FRACTION)) | 1
    allowed_extent = small_alpha.filter(ImageFilter.MaxFilter(margin))
    bounded = ImageChops.multiply(noisy, allowed_extent)

    return bounded.resize(alpha.size, Image.BILINEAR)


def paste_card_with_shadow(page: Image.Image, card: Image.Image, x: int, y: int) -> None:
    """Paste an RGBA (possibly tilted, transparent-cornered) card onto the
    page with a soft, irregularly-shaped drop shadow, so it reads as a card
    placed on a scanner rather than a flat rotated cutout. The shadow's
    shape, direction, offset, blur, and opacity are all randomized per card
    for natural variation."""
    alpha = card.split()[-1]
    shadow_mask = make_irregular_shadow_mask(alpha)

    offset_magnitude = random.uniform(*SHADOW_OFFSET_FRACTION_RANGE) * card.width
    direction = random.uniform(0, 2 * math.pi)
    dx = int(round(offset_magnitude * math.cos(direction)))
    dy = int(round(offset_magnitude * math.sin(direction)))

    blur_radius = max(1.0, random.uniform(*SHADOW_BLUR_FRACTION_RANGE) * card.width)
    opacity = random.uniform(*SHADOW_OPACITY_RANGE)

    shadow = Image.new("RGBA", card.size, SHADOW_COLOR + (0,))
    shadow.putalpha(shadow_mask.point(lambda a: int(a * opacity / 255)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    page.paste(shadow, (x + dx, y + dy), shadow)
    page.paste(card, (x, y), card)


def apply_page_shading(page: Image.Image) -> Image.Image:
    """Darken the whole page with soft, irregular patches whose intensity
    varies randomly across the sheet (and from page to page), mimicking an
    unevenly lit scanner bed rather than a perfectly uniform white sheet.

    Uses the same "blend toward SHADOW_COLOR through a soft alpha mask"
    approach as the card's own drop shadow (rather than a flat brightness
    subtraction), so the two read as one consistent light source instead of
    two visually distinct effects."""
    w, h = page.size
    cols, rows = PAGE_SHADING_BLOB_GRID

    noise = np.random.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
    blob_mask = Image.fromarray((noise * 255).astype(np.uint8), mode="L").resize((w, h), Image.BICUBIC)
    blob_mask = blob_mask.filter(ImageFilter.GaussianBlur(radius=max(w, h) * PAGE_SHADING_BLUR_FRACTION))

    max_intensity = random.uniform(*PAGE_SHADING_INTENSITY_RANGE)
    alpha_mask = blob_mask.point(lambda a: int(a * max_intensity / 255))

    tint = Image.new("RGB", page.size, SHADOW_COLOR)
    return Image.composite(tint, page.convert("RGB"), alpha_mask)


def compose_page(
    images,
    dpi: int,
    page_size_mm,
    margin_mm: float = 15.0,
    gap_mm: float = 8.0,
) -> Image.Image:
    """Place one or more processed card images (stacked vertically, centered)
    onto a white page sized like a real scanned sheet."""
    page_w = mm_to_px(page_size_mm[0], dpi)
    page_h = mm_to_px(page_size_mm[1], dpi)
    margin = mm_to_px(margin_mm, dpi)
    gap = mm_to_px(gap_mm, dpi)

    # Shade the blank page first, then composite the card and its own
    # shadow on top: wherever the page-wide shading is already darker, the
    # card's shadow blends into it instead of a flat wash being painted
    # over everything (card included) as a separate, disconnected step.
    page = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
    page = apply_page_shading(page)

    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin - gap * (len(images) - 1)
    slot_h = usable_h // len(images)

    y = margin
    for card in images:
        scale = min(usable_w / card.width, slot_h / card.height)
        new_size = (max(1, int(card.width * scale)), max(1, int(card.height * scale)))
        resized = card.resize(new_size, Image.LANCZOS)

        x = margin + (usable_w - new_size[0]) // 2
        y_center = y + (slot_h - new_size[1]) // 2
        paste_card_with_shadow(page, resized, x, y_center)
        y += slot_h + gap

    return page


def build_pdf(
    input_paths,
    output_path: str,
    dpi: int = 300,
    page_size: str = "a4",
    grayscale: bool = False,
    same_page: bool = False,
    tilt_min: float = 3.0,
    tilt_max: float = 8.0,
    grain: float = 5.0,
    scan_dpi: float = 100.0,
):
    if page_size not in PAGE_SIZES_MM:
        raise ValueError(f"Unknown page size '{page_size}'. Choose from {list(PAGE_SIZES_MM)}")
    page_size_mm = PAGE_SIZES_MM[page_size]

    cards = [
        apply_scan_look(
            load_image(path),
            grayscale=grayscale,
            tilt_min=tilt_min,
            tilt_max=tilt_max,
            grain=grain,
            scan_dpi=scan_dpi,
        )
        for path in input_paths
    ]

    if same_page:
        pages = [compose_page(cards, dpi=dpi, page_size_mm=page_size_mm)]
    else:
        pages = [compose_page([card], dpi=dpi, page_size_mm=page_size_mm) for card in cards]

    first, rest = pages[0], pages[1:]
    first.save(
        output_path,
        "PDF",
        resolution=float(dpi),
        save_all=True,
        append_images=rest,
    )


def list_images_in_dir(folder: str):
    entries = sorted(os.listdir(folder))
    return [
        os.path.join(folder, name)
        for name in entries
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]


def batch_convert_folder(
    input_dir: str,
    output_dir: str,
    dpi: int = 300,
    page_size: str = "a4",
    grayscale: bool = False,
    tilt_min: float = 3.0,
    tilt_max: float = 8.0,
    grain: float = 5.0,
    scan_dpi: float = 100.0,
):
    """Convert every image in input_dir into its own scanned-style PDF,
    named after the source image, written into output_dir."""
    image_paths = list_images_in_dir(input_dir)
    if not image_paths:
        raise ValueError(f"No image files found in '{input_dir}' (looked for {sorted(IMAGE_EXTENSIONS)})")

    os.makedirs(output_dir, exist_ok=True)

    output_paths = []
    for path in image_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(output_dir, f"{stem}.pdf")
        build_pdf(
            input_paths=[path],
            output_path=output_path,
            dpi=dpi,
            page_size=page_size,
            grayscale=grayscale,
            same_page=False,
            tilt_min=tilt_min,
            tilt_max=tilt_max,
            grain=grain,
            scan_dpi=scan_dpi,
        )
        output_paths.append(output_path)

    return output_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "images",
        nargs="+",
        help="Path(s) to ID card image(s) (front.jpg back.jpg ...), or a single folder path "
        "to batch-convert every image in it into its own same-named PDF",
    )
    parser.add_argument("-o", "--output", default="id_card_scan.pdf", help="Output PDF path (ignored in folder/batch mode)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder mode only: where to write the per-image PDFs (default: same folder as the input images)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in DPI (default: 300)")
    parser.add_argument("--page-size", choices=list(PAGE_SIZES_MM), default="a4", help="Output page size")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale instead of keeping color (default: color)")
    parser.add_argument(
        "--same-page",
        action="store_true",
        help="Place all images (e.g. front and back) on a single page instead of one page each",
    )
    parser.add_argument("--tilt-min", type=float, default=3.0, help="Minimum random tilt angle in degrees (default: 3.0)")
    parser.add_argument("--tilt-max", type=float, default=8.0, help="Maximum random tilt angle in degrees, left or right (default: 8.0, 0 to disable tilt)")
    parser.add_argument("--grain", type=float, default=5.0, help="Scan grain/noise intensity (default: 5.0, 0 to disable)")
    parser.add_argument(
        "--scan-dpi",
        type=float,
        default=100.0,
        help="Apparent scan resolution used to blur the card via a realistic downsample/upsample "
        "(default: 100.0; lower = blurrier, e.g. 75; raise toward 300 to sharpen, 0 to disable)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible tilt/grain")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if len(args.images) == 1 and os.path.isdir(args.images[0]):
        input_dir = args.images[0]
        output_dir = args.output_dir or input_dir
        output_paths = batch_convert_folder(
            input_dir=input_dir,
            output_dir=output_dir,
            dpi=args.dpi,
            page_size=args.page_size,
            grayscale=args.grayscale,
            tilt_min=args.tilt_min,
            tilt_max=args.tilt_max,
            grain=args.grain,
            scan_dpi=args.scan_dpi,
        )
        print(f"Saved {len(output_paths)} scanned-style PDF(s) to {output_dir}:")
        for path in output_paths:
            print(f"  {path}")
        return

    build_pdf(
        input_paths=args.images,
        output_path=args.output,
        dpi=args.dpi,
        page_size=args.page_size,
        grayscale=args.grayscale,
        same_page=args.same_page,
        tilt_min=args.tilt_min,
        tilt_max=args.tilt_max,
        grain=args.grain,
        scan_dpi=args.scan_dpi,
    )
    print(f"Saved scanned-style PDF to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
