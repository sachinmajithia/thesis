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
import os
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PAGE_SIZES_MM = {
    "a4": (210, 297),
    "letter": (215.9, 279.4),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Resolution above which a card is treated as "sharp" when simulating a
# lower-DPI scan; --scan-dpi is expressed relative to this.
SHARP_REFERENCE_DPI = 300.0

# Drop shadow behind the tilted card, sized relative to the card's own width,
# so the card reads as a physical object placed on the page rather than a
# flat rotated cutout.
SHADOW_OFFSET_FRACTION = 0.012
SHADOW_BLUR_FRACTION = 0.02
SHADOW_OPACITY = 90  # 0-255


def load_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect camera orientation
    return img.convert("RGB")


def add_scan_grain(img: Image.Image, amount: float = 6.0) -> Image.Image:
    """Add subtle Gaussian noise so the page doesn't look like a raw digital photo."""
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.normal(0, amount, arr.shape)
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
    tilt_min: float = 5.0,
    tilt_max: float = 10.0,
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


def paste_card_with_shadow(page: Image.Image, card: Image.Image, x: int, y: int) -> None:
    """Paste an RGBA (possibly tilted, transparent-cornered) card onto the
    page with a soft drop shadow, so it reads as a card placed on a scanner
    rather than a flat rotated cutout."""
    alpha = card.split()[-1]

    offset = max(1, int(card.width * SHADOW_OFFSET_FRACTION))
    blur_radius = max(1.0, card.width * SHADOW_BLUR_FRACTION)

    shadow = Image.new("RGBA", card.size, (40, 40, 40, 0))
    shadow.putalpha(alpha.point(lambda a: int(a * SHADOW_OPACITY / 255)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    page.paste(shadow, (x + offset, y + offset), shadow)
    page.paste(card, (x, y), card)


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

    page = Image.new("RGBA", (page_w, page_h), color=(255, 255, 255, 255))

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

    return page.convert("RGB")


def build_pdf(
    input_paths,
    output_path: str,
    dpi: int = 300,
    page_size: str = "a4",
    grayscale: bool = False,
    same_page: bool = False,
    tilt_min: float = 5.0,
    tilt_max: float = 10.0,
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
    tilt_min: float = 5.0,
    tilt_max: float = 10.0,
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
    parser.add_argument("--tilt-min", type=float, default=5.0, help="Minimum random tilt angle in degrees (default: 5.0)")
    parser.add_argument("--tilt-max", type=float, default=10.0, help="Maximum random tilt angle in degrees, left or right (default: 10.0, 0 to disable tilt)")
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
