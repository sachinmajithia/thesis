#!/usr/bin/env python3
"""
Convert one or more ID card photos into a single PDF that looks like a
flatbed-scanned document (deskewed, high-contrast/grayscale, paper-white
page background, light scan grain).

Usage:
    python id_card_to_scanned_pdf.py front.jpg back.jpg -o id_card_scan.pdf
    python id_card_to_scanned_pdf.py front.jpg back.jpg --same-page
    python id_card_to_scanned_pdf.py *.jpg --color --page-size letter
"""

import argparse
import io
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PAGE_SIZES_MM = {
    "a4": (210, 297),
    "letter": (215.9, 279.4),
}


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
    if angle == 0:
        return img
    return img.rotate(
        angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC
    )


def apply_scan_look(
    img: Image.Image,
    grayscale: bool = True,
    jitter_angle: float = 1.0,
    contrast: float = 1.25,
    brightness: float = 1.08,
    sharpen: bool = True,
    grain: float = 5.0,
) -> Image.Image:
    """Process a single ID card photo to resemble a scanned card."""
    processed = img

    angle = random.uniform(-jitter_angle, jitter_angle) if jitter_angle else 0
    processed = deskew(processed, angle)

    if grayscale:
        processed = ImageOps.grayscale(processed).convert("RGB")

    processed = ImageEnhance.Contrast(processed).enhance(contrast)
    processed = ImageEnhance.Brightness(processed).enhance(brightness)

    if sharpen:
        processed = processed.filter(ImageFilter.UnsharpMask(radius=2, percent=60))

    if grain:
        processed = add_scan_grain(processed, amount=grain)

    return processed


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


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

    page = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))

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
        page.paste(resized, (x, y_center))
        y += slot_h + gap

    return page


def build_pdf(
    input_paths,
    output_path: str,
    dpi: int = 300,
    page_size: str = "a4",
    grayscale: bool = True,
    same_page: bool = False,
    jitter_angle: float = 1.0,
    grain: float = 5.0,
):
    if page_size not in PAGE_SIZES_MM:
        raise ValueError(f"Unknown page size '{page_size}'. Choose from {list(PAGE_SIZES_MM)}")
    page_size_mm = PAGE_SIZES_MM[page_size]

    cards = [
        apply_scan_look(
            load_image(path),
            grayscale=grayscale,
            jitter_angle=jitter_angle,
            grain=grain,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Path(s) to ID card image(s), e.g. front.jpg back.jpg")
    parser.add_argument("-o", "--output", default="id_card_scan.pdf", help="Output PDF path")
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution in DPI (default: 300)")
    parser.add_argument("--page-size", choices=list(PAGE_SIZES_MM), default="a4", help="Output page size")
    parser.add_argument("--color", action="store_true", help="Keep color instead of converting to grayscale")
    parser.add_argument(
        "--same-page",
        action="store_true",
        help="Place all images (e.g. front and back) on a single page instead of one page each",
    )
    parser.add_argument(
        "--jitter", type=float, default=1.0, help="Max random skew angle in degrees to mimic a hand-placed scan (default: 1.0, 0 to disable)"
    )
    parser.add_argument("--grain", type=float, default=5.0, help="Scan grain/noise intensity (default: 5.0, 0 to disable)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible skew/grain")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    build_pdf(
        input_paths=args.images,
        output_path=args.output,
        dpi=args.dpi,
        page_size=args.page_size,
        grayscale=not args.color,
        same_page=args.same_page,
        jitter_angle=args.jitter,
        grain=args.grain,
    )
    print(f"Saved scanned-style PDF to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
