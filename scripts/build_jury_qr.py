"""Create standard QR assets for AGMEN's public jury page (Pillow + reportlab)."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageDraw, ImageFont
from reportlab.graphics.barcode.qr import QrCodeWidget


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts") / ("segoeuib.ttf" if bold else "segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def build(url: str, output: Path):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/demo"):
        raise ValueError("Use a public HTTPS /demo/ URL without credentials or query parameters")
    output.mkdir(parents=True, exist_ok=True)
    code = QrCodeWidget(url, barLevel="M")
    code.qr.make()
    modules = code.qr.modules
    quiet, scale = 4, 30
    size = len(modules) + quiet * 2
    qr = Image.new("RGB", (size * scale, size * scale), "white")
    draw = ImageDraw.Draw(qr)
    paths = []
    for y, row in enumerate(modules):
        for x, black in enumerate(row):
            if black:
                left, top = x + quiet, y + quiet
                draw.rectangle((left * scale, top * scale, (left + 1) * scale - 1, (top + 1) * scale - 1), fill="black")
                paths.append(f"M{left},{top}h1v1h-1z")
    qr.save(output / "AGMEN-demo-QR.png")
    (output / "AGMEN-demo-QR.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size*scale}" height="{size*scale}" viewBox="0 0 {size} {size}">'
        f'<title>AGMEN jury demo: {url}</title><rect width="{size}" height="{size}" fill="white"/>'
        f'<path d="{"".join(paths)}" fill="black" shape-rendering="crispEdges"/></svg>\n', encoding="utf-8")
    card = Image.new("RGB", (1500, 1900), "#f4f1ea")
    canvas = ImageDraw.Draw(card)
    ink, olive = "#292a25", "#526137"

    def centered(text, y, size, color=ink, bold=False):
        canvas.text((750, y), text, font=font(size, bold), fill=color, anchor="mt")

    centered("AGMEN", 85, 100, bold=True)
    centered("Демо для жюри", 225, 55, bold=True)
    centered("Сканируйте камерой телефона", 320, 32, "#606056")
    card.paste(qr, ((1500 - qr.width) // 2, 410))
    centered("Без регистрации и API-ключа", 1690, 35, olive, bold=True)
    centered(parsed.netloc + "/demo/", 1770, 30)
    card.save(output / "AGMEN-jury-card.png")
    (output / "demo-url.txt").write_text(url + "\n", encoding="utf-8")
    print(f"QR URL: {url}\nOutput: {output.resolve()}\nQR modules: {len(modules)}; quiet zone: {quiet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "jury")
    options = parser.parse_args()
    build(options.url, options.output)
