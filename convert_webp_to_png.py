from pathlib import Path
from PIL import Image


def convert_webp_to_png(directory: str) -> None:
    """Convert all image files that are **not** PNG under ``directory`` to PNG.

    The conversion walks recursively through ``directory`` and replaces the
    original files when the conversion succeeds.
    """
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"{directory} is not a valid directory")

    for img_path in root.rglob("*"):
        # Skip directories and already-converted images
        if not img_path.is_file() or img_path.suffix.lower() == ".png":
            continue

        try:
            with Image.open(img_path) as img:
                png_path = img_path.with_suffix(".png")
                img.save(png_path, "PNG")
            img_path.unlink()
        except Exception as exc:
            print(f"Erro ao converter {img_path}: {exc}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python convert_webp_to_png.py <diretorio>")
        sys.exit(1)

    convert_webp_to_png(sys.argv[1])
