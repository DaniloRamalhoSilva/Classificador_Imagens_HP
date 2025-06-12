from pathlib import Path
from PIL import Image


def convert_webp_to_png(directory: str) -> None:
    """Convert all .webp images under ``directory`` to .png, replacing the originals."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"{directory} is not a valid directory")
    for webp_path in root.rglob("*.webp"):
        try:
            with Image.open(webp_path) as img:
                png_path = webp_path.with_suffix('.png')
                img.save(png_path, 'PNG')
            webp_path.unlink()
        except Exception as exc:
            print(f"Erro ao converter {webp_path}: {exc}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python convert_webp_to_png.py <diretorio>")
        sys.exit(1)

    convert_webp_to_png(sys.argv[1])
