import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_png(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext == ".png"


def convert_png_to_tif(src_path: str, dst_path: str, dpi=(300, 300), compression: str = "tiff_deflate") -> bool:
    try:
        with Image.open(src_path) as img:
            # Remove alpha channel if present (TIFF RGB)
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img, mask=img.split()[1])
                img = background
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # Try save with compression; fallback to no compression
            try:
                img.save(dst_path, format="TIFF", compression=compression, dpi=dpi)
            except Exception:
                img.save(dst_path, format="TIFF", dpi=dpi)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to convert '{src_path}': {e}")
        return False


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(src_dir, "tif")

    ensure_dir(out_dir)

    png_files = [f for f in os.listdir(src_dir) if is_png(f)]
    if not png_files:
        print("[INFO] No PNG files found in:", src_dir)
        return

    total = len(png_files)
    success = 0
    for fname in png_files:
        src_path = os.path.join(src_dir, fname)
        base, _ = os.path.splitext(fname)
        dst_path = os.path.join(out_dir, f"{base}.tif")
        if convert_png_to_tif(src_path, dst_path):
            success += 1
            print(f"[OK] {fname} -> tif/{base}.tif")
        else:
            print(f"[FAIL] {fname}")

    print(f"[SUMMARY] Converted {success}/{total} PNG files to TIF in 'tif' folder.")


if __name__ == "__main__":
    main()