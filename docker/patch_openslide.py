"""Backport OpenSlide's partial-bottom-tile fix without upgrading the 4.0.0 ABI.

Derived from OpenSlide 4.0.1, src/openslide-decode-tiff.c (LGPL-2.1-only):
https://github.com/openslide/openslide/security/advisories/GHSA-f734-jv98-5677
https://github.com/openslide/openslide/blob/v4.0.1/src/openslide-decode-tiff.c
Ubuntu's libtiff updates include the behavior change described by that advisory.
"""

import sys
from pathlib import Path


def patch(path: Path) -> None:
    original = """    return tiff_read_region(tiff, dest,
                            tile_col * tiffl->tile_w, tile_row * tiffl->tile_h,
                            tiffl->tile_w, tiffl->tile_h, err);"""
    replacement = """    // Backport: limit the raster height for libtiff's partial bottom tiles.
    uint32_t clipped_h =
      MIN(tiffl->tile_h, MAX(tiffl->image_h - tile_row * tiffl->tile_h, 0));
    return tiff_read_region(tiff, dest,
                            tile_col * tiffl->tile_w, tile_row * tiffl->tile_h,
                            tiffl->tile_w, clipped_h, err);"""
    source = path.read_text(encoding="utf-8")
    if source.count(original) != 1:
        raise ValueError("OpenSlide source changed; review the partial-tile backport")
    path.write_text(source.replace(original, replacement), encoding="utf-8")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
