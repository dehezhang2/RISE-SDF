#!/usr/bin/env python3
"""Build the media of the slides (talk.html) into image/talk/.

usage: tools/build_media.py <arxiv-source-dir>

GIFs of this page -> H.264 MP4 + JPEG poster (image/talk/video/), since the GIFs are
8-41 MB each. Paper figures from the arXiv source (https://arxiv.org/e-print/2409.20140)
-> resized JPEG/PNG under image/talk/. Needs ffmpeg and Pillow.
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image

PAGE = Path(__file__).resolve().parents[1]
FIG = Path(sys.argv[1]) / "figures"
OUT = PAGE / "image/talk"
(OUT / "video").mkdir(parents=True, exist_ok=True)

# ---- videos ----
for name, width in [("teaser", 1600), ("relight_helmet", 1600), ("relighting", 960), ("material", 1600)]:
    src, mp4 = PAGE / f"image/{name}.gif", OUT / f"video/{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", f"fps=10,scale={width}:-2:flags=lanczos,format=yuv420p",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-movflags", "+faststart",
                    "-an", str(mp4)], check=True)
    im = Image.open(src)
    im.seek(0)
    im = im.convert("RGB")
    im.thumbnail((width, width))
    im.save(OUT / f"video/{name}.jpg", quality=86)


def save(src, dst, width, png=False):
    im = Image.open(FIG / src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    dst = OUT / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if png:
        im.save(dst.with_suffix(".png"), optimize=True)
    else:
        im.save(dst.with_suffix(".jpg"), quality=86)


# ---- paper figures ----
save("limitation/appendix_limitatoin.png", "fig/limitation", 1400)

relight = {"toaster": 0, "helmet": 0, "musclecar": 3}
for s, k in relight.items():
    for m in ["gt", "ours", "ndrmc", "gs", "nmf", "envidr", "nero"]:
        save(f"figure_only/relight/{s}_{m}_relight_{k}.jpeg", f"cmp/relight_{s}_{m}", 640)
for s in ["toaster", "helmet", "musclecar"]:
    for m in ["gt", "ours", "ndr", "ndrmc", "nmf", "nero"]:
        save(f"figure_only/material/{s}_{m}_material.jpeg", f"cmp/material_{s}_{m}", 760)
for s in ["coffee", "musclecar"]:
    for m in ["gt", "ours", "ndrmc", "gs", "nmf", "envidr", "nero"]:
        save(f"figure_only/normal/{s}_{m}_normal.jpeg", f"cmp/normal_{s}_{m}", 560)
# information-sharing ablation (toaster): gt, ours (15k), w/o sharing at 5k (joint1) and 15k (joint)
for m in ["gt", "ours", "joint1", "joint"]:
    for q in ["albedo", "roughness", "normal"]:
        save(f"info_share/toaster_{m}_{q}.jpeg", f"share/{m}_{q}", 520)
# indirect-illumination ablation (toaster, fireplace map): full frame and zoomed crop
for m in ["gt", "ours", "ind", "second"]:
    save(f"indirect_ablation/toaster_{m}_relight_3.jpeg", f"ind/{m}", 640)
    save(f"indirect_ablation/toaster_{m}_relight_3_l.jpeg", f"ind/{m}_crop", 520)
# Monte Carlo vs split-sum
for s, gt in [("helmet", "r_0_bridge_d"), ("musclecar", "r_0_city_d")]:
    save(f"split_vs_mc/{gt}.jpeg", f"mc/{s}_gt", 520)
    for m in ["splitsum", "mc_128", "mc_256"]:
        save(f"split_vs_mc/{s}_{m}_c.jpeg", f"mc/{s}_{m}", 360)
        save(f"split_vs_mc/{s}_{m}_error_c.jpeg", f"mc/{s}_{m}_err", 360)
# Glossy-Real (NeRO real captures)
env = {"bear": "sunset_wenv", "bunny": "sunset_wenv", "coral": "fireplace_wenv", "maneki": "bridge_wenv", "vase": "courtyard_wenev"}
for o, e in env.items():
    save(f"glossy_real/glossy_real_nvs/{o}_nvs.jpeg", f"real/{o}_nvs", 520)
    save(f"glossy_real/glossy_real_mesh_render/{o}.jpeg", f"real/{o}_mesh", 520)
    save(f"glossy_real/glossy_real_relight/{o}_{e}.jpeg", f"real/{o}_relight", 520)

total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
print(f"image/talk: {sum(1 for f in OUT.rglob('*') if f.is_file())} files, {total / 1e6:.1f} MB")
