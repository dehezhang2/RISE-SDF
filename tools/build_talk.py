#!/usr/bin/env python3
"""Build talk.html, the interactive slides of the RISE-SDF talk.

tools/talk.html is the deck in claude.ai Artifact form (no document skeleton), so the
same file can be previewed as an Artifact. This script wraps it in a document. Media
paths (image/...) are already relative to the page root, so they are only checked: a
referenced file that does not exist stops the build. Numbers come from
tools/extract_tables.py, media from tools/build_media.py.
"""
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1]
src = (PAGE / "tools/talk.html").read_text()

refs = set(re.findall(r"""["'](image/[A-Za-z0-9_./-]+\.(?:jpg|png|mp4))["']""", src))
# paths the deck assembles in its script, e.g. image/talk/cmp/${out}_${sc}_${m}.jpg
for out, scenes, methods in [("relight", ["toaster", "helmet", "musclecar"], ["gt", "ours", "nero", "envidr", "nmf", "ndrmc", "gs"]),
                             ("material", ["toaster", "helmet", "musclecar"], ["gt", "ours", "nero", "nmf", "ndrmc", "ndr"]),
                             ("normal", ["coffee", "musclecar"], ["gt", "ours", "nero", "envidr", "nmf", "ndrmc", "gs"])]:
    refs |= {f"image/talk/cmp/{out}_{s}_{m}.jpg" for s in scenes for m in methods}
for s in ["helmet", "musclecar"]:
    refs |= {f"image/talk/mc/{s}_gt.jpg"} | {f"image/talk/mc/{s}_{m}{e}.jpg" for m in ["splitsum", "mc_128", "mc_256"] for e in ["", "_err"]}
for o in ["bear", "bunny", "coral", "maneki", "vase"]:
    refs |= {f"image/talk/real/{o}_{k}.jpg" for k in ["nvs", "mesh", "relight"]}
for v in ["relighting", "relight_helmet", "material", "teaser"]:
    refs |= {f"image/talk/video/{v}.mp4", f"image/talk/video/{v}.jpg"}
missing = sorted(r for r in refs if not (PAGE / r).is_file())
if missing:
    raise SystemExit("talk.html: missing assets: " + ", ".join(missing))
if "/*DATA*/\n/*END DATA*/" in src:
    raise SystemExit("talk.html: data block is empty; run tools/extract_tables.py first")

# the deck's title, fonts, MathJax and styles come before its first element
head, sep, body = src.partition('<div id="viewport">')
if not sep:
    raise SystemExit('talk.html: <div id="viewport"> not found in tools/talk.html')
doc = ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       '<style>html,body{margin:0}img{max-width:100%}</style>\n'
       + head.strip() + '\n</head>\n<body>\n' + sep + body.strip() + '\n</body>\n</html>\n')
(PAGE / "talk.html").write_text(doc)
print("talk.html", len(doc) // 1000, "kB,", len(refs), "assets checked")
