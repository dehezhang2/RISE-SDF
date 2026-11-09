#!/usr/bin/env python3
"""Read the result tables of the RISE-SDF arXiv source and write them into the deck.

usage: tools/extract_tables.py <arxiv-source-dir>

The arXiv source (https://arxiv.org/e-print/2409.20140, unpacked) holds the tables as
LaTeX under table/. Their numbers replace the block between `/*DATA*/` and
`/*END DATA*/` in tools/talk.html, so no number in the deck is typed by hand.
"""
import json
import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) / "table"


def cells(line):
    """Split one LaTeX table row into clean cell strings."""
    line = re.sub(r"\\cellcolor\{[^}]*\}", "", line)
    line = re.sub(r"\\cite\{[^}]*\}|~", "", line)
    line = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", line)
    line = line.replace("\\\\", "").replace("\\hline", "")
    return [c.strip() for c in line.split("&")]


def num(c):
    c = c.replace("$>$", "").replace("hrs", "").replace("*", "").strip()
    try:
        return float(c)
    except ValueError:
        return None


def rows(name):
    """Body rows of a tabular; a row may span several source lines and ends at \\\\."""
    text = "\n".join(l.split("%")[0] for l in (SRC / name).read_text().splitlines())
    body = text.split("\\midrule" if "\\midrule" in text else "\\toprule", 1)[1]
    body = body.split("\\bottomrule")[0].split("\\end{tabular}")[0]
    out = []
    for row in re.split(r"\\\\", body):
        row = " ".join(row.split())
        if "&" not in row or "PSNR" in row or "Method" in row:
            continue
        out.append(cells(row))
    return out


# Main comparison on the shiny synthetic set: method, PB, E2ER, normal MAE, roughness PSNR,
# albedo PSNR/SSIM/LPIPS, NVS PSNR/SSIM/LPIPS, relight PSNR/SSIM/LPIPS, training time.
main = []
for c in rows("shiny_inv_cmp.tex"):
    c = [x for x in c if x != ""]
    name = c[0]
    pb, e2e = c[1] == "\\cmark", c[2] == "\\cmark"
    vals = [num(x) if x != "-" else None for x in c[3:14]]
    main.append({"m": name, "pb": pb, "e2e": e2e, "v": vals, "t": c[14].replace("$>$", ">").strip()})

# Per-scene results: scene -> method -> [MAE, rough, alb P/S/L, nvs P/S/L, rel P/S/L].
scene = {}
cur = None
for c in rows("full_result.tex"):
    c = [x for x in c]
    if c[0]:
        cur = c[0]
    vals = [x for x in c[1:] if x != ""]
    name, vals = vals[0], [num(x) if x != "-" else None for x in vals[1:]]
    scene.setdefault(cur, {})[name] = vals

# Glossy-Real Chamfer distance.
real = {}
lines = (SRC / "glossy_real.tex").read_text().splitlines()
head = None
for line in lines:
    if "&" not in line:
        continue
    c = cells(line.split("%")[0])
    if head is None:
        head = [x.replace("$^*$", "").strip() for x in c[1:]]
        continue
    real[c[0]] = [num(x) for x in c[1:]]

# TensoIR (diffuse) averages: method -> [MAE, alb P/S/L, nvs P/S/L, rel P/S/L].
diffuse = {}
cur = None
for c in rows("diffuse_result.tex"):
    if c[0]:
        cur = c[0]
    vals = [x for x in c[1:] if x != ""]
    if cur.startswith("Avg"):
        diffuse[vals[0]] = [num(x) for x in vals[1:]]

# Relighting runtime (fps).
fps = {}
for line in (SRC / "runtime.tex").read_text().splitlines():
    if "&" in line and "NeRO" not in line:
        c = cells(line)
        fps[c[0]] = [num(c[1]), num(c[2])]

data = {"main": main, "scene": scene, "real": {"methods": head, "rows": real}, "diffuse": diffuse, "fps": fps}
block = "/*DATA*/\nconst D=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n/*END DATA*/"
talk = PAGE / "tools/talk.html"
src = talk.read_text()
new, n = re.subn(r"/\*DATA\*/.*?/\*END DATA\*/", lambda m: block, src, flags=re.S)
if n != 1:
    raise SystemExit("tools/talk.html: expected one /*DATA*/ ... /*END DATA*/ block, found %d" % n)
talk.write_text(new)
print(json.dumps(data, indent=1)[:4000])
