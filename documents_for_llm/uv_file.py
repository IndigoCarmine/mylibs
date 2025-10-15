import sys
from pathlib import Path

# src のパスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import base_utils.plotlib as bpl

import ref_for_llm.deco as deco

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    text =  deco.generate_prompt("なし(Baseutilなど直接importしなさい)", "UV spectraをよみこみ描画するライブラリです。", "free")
    with open("uv_file.txt", "w", encoding="utf-8") as f:
        f.write(text)

    md = deco.generate_prompt_markdown("なし(Baseutilなど直接importしなさい)", "UV spectraをよみこみ描画するライブラリです。", "free")
    with open("uv_file.md", "w", encoding="utf-8") as f:
        f.write(md)
