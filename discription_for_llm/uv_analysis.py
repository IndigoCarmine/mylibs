import ref_for_llm as rf

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import base_utils.plotlib


if __name__ == "__main__":
    json_data = rf.deco.generate_prompt(
        "mylib", "uv解析のためのライブラリ", "Free to use UV analysis library"
    )

    with open("discription_for_llm/uv_analysis.txt", "w", encoding="utf-8") as f:
        f.write(json_data)

    md = rf.deco.generate_prompt_markdown(
        "mylib", "uv解析のためのライブラリ", "Free to use UV analysis library"
    )
    with open("discription_for_llm/uv_analysis.md", "w", encoding="utf-8") as f:
        f.write(md)
