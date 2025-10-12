import sys
from pathlib import Path
import os
# src のパスを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import matplotlib.pyplot as plt
import base_utils.plotlib as bpl

# 作業ディレクトリをスクリプトの場所に変更
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# データ読み込み
data_file = "sample_cooling_file.txt"
data_list: list[bpl.XYData] = bpl.load_jasco_data(data_file)

# プロット準備
fig, ax = plt.subplots(figsize=(6, 4))

# 軸やタイトルなどのテキストを消す
bpl.remove_all_text(ax)

# 系列ごとの描画
for i, data in enumerate(data_list):
    if i == 0:
        # 最初の系列は赤
        color = bpl.Color.single_color("red")
        linestyle = "-"  # 実線
        linewidth = 1.5
    elif i == len(data_list) - 1:
        # 最後の系列は青
        color = bpl.Color.single_color("blue")
        linestyle = "-"  # 実線
        linewidth = 1.5
    else:
        # 中間系列は細い破線
        color = bpl.Color.single_color("gray")
        linestyle = "--"
        linewidth = 0.8

    plot_option = bpl.PlotOption(color=color, linestyle=linestyle, linewidth=linewidth, marker=None, markersize=0)
    bpl.plot_simple(ax, data, plot_option=plot_option)

# SVGで保存
output_svg = "sample_cooling_file_custom.svg"
fig.savefig(output_svg, format="svg")
plt.close(fig)

print(f"SVGファイルを保存しました: {output_svg}")
