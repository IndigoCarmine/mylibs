import os

import matplotlib.pyplot as plt
import base_utils.plotlib as bpl


# おまじない
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Load data
    cooling_data_file = r"sample_cooling_file.txt"
    data: list[bpl.XYData] = bpl.load_jasco_data(cooling_data_file)

    """
    XYDataはX軸とY軸のデータを保持するクラスです。
    Cooling_data_fileには、複数の温度における吸光度のデータが含まれているので、Listでデータが得られます。
    """

    # -------------BaseLine補正-------------
    # テストデータの分子は400nm以降に吸収がないので、そこを基準にします。
    new_data = []
    for single_data in data:
        baseline = single_data.get_y_at_range(
            xmin=400
        ).mean()  # 400nm以降の吸光度の平均値
        fixed_single_data = single_data.yshift(-baseline)  # 吸収度を基準にシフト
        new_data.append(fixed_single_data)

    data = new_data
    # -------------BaseLine補正　終わり-------------

    # Plot data (Matplotlibをシンプルに使う場合)
    # for d in data:
    #     ax.plot(d.X, d.Y)

    # Plot data (plotlibを使う場合)
    plot_option: bpl.PlotOption = (
        bpl.PlotOptions.paper
    )  # 論文用のプロットオプションを指定
    plot_option.linewidth = 0.5  # その内、線の太さを変更
    plot_option.linestyle = ":"  # その内、線のスタイルを変更
    bpl.plot2d(
        ax,
        data,
        style=bpl.Style.paper,
        plot_option=plot_option,
    )

    # 実際に使うとき用微調整
    ax.plot(data[0].X, data[0].Y, color="red")  # 高温側を赤色でプロット
    ax.plot(data[-1].X, data[-1].Y, color="blue")  # 低温側を青色でプロット
    ax.set_xlim(250, 450)  # X軸の範囲を指定
    ax.set_title("Spectra of cooling")  # タイトルを指定
    fig.set_size_inches(4, 3)  # グラフのサイズを指定

    # # Show plot
    # plt.show()

    # Save plot
    plt.savefig("sample_cooling.svg", transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    main()
