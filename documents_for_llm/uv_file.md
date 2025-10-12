# Library: なし(Baseutilなど直接importしなさい)

**Description:** UV spectraをよみこみ描画するライブラリです。

**Usage Policy:** free

## Context
- **Role:** ユーザー要望を自然言語からPythonコードに変換するアシスタント
- **Rules:**
  1. ユーザーは自然言語で要望を述べる
  2. あなたは要望を分析し、関数やクラスメソッドを組み合わせてPythonコードを生成する
  3. 不明確な要望は質問で確認してからコードを生成する

## Classes
### Class: Color
**Description:** Manages colors for different plotting styles.
Allows defining a set of colors that adapt to various output contexts (e.g., paper, presentation).

#### Methods
- `__init__` — None (Signature: `(self, colordict: dict[base_utils.plotlib.Style, str | None])`) None
- `single_color` — Creates a Color instance where the same color is used across all styles. (Signature: `(cls, color: str) -> 'Color'`) Creates a Color instance where the same color is used across all styles.
- `single_color` — Creates a Color instance where the same color is used across all styles. (Signature: `(cls, color: str) -> 'Color'`) Creates a Color instance where the same color is used across all styles.

### Class: XYData
**Description:** Represents a set of X-Y data points with associated labels and an optional title.
Provides methods for data manipulation such as renaming, shifting, scaling, and normalization.

#### Methods
- `clip` — Returns a new XYData object with data clipped to a specified X range. (Signature: `(self, xmin: float, xmax: float) -> 'XYData'`) Returns a new XYData object with data clipped to a specified X range.
- `get_y_at_nearest_x` — Returns the Y value corresponding to the X value nearest to the given 'x'. (Signature: `(self, x: float) -> float`) Returns the Y value corresponding to the X value nearest to the given 'x'.
- `get_y_at_range` — Returns the Y values within a specified X range. (Signature: `(self, xmin: float = -inf, xmax: float = inf) -> numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[numpy.float64]]`) Returns the Y values within a specified X range.
- `normalize` — Returns a new XYData object with Y values normalized to a range of 0 to 1. (Signature: `(self) -> 'XYData'`) Returns a new XYData object with Y values normalized to a range of 0 to 1.
- `rename_labels` — Returns a new XYData object with updated DataLabel. (Signature: `(self, label: base_utils.plotlib.DataLabel) -> 'XYData'`) Returns a new XYData object with updated DataLabel.
- `rename_title` — Returns a new XYData object with an updated title. (Signature: `(self, title: str) -> 'XYData'`) Returns a new XYData object with an updated title.
- `xscale` — Returns a new XYData object with X values scaled by a given factor. (Signature: `(self, scale: float) -> 'XYData'`) Returns a new XYData object with X values scaled by a given factor.
- `xshift` — Returns a new XYData object with X values shifted by a given amount. (Signature: `(self, shift: float) -> 'XYData'`) Returns a new XYData object with X values shifted by a given amount.
- `yscale` — Returns a new XYData object with Y values scaled by a given factor. (Signature: `(self, scale: float) -> 'XYData'`) Returns a new XYData object with Y values scaled by a given factor.
- `yshift` — Returns a new XYData object with Y values shifted by a given amount. (Signature: `(self, shift: float) -> 'XYData'`) Returns a new XYData object with Y values shifted by a given amount.

### Class: PlotOption
**Description:** Defines options for plotting individual data series, such as color, marker, and line style.


## Functions
### Function: load_xvgdata
**Signature:** `(path: str) -> base_utils.plotlib.XYData`
**Description:** 


### Function: convert_from_df
**Signature:** `(df: pandas.core.frame.DataFrame, label: base_utils.plotlib.DataLabel) -> list[base_utils.plotlib.XYData]`
**Description:** 


### Function: load_jasco_data
**Signature:** `(p: str) -> list[base_utils.plotlib.XYData]`
**Description:** 


### Function: load_dat
**Signature:** `(path: str, x_label: str = '', y_label: str = '', x_unit: str = '', y_unit: str = '', has_header: bool = False) -> base_utils.plotlib.XYData`
**Description:** 


### Function: load_csv
**Signature:** `(path: str) -> list[base_utils.plotlib.XYData]`
**Description:** 


### Function: slice_data
**Signature:** `(data: list[base_utils.plotlib.XYData], x_value: float, new_x_values: list[float]) -> base_utils.plotlib.XYData`
**Description:** 


### Function: plot_simple
**Signature:** `(ax: matplotlib.axes._axes.Axes, data: base_utils.plotlib.XYData, plot_option: base_utils.plotlib.PlotOption = PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), figure_option: base_utils.plotlib.FigureOption = FigureOption(size=(4, 3), plot_option=PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), is_white_background=True, plot_option_override=False), style: base_utils.plotlib.Style = <Style.paper: 0>) -> None`
**Description:** 


### Function: plot1d
**Signature:** `(ax: matplotlib.axes._axes.Axes, data: base_utils.plotlib.XYData, style: base_utils.plotlib.Style, plot_option: base_utils.plotlib.PlotOption = PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), figure_option: base_utils.plotlib.FigureOption = FigureOption(size=(4, 3), plot_option=PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), is_white_background=True, plot_option_override=False), range: Optional[tuple[float, float]] = None) -> None`
**Description:** 


### Function: plot2d
**Signature:** `(ax: matplotlib.axes._axes.Axes, data: list[base_utils.plotlib.XYData], style: base_utils.plotlib.Style, plot_option: base_utils.plotlib.PlotOption = PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), figure_option: base_utils.plotlib.FigureOption = FigureOption(size=(4, 3), plot_option=PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), is_white_background=True, plot_option_override=False), xrange: Optional[tuple[float, float]] = None, yrange: Optional[tuple[float, float]] = None) -> None`
**Description:** 


### Function: for_white_background
**Signature:** `(ax: matplotlib.axes._axes.Axes) -> None`
**Description:** 


### Function: for_black_background
**Signature:** `(ax: matplotlib.axes._axes.Axes) -> None`
**Description:** 


### Function: remove_all_text
**Signature:** `(ax: matplotlib.axes._axes.Axes) -> None`
**Description:** 


## Enums
### Enum: Style
**Description:** Defines different plotting styles for various output formats (e.g., paper, presentation).

**Values:**
- `paper`
- `presentation_black`
- `presentation_white`
- `poster_black_highcontrast`

### Enum: DataLabels
**Description:** Predefined DataLabel instances for common spectroscopic data types.

**Values:**
- `UV`
- `FL`
- `CD`
- `IR`
- `XRD`

