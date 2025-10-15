# Library: mylib

**Description:** uv解析のためのライブラリ

**Usage Policy:** Free to use UV analysis library

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

### Class: DataLabel
**Description:** Represents labels and units for X and Y axes of a plot.


### Class: XYData
**Description:** Represents a set of X-Y data points with associated labels and an optional title.
Provides methods for data manipulation such as renaming, shifting, scaling, and normalization.

#### Methods
- `clip` — Returns a new XYData object with data clipped to a specified X range. (Signature: `(self, xmin: float, xmax: float) -> 'XYData'`) Returns a new XYData object with data clipped to a specified X range.
- `get_y_at_nearest_x` — Returns the Y value corresponding to the X value nearest to the given 'x'. (Signature: `(self, x: float) -> float`) Returns the Y value corresponding to the X value nearest to the given 'x'.
- `get_y_at_range` — Returns the Y values within a specified X range. (Signature: `(self, xmin: float = -inf, xmax: float = inf) -> numpy.ndarray`) Returns the Y values within a specified X range.
- `normalize` — Returns a new XYData object with Y values normalized to a range of 0 to 1. (Signature: `(self) -> 'XYData'`) Returns a new XYData object with Y values normalized to a range of 0 to 1.
- `rename_labels` — Returns a new XYData object with updated DataLabel. (Signature: `(self, label: base_utils.plotlib.DataLabel) -> 'XYData'`) Returns a new XYData object with updated DataLabel.
- `rename_title` — Returns a new XYData object with an updated title. (Signature: `(self, title: str) -> 'XYData'`) Returns a new XYData object with an updated title.
- `xscale` — Returns a new XYData object with X values scaled by a given factor. (Signature: `(self, scale: float) -> 'XYData'`) Returns a new XYData object with X values scaled by a given factor.
- `xshift` — Returns a new XYData object with X values shifted by a given amount. (Signature: `(self, shift: float) -> 'XYData'`) Returns a new XYData object with X values shifted by a given amount.
- `yscale` — Returns a new XYData object with Y values scaled by a given factor. (Signature: `(self, scale: float) -> 'XYData'`) Returns a new XYData object with Y values scaled by a given factor.
- `yshift` — Returns a new XYData object with Y values shifted by a given amount. (Signature: `(self, shift: float) -> 'XYData'`) Returns a new XYData object with Y values shifted by a given amount.

### Class: PlotOption
**Description:** Defines options for plotting individual data series, such as color, marker, and line style.


### Class: PlotOptions
**Description:** Predefined PlotOption instances for common plotting scenarios.


### Class: FigureOption
**Description:** Defines options for the overall figure, including size, plot options, and background.


### Class: FigureOptions
**Description:** Predefined FigureOption instances for common figure setups.


## Functions
### Function: load_xvgdata
**Signature:** `(path: str) -> base_utils.plotlib.XYData`
**Description:** Loads data from an XVG-formatted file.
Parses title, axis labels, and legend from the file and returns an XYData object.


### Function: convert_from_df
**Signature:** `(df: pandas.core.frame.DataFrame, label: base_utils.plotlib.DataLabel) -> list[base_utils.plotlib.XYData]`
**Description:** Converts a pandas DataFrame into a list of XYData objects.
Assumes the first column of the DataFrame is the X-axis data, and subsequent columns are Y-axis data.


### Function: load_jasco_data
**Signature:** `(p: str) -> list[base_utils.plotlib.XYData]`
**Description:** Loads JASCO-formatted data from a file and returns it as a list of XYData objects.
Handles both 1D and 2D data formats.


### Function: load_dat
**Signature:** `(path: str, x_label: str = '', y_label: str = '', x_unit: str = '', y_unit: str = '', has_header: bool = False) -> base_utils.plotlib.XYData`
**Description:** Loads data from a .dat file (tab-separated) and returns it as an XYData object.
Allows specifying axis labels, units, and whether the file has a header.


### Function: load_csv
**Signature:** `(path: str) -> list[base_utils.plotlib.XYData]`
**Description:** Loads data from a CSV file and returns it as a list of XYData objects.
Assumes the first column is the X-axis data and subsequent columns are Y-axis data.


### Function: slice_data
**Signature:** `(data: list[base_utils.plotlib.XYData], x_value: float, new_x_values: list[float]) -> base_utils.plotlib.XYData`
**Description:** Slices a list of XYData objects at a specific X-value and interpolates Y-values
onto a new set of X-values.


### Function: plot1d
**Signature:** `(ax: matplotlib.axes._axes.Axes, data: base_utils.plotlib.XYData, style: base_utils.plotlib.Style, plot_option: base_utils.plotlib.PlotOption = PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), figure_option: base_utils.plotlib.FigureOption = FigureOption(size=(4, 3), plot_option=PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), is_white_background=True, plot_option_override=False), range: Optional[tuple[float, float]] = None) -> None`
**Description:** Plots a single 1D XYData series on a given Matplotlib Axes object.
Allows setting X-axis range.


### Function: plot2d
**Signature:** `(ax: matplotlib.axes._axes.Axes, data: list[base_utils.plotlib.XYData], style: base_utils.plotlib.Style, plot_option: base_utils.plotlib.PlotOption = PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), figure_option: base_utils.plotlib.FigureOption = FigureOption(size=(4, 3), plot_option=PlotOption(color=Color(colordict={<Style.paper: 0>: 'black', <Style.presentation_black: 1>: 'white', <Style.presentation_white: 2>: 'black', <Style.poster_black_highcontrast: 3>: 'white'}), marker=None, markersize=1, linestyle='-', linewidth=1.5), is_white_background=True, plot_option_override=False), xrange: Optional[tuple[float, float]] = None, yrange: Optional[tuple[float, float]] = None) -> None`
**Description:** Plots multiple XYData series on a given Matplotlib Axes object, suitable for 2D plots.
Allows setting X and Y axis ranges.


### Function: for_white_background
**Signature:** `(ax: matplotlib.axes._axes.Axes) -> None`
**Description:** Configures the given Matplotlib Axes object for a white background theme.
Sets spine, label, and tick colors to black.


### Function: for_black_background
**Signature:** `(ax: matplotlib.axes._axes.Axes) -> None`
**Description:** Configures the given Matplotlib Axes object for a black background theme.
Sets spine, label, and tick colors to white.


### Function: remove_all_text
**Signature:** `(ax: matplotlib.axes._axes.Axes) -> None`
**Description:** Removes all text elements (labels, title, legend) from a given Matplotlib Axes object.


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

