"""
This module provides a comprehensive set of tools for creating and customizing plots using Matplotlib.
It includes functionalities for defining plot styles, handling data labels, managing plot options,
loading various data formats (JASCO, XVG, DAT), and generating 1D and 2D plots.
"""

import csv
import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum
from math import isnan

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from typing_extensions import deprecated

from ref_for_llm.deco import llm_public

FLOAT = np.float64

# matplotlib setting (スライドに直接貼っても問題ないようにするため)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 16

# defalut color
github_color_red = "#FF9492"
github_color_blue = "#9296f8"
github_color_boarder = "#707070"
github_color_text_gray = "#E1E4E8"

poster_color_red = "#FF0000"
poster_color_blue = "#00A0FF"
poster_color_yellow = "#FFFF00"

slide_color_orange = "#ff7b00"

@llm_public()
class Style(Enum):
    """
    Defines different plotting styles for various output formats (e.g., paper, presentation).
    """

    paper = 0
    presentation_black = 1
    presentation_white = 2
    poster_black_highcontrast = 3

@llm_public()
@dataclass
class Color:
    """
    Manages colors for different plotting styles.
    Allows defining a set of colors that adapt to various output contexts (e.g., paper, presentation).
    """

    colordict: dict[Style, str | None]

    @llm_public()
    def __init__(self, colordict: dict[Style, str | None]):
        self.colordict = colordict

    def get_color(self, style: Style) -> str | None:
        """
        Retrieves the color string for a given plotting style.
        """
        return self.colordict[style]

    @llm_public()
    @classmethod
    def single_color(cls, color: str) -> "Color":
        """
        Creates a Color instance where the same color is used across all styles.
        """
        return cls(
            {
                Style.paper: color,
                Style.presentation_black: color,
                Style.presentation_white: color,
                Style.poster_black_highcontrast: color,
            }
        )


colors: dict[str, Color] = {
    "red": Color(
        {
            Style.paper: "red",
            Style.presentation_black: github_color_red,
            Style.presentation_white: "red",
            Style.poster_black_highcontrast: poster_color_red,
        }
    ),
    "blue": Color(
        {
            Style.paper: "blue",
            Style.presentation_black: github_color_blue,
            Style.presentation_white: "blue",
            Style.poster_black_highcontrast: poster_color_blue,
        }
    ),
    "yellow": Color(
        {
            Style.paper: "yellow",
            Style.presentation_black: "yellow",
            Style.presentation_white: "yellow",
            Style.poster_black_highcontrast: poster_color_yellow,
        }
    ),
    "green": Color(
        {
            Style.paper: "green",
            Style.presentation_black: "green",
            Style.presentation_white: "green",
            Style.poster_black_highcontrast: "green",
        }
    ),
    "monotone": Color(
        {
            Style.paper: "black",
            Style.presentation_black: "white",
            Style.presentation_white: "black",
            Style.poster_black_highcontrast: "white",
        }
    ),
    "None": Color(
        {
            Style.paper: None,
            Style.presentation_black: None,
            Style.presentation_white: None,
            Style.poster_black_highcontrast: None,
        }
    ),
}


@dataclass
class DataLabel:
    """
    Represents labels and units for X and Y axes of a plot.
    """

    X_label: Optional[str]
    X_unit: Optional[str]
    Y_label: Optional[str]
    Y_unit: Optional[str]

@llm_public()
class DataLabels(Enum):
    """
    Predefined DataLabel instances for common spectroscopic data types.
    """

    UV = DataLabel("wavelength", "nm", "absorbance", "-")
    FL = DataLabel("wavelength", "nm", "intensity", "-")
    CD = DataLabel("wavelength", "nm", "mdeg", "-")
    IR = DataLabel("wavenumber", "cm-1", "absorbance", "-")
    XRD = DataLabel("2theta", "degree", "intensity", "-")

@llm_public()
@dataclass(frozen=True)
class XYData:
    """
    Represents a set of X-Y data points with associated labels and an optional title.
    Provides methods for data manipulation such as renaming, shifting, scaling, and normalization.
    """
    X: npt.NDArray[FLOAT]
    Y: npt.NDArray[FLOAT]
    dataLabel: DataLabel
    Title: str = ""

    def __post_init__(self):
        """
        Initializes the XYData object and performs validation checks.
        Raises ValueError if X and Y arrays have different lengths or are empty.
        """
        if len(self.X) != len(self.Y):
            raise ValueError(
                f"X and Y must have the same length: {len(self.X)} != {len(self.Y)}"
            )
        if len(self.X) == 0:
            raise ValueError("X and Y must not be empty")
    @llm_public()
    def rename_labels(self, label: DataLabel) -> "XYData":
        """
        Returns a new XYData object with updated DataLabel.
        """
        return XYData(self.X, self.Y, label, self.Title)
    @llm_public()
    def rename_title(self, title: str) -> "XYData":
        """
        Returns a new XYData object with an updated title.
        """
        return XYData(self.X, self.Y, self.dataLabel, title)
    @llm_public()
    def get_y_at_range(
        self, xmin: float = -np.inf, xmax: float = np.inf
    ) -> npt.NDArray[FLOAT]:
        """
        Returns the Y values within a specified X range.
        """
        mask = (self.X >= xmin) & (self.X <= xmax)
        return self.Y[mask]
    @llm_public()
    def get_y_at_nearest_x(self, x: float) -> float:
        """
        Returns the Y value corresponding to the X value nearest to the given 'x'.
        """
        idx = np.argmin(np.abs(self.X - x))
        return self.Y[idx]
    @llm_public()
    def xshift(self, shift: float) -> "XYData":
        """
        Returns a new XYData object with X values shifted by a given amount.
        """
        return XYData(self.X + shift, self.Y, self.dataLabel, self.Title)
    @llm_public()
    def yshift(self, shift: float) -> "XYData":
        """
        Returns a new XYData object with Y values shifted by a given amount.
        """
        return XYData(self.X, self.Y + shift, self.dataLabel, self.Title)
    @llm_public()
    def xscale(self, scale: float) -> "XYData":
        """
        Returns a new XYData object with X values scaled by a given factor.
        """
        return XYData(self.X * scale, self.Y, self.dataLabel, self.Title)
    @llm_public()
    def yscale(self, scale: float) -> "XYData":
        """
        Returns a new XYData object with Y values scaled by a given factor.
        """
        return XYData(self.X, self.Y * scale, self.dataLabel, self.Title)
    @llm_public()
    def clip(self, xmin: float, xmax: float) -> "XYData":
        """
        Returns a new XYData object with data clipped to a specified X range.
        """
        mask = (self.X >= xmin) & (self.X <= xmax)
        return XYData(self.X[mask], self.Y[mask], self.dataLabel, self.Title)
    @llm_public()
    def normalize(self) -> "XYData":
        """
        Returns a new XYData object with Y values normalized to a range of 0 to 1.
        """
        y = self.Y - np.min(self.Y)
        y = y / np.max(y)
        print("scaling factor is", 1 / np.max(self.Y))
        return XYData(self.X, y, self.dataLabel, self.Title)

@llm_public()
@dataclass
class PlotOption:
    """
    Defines options for plotting individual data series, such as color, marker, and line style.
    """

    color: Color
    marker: str | None
    markersize: float
    linestyle: str
    linewidth: float

class PlotOptions:
    """
    Predefined PlotOption instances for common plotting scenarios.
    """

    paper = PlotOption(colors["monotone"], None, 1, "-", 1.5)
    presentation = PlotOption(colors["monotone"], "o", 2, "-", 3)


@dataclass
class FigureOption:
    """
    Defines options for the overall figure, including size, plot options, and background.
    """

    size: tuple[float, float]
    plot_option: PlotOption
    is_white_background: bool = True
    plot_option_override: bool = False


default_figure_size = (4, 3)

class FigureOptions:
    """
    Predefined FigureOption instances for common figure setups.
    """

    papar = FigureOption(default_figure_size, PlotOptions.paper)
    presentation_white = FigureOption(
        default_figure_size, PlotOptions.presentation, is_white_background=True
    )
    presentation_black = FigureOption(
        default_figure_size, PlotOptions.presentation, is_white_background=False
    )


def _change_escape(text: str) -> str:
    """
    Replaces Windows-style carriage return and newline characters (\r\n) with Unix-style newlines (\n).
    """
    return text.replace("\r\n", "\n")


def _load_jasco_data(path: str) -> tuple[DataLabel, pd.DataFrame]:
    """
    Loads data from a JASCO-formatted file (or CSV that can be converted to JASCO-like tab-separated).
    Extracts data type and returns a DataLabel and a pandas DataFrame.
    """
    f = open(path, "r")
    text = f.read()
    f.close()

    if os.path.splitext(path)[1] == ".csv":
        text = text.replace(",", "\t")

    label: Optional[DataLabel] = None
    # extract data type
    DataTypeline = text.split("DATA TYPE")
    if len(DataTypeline) == 1:
        raise ValueError("Unknown data type")
    DataType = DataTypeline[1].split("\n")[0].split("\n\n")[0].strip()
    if DataType == "ULTRAVIOLET SPECTRUM":
        label = DataLabels.UV.value
    elif DataType == "FLUORESCENCE SPECTRUM":
        label = DataLabels.FL.value
    elif DataType == "INFRARED SPECTRUM":
        label = DataLabels.IR.value
    else:
        raise ValueError(f"Unknown data type: {DataType}")

    # extract part of data
    text = _change_escape(text)
    text = text.split("XYDATA\n")[1].split("\n\n")[0]
    df = csv.reader(text.split("\n"), delimiter="\t")
    data = [row for row in df]
    # if cell is empty, fill with 0
    for row in data:
        for i in range(len(row)):
            if row[i] == "":
                row[i] = "nan"

    # convert to float and transpose
    array = pd.DataFrame(data).astype(float)

    return label, array


@deprecated("Use load_data instead")
def load_1ddata(path: str) -> XYData:
    """
    Loads 1D data from a JASCO-formatted file.
    This function is deprecated; use `load_jasco_data` instead.
    """
    label, array = _load_jasco_data(path)
    return XYData(array[0].to_numpy(), array[1].values.to_numpy(), label)  # type: ignore


@deprecated("Use load_data instead")
def load_2ddata(path: str) -> list[XYData]:
    """
    Loads 2D data from a JASCO-formatted file.
    This function is deprecated; use `load_jasco_data` instead.
    """
    label, array = _load_jasco_data(path)

    return [
        XYData(
            array[0].to_numpy()[1:-1],
            array[i].to_numpy()[1:-1],
            label,
            str(array[i].values[0]),
        )
        for i in range(1, len(array.columns))
    ][0:-1]

@llm_public()
def load_xvgdata(path: str) -> XYData:
    """
    Loads data from an XVG-formatted file.
    Parses title, axis labels, and legend from the file and returns an XYData object.
    """
    f = open(path, "r")
    data = f.readlines()
    f.close()
    data = [d for d in data if d[0] != "#"]
    title = ""
    xaxis = ""
    yaxis = ""
    legend = ""

    for d in data:
        if d[0] == "@":
            if "title" in d:
                title = d.split('"')[1]
            if "xaxis" in d:
                xaxis = d.split('"')[1]
            if "yaxis" in d:
                yaxis = d.split('"')[1]
            if "s0 legend" in d:
                legend = d.split('"')[1]

    data1 = [d.split() for d in data if not d.startswith("@")]
    data2 = np.array(data1).astype(float)
    # remove first row ( it is empty)
    data2 = data2[1:]
    return XYData(
        data2[:, 0], data2[:, 1], DataLabel(xaxis, "", yaxis, ""), title + " " + legend
    )

@llm_public()
def convert_from_df(df: pd.DataFrame, label: DataLabel) -> list[XYData]:
    """
    Converts a pandas DataFrame into a list of XYData objects.
    Assumes the first column of the DataFrame is the X-axis data, and subsequent columns are Y-axis data.
    """
    return [
        XYData(np.array(df.iloc[:, 0].values), np.array(df.iloc[:, i].values), label)
        for i in range(1, len(df.columns))
    ]

@llm_public()
def load_jasco_data(p: str) -> list[XYData]:
    """
    Loads JASCO-formatted txt data from a file and returns it as a list of XYData objects.
    Handles both 1D and 2D data formats.
    """
    label, data = _load_jasco_data(p)

    if isnan(data[0][0]):
        # is 2d data
        return [
            XYData(data[0].to_numpy()[1:-1], data[i].to_numpy()[1:-1], label, str(data[i].values[0]))
            for i in range(1, len(data.columns))
        ][0:-1]
    else:
        # is 1d data
        return [
            XYData(
                np.array(data[0].values),
                np.array(data[1].values),
                label,
                os.path.basename(p),
            )
        ]

@llm_public()
def load_dat(
    path: str,
    x_label: str = "",
    y_label: str = "",
    x_unit: str = "",
    y_unit: str = "",
    has_header: bool = False,
) -> XYData:
    """
    Loads data from a .dat file (tab-separated) and returns it as an XYData object.
    Allows specifying axis labels, units, and whether the file has a header.
    """
    f = open(path, "r")
    data = f.readlines()
    f.close()
    data = [d for d in data if d[0] != "#"]
    if has_header:
        data = data[1:]

    reader = csv.reader(data, delimiter="\t")
    data = [row for row in reader]

    for row in data:
        for i in range(len(row)):
            if row[i] == "":
                row[i] = "nan"

    array = pd.DataFrame(data).astype(float)

    return XYData(
        array[0].to_numpy(), array[1].to_numpy(), DataLabel(x_label, x_unit, y_label, y_unit)
    )


@llm_public()
def load_csv(path: str) -> list[XYData]:
    """
    Loads data from a CSV file and returns it as a list of XYData objects.
    Assumes the first column is the X-axis data and subsequent columns are Y-axis data.
    """

    df = pd.read_csv(path)
    if len(df.columns) < 2:
        raise ValueError("CSV file must have at least two columns")

    columns = df.columns.tolist()
    x = columns[0]
    y = columns[1]
    if x == "WAVELENGTH" and y == "ABSORBANCE":
        # UV-Vis data
        label = DataLabels.UV.value
    elif x == "WAVELENGTH" and y == "INTENSITY":
        # Fluorescence data
        label = DataLabels.FL.value
    elif x == "WAVENUMBER" and y == "ABSORBANCE":
        # IR data
        label = DataLabels.IR.value
    else:
        label = DataLabel(x, "", y, "")

    return convert_from_df(df, label)


@llm_public()
def slice_data(data: list[XYData], x_value: float, new_x_values: list[float]) -> XYData:
    """
    Slices a list of XYData objects at a specific X-value and interpolates Y-values
    onto a new set of X-values.
    """

    def nearest(array: npt.NDArray[np.float64], value: float) -> np.intp:
        idx = (np.abs(array - value)).argmin()
        return idx

    x_value_index = np.array([nearest(d.X, x_value) for d in data])
    x_values = np.array([d.X[i] for d, i in zip(data, x_value_index)])
    print("x_values is", x_values.mean(), "+-", x_values.std())

    Y = np.array([d.Y[i] for d, i in zip(data, x_value_index)])
    label = DataLabel("", "", data[0].dataLabel.Y_label, data[0].dataLabel.Y_unit)
    return XYData(np.array(new_x_values), Y, label, str(x_value))


@deprecated("Use plot1d or plot2d instead")
def plot_old(
    data: XYData,
    figure_option: FigureOption = FigureOptions.papar,
    save_path: Optional[str] = None,
):
    """
    Generates a 1D plot of XYData.
    This function is deprecated; use `plot1d` or `plot2d` instead.
    """
    fig = plt.figure()  # type: ignore
    ax = fig.add_subplot(111)  # type: ignore
    fig.set_size_inches(figure_option.size)
    xmin, xmax = np.min(data.X), np.max(data.X)
    ax.plot(  # type: ignore
        data.X,
        data.Y,
        color=figure_option.plot_option.color,
        marker=figure_option.plot_option.marker,
        linestyle=figure_option.plot_option.linestyle,
        linewidth=figure_option.plot_option.linewidth,
    )
    ax.set_xlabel(f"{data.dataLabel.X_label} [{data.dataLabel.X_unit}]")  # type: ignore
    ax.set_ylabel(f"{data.dataLabel.Y_label} [{data.dataLabel.Y_unit}]")  # type: ignore
    ax.set_title(data.Title)  # type: ignore
    ax.set_xlim(xmin, xmax)  # type: ignore
    if save_path is not None:
        plt.savefig(save_path)  # type: ignore
    plt.show()  # type: ignore

@llm_public()
def plot_simple(
    ax: Axes,
    data: XYData,
    plot_option: PlotOption = PlotOptions.paper,
    figure_option: FigureOption = FigureOptions.papar,
    style: Style = Style.paper,
) -> None:
    """
    Plots a single XYData series on a given Matplotlib Axes object.
    Applies specified plot options and style.
    """
    ax.plot(  # type: ignore
        data.X,
        data.Y,
        color=plot_option.color.get_color(style),
        marker=plot_option.marker,
        markersize=plot_option.markersize,
        linestyle=plot_option.linestyle,
        linewidth=plot_option.linewidth,
        label=data.Title,
    )
    ax.set_xlabel(f"{data.dataLabel.X_label} [{data.dataLabel.X_unit}]")  # type: ignore
    ax.set_ylabel(f"{data.dataLabel.Y_label} [{data.dataLabel.Y_unit}]")  # type: ignore
    ax.set_title(data.Title)  # type: ignore

@llm_public()
def plot1d(
    ax: Axes,
    data: XYData,
    style: Style,
    plot_option: PlotOption = PlotOptions.paper,
    figure_option: FigureOption = FigureOptions.papar,
    range: Optional[tuple[float, float]] = None,
) -> None:
    """
    Plots a single 1D XYData series on a given Matplotlib Axes object.
    Allows setting X-axis range.
    """
    plot_simple(ax, data, plot_option, style=style)
    ax.set_xlim(float(np.min(data.X)), float(np.max(data.X)))
    if range is not None:
        ax.set_xlim(range)

@llm_public()
def plot2d(
    ax: Axes,
    data: list[XYData],
    style: Style,
    plot_option: PlotOption = PlotOptions.paper,
    figure_option: FigureOption = FigureOptions.papar,
    xrange: Optional[tuple[float, float]] = None,
    yrange: Optional[tuple[float, float]] = None,
) -> None:
    """
    Plots multiple XYData series on a given Matplotlib Axes object, suitable for 2D plots.
    Allows setting X and Y axis ranges.
    """
    min, max = np.inf, -np.inf
    for d in data:
        plot_simple(ax, d, plot_option, style=style)
        if np.min(d.X) < min:
            min = np.min(d.X)
        if np.max(d.X) > max:
            max = np.max(d.X)

    if xrange is not None:
        min, max = xrange
    ax.set_xlim(min, max)  # type: ignore

    if yrange is not None:
        ax.set_ylim(yrange)

@llm_public()
def for_white_background(ax: Axes) -> None:
    """
    Configures the given Matplotlib Axes object for a white background theme.
    Sets spine, label, and tick colors to black.
    """
    ax.spines[:].set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="x", colors="black", which="both")  # type: ignore
    ax.tick_params(axis="y", colors="black", which="both")  # type: ignore
    ax.title.set_color("black")

@llm_public()
def for_black_background(ax: Axes) -> None:
    """
    Configures the given Matplotlib Axes object for a black background theme.
    Sets spine, label, and tick colors to white.
    """
    ax.spines[:].set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white", which="both")  # type: ignore
    ax.tick_params(axis="y", colors="white", which="both")  # type: ignore
    ax.title.set_color("white")

@llm_public()
def remove_all_text(ax: Axes) -> None:
    """
    Removes all text elements (labels, title, legend) from a given Matplotlib Axes object.
    """
    ax.set_xlabel("")  # type: ignore
    ax.set_ylabel("")  # type: ignore
    ax.set_title("")  # type: ignore
    ax.legend().remove()  # type: ignore
