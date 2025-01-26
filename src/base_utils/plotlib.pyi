from enum import Enum
from typing import Optional
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass  # type: ignore

github_color_red: str
github_color_blue: str
github_color_boarder: str
github_color_text_gray: str
poster_color_red: str
poster_color_blue: str
poster_color_yellow: str
slide_color_orange: str


class Style(Enum):
    paper = 0
    presentation_black = 1
    presentation_white = 2
    poster_black_highcontrast = 3


@dataclass
class Color:
    colordict: dict[Style, str]
    def __init__(self, colordict: dict[Style, str]) -> None: ...
    def get_color(self, style: Style) -> str: ...
    @classmethod
    def single_color(cls, color: str) -> "Color": ...


colors: dict[str, Color]


@dataclass
class DataLabel:
    X_label: Optional[str]
    X_unit: Optional[str]
    Y_label: Optional[str]
    Y_unit: Optional[str]
    def __init__(self, X_label, X_unit, Y_label, Y_unit) -> None: ...


class DataLabels:
    UV = ...
    FL = ...
    CD = ...
    IR = ...
    XRD = ...


@dataclass(frozen=True)
class XYData:
    X: np.ndarray
    Y: np.ndarray
    dataLabel: DataLabel
    Title: str = ...
    def rename_labels(self, label: DataLabel) -> "XYData": ...
    def rename_title(self, title: str) -> "XYData": ...

    def get_y_at_range(
        self, xmin: float = -np.inf, xmax: float = np.inf
    ) -> np.ndarray: ...
    def xshift(self, shift: float) -> "XYData": ...
    def yshift(self, shift: float) -> "XYData": ...
    def xscale(self, scale: float) -> "XYData": ...
    def yscale(self, scale: float) -> "XYData": ...
    def clip(self, xmin: float, xmax: float) -> "XYData": ...
    def normalize(self) -> "XYData": ...
    def __init__(self, X, Y, dataLabel, Title=...) -> None: ...


@dataclass
class PlotOption:
    color: Color
    marker: str | None
    markersize: float
    linestyle: str
    linewidth: float
    def __init__(self, color, marker, markersize,
                 linestyle, linewidth) -> None: ...


class PlotOptions:
    paper = ...
    presentation = ...


default_figure_size: tuple[float, float]


@dataclass
class FigureOption:
    size: tuple[float, float]
    plot_option: PlotOption
    is_white_background: bool = ...
    plot_option_override: bool = ...

    def __init__(
        self,
        size,
        plot_option,
        is_white_background=...,
        plot_option_override=...
    ) -> None: ...


class FigureOptions:
    papar = ...
    presentation_white = ...
    presentation_black = ...


def load_1ddata(path: str) -> XYData: ...
def load_2ddata(path: str) -> list[XYData]: ...
def load_xvgdata(path: str) -> XYData: ...
def convert_from_df(df: pd.DataFrame, label: DataLabel) -> list[XYData]: ...
def load_jasco_data(p: str) -> list[XYData]: ...


def load_dat(
    path: str,
    x_label: str = "",
    y_label: str = "",
    x_unit: str = "",
    y_unit: str = "",
    has_header: bool = False,
) -> XYData: ...


def plot_old(
    data: XYData,
    figure_option: FigureOption = ...,
    save_path: Optional[str] = None
): ...


def plot_simple(
    ax: Axes,
    data: XYData,
    plot_option: PlotOption = ...,
    figure_option: FigureOption = ...,
    style: Style = ...,
) -> None: ...


def plot1d(
    ax: Axes,
    data: XYData,
    style: Style,
    plot_option: PlotOption = ...,
    figure_option: FigureOption = ...,
    range: Optional[tuple[float, float]] = None,
) -> None: ...


def plot2d(
    ax: Axes,
    data: list[XYData],
    style: Style,
    plot_option: PlotOption = ...,
    figure_option: FigureOption = ...,
    xrange: Optional[tuple[float, float]] = None,
    yrange: Optional[tuple[float, float]] = None,
) -> None: ...
def for_white_background(ax: Axes) -> None: ...
def for_black_background(ax: Axes) -> None: ...


def slice_data(
    data: list[XYData], x_value: float, new_x_values: list[float]
) -> XYData: ...
