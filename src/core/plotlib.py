import csv
import os
from dataclasses import dataclass
from enum import Enum
from math import isnan
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from typing_extensions import deprecated

from core import typecheck

# matplotlib setting (スライドに直接貼っても問題ないようにするため)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.size"] = 10

#defalut color
github_color_red = "#FF9492"
github_color_blue = "#9296f8"
github_color_boarder = "#707070"
github_color_text_gray = "#E1E4E8"

poster_color_red = "#FF0000"
poster_color_blue = "#00A0FF"
poster_color_yellow = "#FFFF00"

class Style(Enum):
    paper = 0
    presentation_black = 1
    presentation_white = 2
    poster_black_highcontrast = 3

@dataclass
class Color():
    colordict:dict[Style,str]
    def __init__(self, colordict:dict[Style,str]):
        self.colordict = colordict
    def get_color(self, style:Style)->str:
        return self.colordict[style]
    

colors:dict[str,Color] = {
    "red": Color({Style.paper: "red", Style.presentation_black: github_color_red, Style.presentation_white: "red", Style.poster_black_highcontrast: poster_color_red}),
    "blue": Color({Style.paper: "blue", Style.presentation_black: github_color_blue, Style.presentation_white: "blue", Style.poster_black_highcontrast: poster_color_blue}),
    "yellow": Color({Style.paper: "yellow", Style.presentation_black: "yellow", Style.presentation_white: "yellow", Style.poster_black_highcontrast: poster_color_yellow}),
    "green": Color({Style.paper: "green", Style.presentation_black: "green", Style.presentation_white: "green", Style.poster_black_highcontrast: "green"}),
    "monotone": Color({Style.paper: "black", Style.presentation_black: "white", Style.presentation_white: "black", Style.poster_black_highcontrast: "white"}),
}
    

@dataclass
class DataLabel:
    X_label: Optional[str]
    X_unit: Optional[str]
    Y_label: Optional[str]
    Y_unit: Optional[str]

class DataLabels(Enum):
    UV = DataLabel("wavelength", "nm", "absorbance", "-" )
    FL = DataLabel("wavelength", "nm", "intensity", "-" )
    CD = DataLabel("wavelength", "nm", "mdeg", "-" )
    IR = DataLabel("wavenumber", "cm-1", "absorbance", "-" )
    XRD = DataLabel("2theta", "degree", "intensity", "-" )
    
    

@dataclass(frozen=True)
class XYData:
    X: np.ndarray
    Y: np.ndarray
    dataLabel: DataLabel
    Title: str = ""

    def rename_labels(self, label:DataLabel) -> 'XYData':
        return XYData(self.X, self.Y, label, self.Title)
    
    
    def xshift(self, shift:float)->'XYData':
        return XYData(self.X + shift, self.Y, self.dataLabel, self.Title)
    
    def yshift(self, shift:float)->'XYData':
        return XYData(self.X, self.Y + shift, self.dataLabel, self.Title)
    
    def xscale(self, scale:float)->'XYData':
        return XYData(self.X * scale, self.Y, self.dataLabel, self.Title)
    
    def yscale(self, scale:float)->'XYData':
        return XYData(self.X, self.Y * scale, self.dataLabel, self.Title)
    
    def clip(self, xmin:float, xmax:float)->'XYData':
        mask = (self.X >= xmin) & (self.X <= xmax)
        return XYData(self.X[mask], self.Y[mask], self.dataLabel, self.Title)
    
    def normalize(self)->'XYData':
        Y = self.Y - np.min(self.Y)
        Y = Y / np.max(Y)
        return XYData(self.X, Y, self.dataLabel, self.Title)

@dataclass
class PlotOption:
    color: Color
    marker:str | None
    markersize: float
    linestyle: str
    linewidth: float
    
class PlotOptions(Enum):
    paper = PlotOption(colors["monotone"], None,1, "-", 1.5)
    presentation = PlotOption(colors["monotone"], "o", 2,"-", 3)

@dataclass
class FigureOption:
    size: Tuple[float, float]
    plot_option: PlotOption
    is_white_background: bool = True
    plot_option_override: bool = False



default_figure_size = (4, 3)    
class FigureOptions(Enum):
    papar = FigureOption(default_figure_size, PlotOptions.paper.value)
    presentation_white = FigureOption(default_figure_size, PlotOptions.presentation.value, is_white_background=True)
    presentation_black = FigureOption(default_figure_size, PlotOptions.presentation.value, is_white_background=False)


def _change_escape(text:str) -> str:
    """
    Change \r\n to \n
    """
    return text.replace("\r\n", "\n")

def _load_jasco_data(path:str)->Tuple[DataLabel, pd.DataFrame]:
    f = open(path, 'r')
    text = f.read()
    f.close()

    if os.path.splitext(path)[1] == ".csv":
        text = text.replace(",", "\t")

    label:Optional[DataLabel] = None
    # extract data type
    DataTypeline = text.split('DATA TYPE')
    if len(DataTypeline) == 1:
        raise ValueError("Unknown data type")
    DataType = DataTypeline[1].split('\n')[0].split("\n\n")[0].strip()
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
    text = text.split('XYDATA\n')[1].split("\n\n")[0]
    df = csv.reader(text.split('\n'), delimiter='\t')
    data = [row for row in df]
    # if cell is empty, fill with 0
    for row in data:
        for i in range(len(row)):
            if row[i] == '':
                row[i] = 'nan'

    # convert to float and transpose
    array = pd.DataFrame(data).astype(float)

    return label,array

@deprecated("Use load_data instead")
@typecheck.type_check
def load_1ddata(path: str) -> XYData:
    label, array = _load_jasco_data(path)
    return XYData(np.array(array[0].values), np.array(array[1].values), label)

@deprecated("Use load_data instead")
@typecheck.type_check
def load_2ddata(path: str) -> list[XYData]:
    label, array = _load_jasco_data(path)
    
    return [XYData(array[0].values[1:-1], array[i].values[1:-1], label,array[i].values[0]) for i in range(1, len(array.columns))][0:-1]

@typecheck.type_check
def load_xvgdata(path: str) -> XYData:
    f = open(path, 'r')
    data = f.readlines()
    f.close()
    data = [d for d in data if d[0] != '#']
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
    return XYData(data2[:,0], data2[:,1], DataLabel(xaxis, "", yaxis, ""), title + " " + legend)

@typecheck.type_check
def convert_from_df(df:pd.DataFrame, label:DataLabel)->list[XYData]:
    '''
    Convert DataFrame to list of XYData
    first column is x, other columns are y
    '''
    return [XYData(np.array(df.iloc[:,0].values), np.array(df.iloc[:,i].values), label) for i in range(1, len(df.columns))]

@typecheck.type_check
def load_jasco_data(p:str)->list[XYData]:
    label,data = _load_jasco_data(p)

    if isnan(data[0][0]):
        # is 2d data
        return [XYData(data[0].values[1:-1], data[i].values[1:-1], label,data[i].values[0]) for i in range(1, len(data.columns))][0:-1]
    else:
        # is 1d data
        return [XYData(np.array(data[0].values), np.array(data[1].values), label, os.path.basename(p))]  

@typecheck.type_check
def load_dat(path:str,x_label:str="",y_label:str="", x_unit:str = "", y_unit:str = "",has_header:bool = False)->XYData:
    f = open(path, 'r')
    data = f.readlines()
    f.close()
    data = [d for d in data if d[0] != '#']
    if has_header:
        data = data[1:]

    reader = csv.reader(data, delimiter='\t')
    data = [row for row in reader]

    for row in data:
        for i in range(len(row)):
            if row[i] == '':
                row[i] = 'nan'
    
    array = pd.DataFrame(data).astype(float)

    return XYData(array[0].values, array[1].values, DataLabel(x_label,x_unit, y_label, y_unit))
    








@deprecated("Use plot1d or plot2d instead")
@typecheck.type_check
def plot_old(data:XYData, figure_option:FigureOption = FigureOptions.papar.value, save_path: Optional[str] = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(figure_option.size)
    xmin, xmax = np.min(data.X), np.max(data.X)
    ax.plot(data.X, data.Y, color=figure_option.plot_option.color, marker=figure_option.plot_option.marker, linestyle=figure_option.plot_option.linestyle, linewidth=figure_option.plot_option.linewidth)
    ax.set_xlabel(f"{data.dataLabel.X_label} [{data.dataLabel.X_unit}]")
    ax.set_ylabel(f"{data.dataLabel.Y_label} [{data.dataLabel.Y_unit}]")
    ax.set_title(data.Title)
    ax.set_xlim(xmin, xmax)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_simple(ax:Axes, data:XYData, plot_option:PlotOption = PlotOptions.paper.value,figure_option:FigureOption = FigureOptions.papar.value,style:Style=Style.paper)->None:
    ax.plot(data.X, data.Y, 
            color=plot_option.color.get_color(style), 
            marker=plot_option.marker, 
            markersize=plot_option.markersize,
            linestyle=plot_option.linestyle, 
            linewidth=plot_option.linewidth, 
            label=data.Title)
    ax.set_xlabel(f"{data.dataLabel.X_label} [{data.dataLabel.X_unit}]")
    ax.set_ylabel(f"{data.dataLabel.Y_label} [{data.dataLabel.Y_unit}]")
    ax.set_title(data.Title)

@typecheck.type_check
def plot1d(ax:Axes, data:XYData, style:Style,
           plot_option:PlotOption = PlotOptions.paper.value,
           figure_option:FigureOption = FigureOptions.papar.value ,
           range:Optional[Tuple[float, float]] = None)->None:
    plot_simple(ax, data, plot_option,style=style)
    ax.set_xlim(np.min(data.X), np.max(data.X))
    if range is not None:
        ax.set_xlim(range)

@typecheck.type_check
def plot2d(ax:Axes, data:List[XYData],style:Style, 
           plot_option:PlotOption = PlotOptions.paper.value,
           figure_option:FigureOption = FigureOptions.papar.value, 
           xrange:Optional[Tuple[float, float]] = None,
           yrange:Optional[Tuple[float, float]] = None)->None:
    min,max = np.inf, -np.inf
    for d in data:
        plot_simple(ax, d, plot_option,style=style)
        if np.min(d.X) < min:
            min = np.min(d.X)
        if np.max(d.X) > max:
            max = np.max(d.X)
    
    if xrange is not None:
        min, max = xrange
    ax.set_xlim(min, max)

    if yrange is not None:
        ax.set_ylim(yrange)

@typecheck.type_check
def for_white_background(ax:Axes)->None:
    ax.spines[:].set_color('black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black', which='both')
    ax.tick_params(axis='y', colors='black', which='both')
    ax.title.set_color('black')

@typecheck.type_check
def for_black_background(ax:Axes)->None:
    ax.spines[:].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white', which='both')
    ax.tick_params(axis='y', colors='white', which='both')
    ax.title.set_color('white')
    

def _test()->None:
    path1d = r"C:\Users\taman\Dropbox\python\plot\20240201_QuinPhTDP_100microM_MCH_NormalCoolingSingleWaveLen.txt"
    path2d = r"C:\Users\taman\Dropbox\python\plot\20240512_iQuinPhTDP_50microM_MCH_NormalCooling.txt"
    
    data1d = load_1ddata(path1d)
    data2d = load_2ddata(path2d)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_option = PlotOptions.paper.value
    plot_option.marker = None
    plot2d(ax, data2d, plot_option, (250,400))
    plt.show()

if __name__ == "__main__":
    _test()