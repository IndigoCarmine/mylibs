import csv
from dataclasses import dataclass
from enum import Enum
from math import isnan
from operator import le
import os
from turtle import title
from matplotlib.axes import Axes
from typing_extensions import deprecated
import numpy as np
import matplotlib.pyplot as plt
from typing import *
import pandas as pd

from core import typecheck


# matplotlib setting (スライドに直接貼っても問題ないようにするため)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.size"] = 10


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
    DataLabel: DataLabel
    Title: str = ""
    def rename_labels(self, X_label: Optional[str]= None, Y_label: Optional[str] = None, Title: Optional[str] =None) -> 'XYData':
        X_label_temp = X_label if X_label is not None else self.DataLabel.X_label
        Y_label_temp = Y_label if Y_label is not None else self.DataLabel.Y_label
        Title_temp = Title if Title is not None else self.Title
        return XYData(self.X, self.Y, DataLabel(X_label_temp, self.DataLabel.X_unit, Y_label_temp, self.DataLabel.Y_unit), Title_temp)

@dataclass
class PlotOption:
    color: Optional[str] 
    marker: Optional[str]
    linestyle: Optional[str]
    linewidth: Optional[float]
    
class PlotOptions(Enum):
    paper = PlotOption("black", "o", "-", 1.5)
    presentation_white = PlotOption("black", "o", "-", 1.5)
    presentation_black = PlotOption("white", "o", "-", 1.5)

@dataclass
class FigureOption:
    size: Tuple[float, float]
    plot_option: PlotOption
    plot_option_override: bool = False



_default_figure_size = (4, 3)    
class FigureOptions(Enum):
    papar = FigureOption(_default_figure_size, PlotOptions.paper.value)
    presentation_white = FigureOption(_default_figure_size, PlotOptions.presentation_white.value)
    presentation_black = FigureOption(_default_figure_size, PlotOptions.presentation_black.value)

@typecheck.type_check
def _change_escape(text:str) -> str:
    """
    Change \r\n to \n
    """
    return text.replace("\r\n", "\n")

@typecheck.type_check
def _load_xydata(path:str)->Tuple[DataLabel, pd.DataFrame]:
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
    label, array = _load_xydata(path)
    return XYData(np.array(array[0].values), np.array(array[1].values), label)

@deprecated("Use load_data instead")
@typecheck.type_check
def load_2ddata(path: str) -> list[XYData]:
    label, array = _load_xydata(path)
    
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
            if "legend" in d:
                legend = d.split('"')[1]
    
    data1 = [d.split() for d in data if d[0] != '@']
    data2 = np.array(data).astype(float)
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
def load_data(p:str)->list[XYData]:
    label,data = _load_xydata(p)

    if isnan(data[0][0]):
        # is 2d data
        return [XYData(data[0].values[1:-1], data[i].values[1:-1], label,data[i].values[0]) for i in range(1, len(data.columns))][0:-1]
    else:
        # is 1d data
        return [XYData(np.array(data[0].values), np.array(data[1].values), label, os.path.basename(p))]  


@deprecated("Use plot1d or plot2d instead")
@typecheck.type_check
def plot_old(data:XYData, figure_option:FigureOption = FigureOptions.papar.value, save_path: Optional[str] = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches(figure_option.size)
    xmin, xmax = np.min(data.X), np.max(data.X)
    ax.plot(data.X, data.Y, color=figure_option.plot_option.color, marker=figure_option.plot_option.marker, linestyle=figure_option.plot_option.linestyle, linewidth=figure_option.plot_option.linewidth)
    ax.set_xlabel(f"{data.DataLabel.X_label} ({data.DataLabel.X_unit})")
    ax.set_ylabel(f"{data.DataLabel.Y_label} ({data.DataLabel.Y_unit})")
    ax.set_title(data.Title)
    ax.set_xlim(xmin, xmax)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

@typecheck.type_check
def plot_simple(ax:Axes, data:XYData, plot_option:PlotOption = PlotOptions.paper.value)->None:
    ax.plot(data.X, data.Y, color=plot_option.color, marker=plot_option.marker, linestyle=plot_option.linestyle, linewidth=plot_option.linewidth, label=data.Title)
    ax.set_xlabel(f"{data.DataLabel.X_label} ({data.DataLabel.X_unit})")
    ax.set_ylabel(f"{data.DataLabel.Y_label} ({data.DataLabel.Y_unit})")
    ax.set_title(data.Title)

@typecheck.type_check
def plot1d(ax:Axes, data:XYData, plot_option:PlotOption = PlotOptions.paper.value,range:Optional[Tuple[float, float]] = None)->None:
    plot_simple(ax, data, plot_option)
    ax.set_xlim(np.min(data.X), np.max(data.X))
    if range is not None:
        ax.set_xlim(range)

def plot2d(ax:Axes, data:List[XYData], plot_option:PlotOption = PlotOptions.paper.value, xrange:Optional[Tuple[float, float]] = None,yrange:Optional[Tuple[float, float]] = None)->None:
    min,max = np.inf, -np.inf
    for d in data:
        plot_simple(ax, d, plot_option)
        if np.min(d.X) < min:
            min = np.min(d.X)
        if np.max(d.X) > max:
            max = np.max(d.X)
    
    if xrange is not None:
        min, max = xrange
    ax.set_xlim(min, max)

    if yrange is not None:
        ax.set_ylim(yrange)
    
    

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