"""
This module provides tools for analyzing GROMACS simulation data.
It includes functionalities for recording and managing analysis results,
processing multiple simulation files, and generating analysis scripts.
"""

from concurrent.futures import ProcessPoolExecutor
import copy
import os
from typing import Callable, Iterator, Protocol
import MDAnalysis as mda
from functools import partial, total_ordering
from base_utils import cui_utils
import numpy as np
import numpy.typing as npt
import pandas as pd


@total_ordering
class Recorder:
    """
    A class to record and manage analysis results for a specific calculation.
    It stores named values and log data, and supports concatenation with other Recorders.
    """

    calc_name: str
    values: list[tuple[str, float]]
    log_data: list[str]

    def __init__(self, calculation_name: str):
        """
        Initializes a new Recorder instance.
        Args:
            calculation_name (str): The name of the calculation this recorder is for.
        """
        self.calc_name = calculation_name
        self.values = []
        self.log_data = []

    def log(self, val):
        """
        Adds a log entry to the recorder.
        Args:
            val: The value to log.
        """
        self.log_data.append(str(val))

    def add_value(self, name: str, val: float):
        """
        Adds a named numerical value to the recorder.
        Args:
            name (str): The name of the value.
            val (float): The numerical value.
        """
        self.values.append((name, val))

    def add_value_of_default_array_analysis(
        self, name: str, array: npt.NDArray[np.number]
    ):
        """
        Adds mean and standard deviation of a numerical array as named values.
        Args:
            name (str): The base name for the values (e.g., "energy" will result in "energy_mean" and "energy_standard").
            array (float): The numerical array to analyze.
        """
        self.add_value(f"{name}_mean", np.mean(array))
        self.add_value(f"{name}_standard", np.std(array))

    def get_all_valuename(self) -> list[str]:
        """
        Returns a list of all value names recorded.
        Returns:
            list[str]: A list of value names.
        """
        return [name for name, _ in self.values]

    def concat(self, recorder: "Recorder"):
        """
        Concatenates another Recorder's values into this recorder.
        Raises ValueError if calculation names do not match or if there are duplicate value names.
        Args:
            recorder (Recorder): The Recorder object to concatenate.
        """
        if self.calc_name != recorder.calc_name:
            raise ValueError(
                "Recorders dont have a same calculation_name: self is {} but arg is {}".format(
                    self.calc_name, recorder.calc_name
                )
            )

        value_name = [val for val, _ in self.values]
        for name, val in recorder.values:
            if name in value_name:
                raise ValueError(f"Recorders have a same value:{val}")
            else:
                self.values.append((name, val))

    def __eq__(self, other):
        """
        Compares two Recorder objects for equality based on their calculation name.
        """
        if not isinstance(other, Recorder):
            raise TypeError("Cannot compare with Recorder and {}".format(type(other)))

        return self.calc_name == other.calc_name

    def __lt__(self, other):
        """
        Compares two Recorder objects for less than based on their calculation name.
        """
        if not isinstance(other, Recorder):
            return TypeError("Cannot compare with Recorder and {}".format(type(other)))

        return self.calc_name < other.calc_name


def generate_excel(file_name: str, recoders: list[Recorder]):
    """
    Generates an Excel file from a list of Recorder objects.
    Each Recorder's values are written as a row in the Excel file.
    Args:
        file_name (str): The name of the Excel file to generate.
        recoders (list[Recorder]): A list of Recorder objects containing the data.
    """
    # all = set()
    # for recoder in recoders:
    #     for val in recoder:
    #         all.add(val)
    all = [name for name, _ in recoders[0].values]
    # all = ["name"] + all
    data = pd.DataFrame()

    for recoder in recoders:
        data[recoder.calc_name] = [val for _, val in recoder.values]
    print(data)
    data = data.T
    data.columns = all
    data.to_excel(file_name)


def mixing_recorders(
    base_recorder_list: list[Recorder], recorder_list: list[Recorder]
) -> list[Recorder]:
    """
    Mixes (concatenates) values from a list of new recorders into a base list of recorders.
    Recorders are matched by their `calc_name`.
    Args:
        base_recorder_list (list[Recorder]): The base list of recorders to which values will be added.
        recorder_list (list[Recorder]): The list of recorders whose values will be added.
    Returns:
        list[Recorder]: A new list of recorders with mixed values.
    """
    new_recorders = copy.deepcopy(base_recorder_list)

    for recorder in new_recorders:
        for add_recorder in recorder_list:
            if recorder.calc_name == add_recorder.calc_name:
                recorder.concat(add_recorder)

    return new_recorders


def perocess_files(
    calculation_basedir: str,
    last_calc,
    work: Callable[[str], Recorder | None],
    do_parallel: bool = False,
) -> list[Recorder]:
    """
    Processes files within a specified calculation base directory.
    Applies a given work function to each relevant directory, optionally in parallel.
    Args:
        calculation_basedir (str): The base directory containing calculation folders.
        last_calc: The name of the last calculation subdirectory (e.g., "4_md_main").
        work (Callable[[str], Recorder | None]): A function that takes a directory path and returns a Recorder object or None.
        do_parallel (bool): If True, processes files in parallel using a ProcessPoolExecutor.
    Returns:
        list[Recorder]: A list of Recorder objects generated by the work function.
    """

    # get fullpathes of all file and dir in calculation_basedir
    directry = [
        os.path.join(calculation_basedir, dirname)
        for dirname in os.listdir(calculation_basedir)
    ]
    # cui_utils.notice("debug code is running :analysis work spesific files")

    # convert to pathes of last_calc dir.
    directry = [dir for dir in directry if os.path.isdir(dir)]
    directry = [os.path.join(dir, last_calc) for dir in directry]
    directry = [dir for dir in directry if os.path.exists(dir)]
    recorders = []
    if do_parallel:
        with ProcessPoolExecutor() as executor:
            result = executor.map(work, directry)
            for recorder in result:
                if recorder is not None:
                    recorders.append(recorder)

    else:
        for dir in directry:
            recorder = work(dir)
            if recorder is not None:
                recorders.append(recorder)

    return recorders


def _analyze_trj_inner(
    path: str, work: Callable[[mda.Universe, str], Recorder]
) -> Recorder | None:
    """
    Internal helper function to analyze a single trajectory file.
    Loads the GROMACS universe and applies a work function.
    Args:
        path (str): The path to the directory containing output.gro and output.xtc.
        work (Callable[[mda.Universe, str], Recorder]): A function that takes an MDAnalysis Universe object and the path, and returns a Recorder.
    Returns:
        Recorder | None: A Recorder object with analysis results, or None if files are not found.
    """
    grofile = os.path.join(path, "output.gro")
    xtcfile = os.path.join(path, "output.xtc")
    if not os.path.exists(grofile):
        cui_utils.warning(f"{grofile} does not exist")
        return None
    u = mda.Universe(grofile)
    if xtcfile is not None:
        u.load_new(xtcfile)
    return work(u, path)


def analyze_trj(
    calculation_basedir: str,
    last_calc,
    work: Callable[[mda.Universe, str], Recorder | None],
    do_parallel: bool = False,
) -> list[Recorder]:
    """
    Analyzes GROMACS trajectory files across multiple calculation directories.
    It uses `perocess_files` internally to handle parallel processing.
    Args:
        calculation_basedir (str): The base directory containing calculation folders.
        last_calc: The name of the last calculation subdirectory (e.g., "4_md_main").
        work (Callable[[mda.Universe, str], Recorder | None]): A function that takes an MDAnalysis Universe object and the path, and returns a Recorder.
        do_parallel (bool): If True, processes files in parallel.
    Returns:
        list[Recorder]: A list of Recorder objects with analysis results.
    """
    return perocess_files(
        calculation_basedir,
        last_calc,
        partial(_analyze_trj_inner, work=work),
        do_parallel,
    )


def get_calcname(calc_path: str) -> str:
    """
    Extracts the calculation name from a given calculation path.
    Example: "...../iQuin/4_md_main" -> "iQuin"
    Args:
        calc_path (str): The full path to the calculation directory.
    Returns:
        str: The extracted calculation name.
    """

    splitedpath = os.path.split(calc_path)

    if not splitedpath[1][0].isdigit():
        cui_utils.warning(f"{calc_path} may not be calculation path.")

    return os.path.split(splitedpath[0])[1]


def grouping[T](groups: list[T], group_size: int) -> Iterator[list[T]]:
    """
    Groups a list into sub-lists of a specified size.
    Args:
        groups (list[T]): The list to be grouped.
        group_size (int): The desired size of each sub-group.
    Returns:
        Iterator[list[T]]: An iterator yielding sub-lists.

    Example:
    for group in grouping([1,2,3,4,5,6,7,8,9], 3):
        print(group)
        # [[1, 2, 3],
        # [4, 5, 6],
        # [7, 8, 9]]
    """
    for i in range(0, len(groups), group_size):
        yield groups[i : i + group_size]


class _Addable[T](Protocol):
    """
    A Protocol defining types that support the addition operation.
    """

    def __add__(self: T, other: T) -> T: ...


def concat(group: list[_Addable]) -> _Addable:
    """
    Concatenates a list of addable objects using their __add__ method.
    Args:
        group (list[_Addable]): A list of objects that support addition.
    Returns:
        _Addable: The concatenated result.
    """
    temp = group[0]
    for i in range(1, len(group)):
        temp = temp + group[i]
    return temp


def make_analyze_command_script(
    basedir: str, calcname: str, command: str, commandname: str
):
    """
    Generates a bash script to run an analysis command across multiple calculation directories.
    Args:
        basedir (str): The base directory containing the calculation folders.
        calcname (str): The name of the calculation subdirectory (e.g., "4_md").
        command (str): The shell command to execute for analysis (e.g., "echo 0 0 | gmx hbond -f output.xtc -s output.tpr -num hbond.xvg").
        commandname (str): The desired name for the generated script file (e.g., "hbond").

    Example:
        command = "echo 0 0 | gmx hbond -f output.xtc -s output.tpr -num hbond.xvg"
        make_analyze_command_script(
            path, "4_md", command, "hbond"
        )
    """
    dirs = os.listdir(basedir)
    dirs = [f for f in dirs if os.path.isdir(os.path.join(basedir, f))]

    script = ["#!/bin/bash"]
    for folder in dirs:
        fullpath = os.path.join(basedir, folder, calcname)
        if not os.path.exists(fullpath):
            print(f"Folder {fullpath} does not exist")
            continue

        p = "/".join([folder, calcname])

        script.append(f"cd {p}")
        script.append(command)
        script.append("cd ../../")
        script.append("")
        script.append("")

    with open(os.path.join(basedir, f"{commandname}.sh"), "w", newline="\n") as f:
        f.write("\n".join(script))
    print("done")
