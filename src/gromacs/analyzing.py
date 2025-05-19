from concurrent.futures import ProcessPoolExecutor
import copy
import os
from typing import Callable, Iterator, Protocol
import MDAnalysis as mda
from functools import partial, total_ordering
from base_utils import cui_utils
import numpy as np
import pandas as pd

import base_utils.typecheck as tc


@total_ordering
class Recorder:
    calc_name: str
    values: list[tuple[str, float]]
    log_data: list[str]

    def __init__(self, calculation_name: str):
        self.calc_name = calculation_name
        self.values = []
        self.log_data = []

    def log(self, val):
        self.log_data.append(str(val))

    def add_value(self, name: str, val: float):
        self.values.append((name, val))

    def add_value_of_default_array_analysis(self, name: str, array: float):
        self.add_value(f"{name}_mean", np.mean(array))
        self.add_value(f"{name}_standard", np.std(array))

    def get_all_valuename(self) -> list[str]:
        return [name for name, _ in self.values]

    def concat(self, recorder: "Recorder"):
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
        if not isinstance(other, Recorder):
            raise TypeError("Cannot compare with Recorder and {}".format(type(other)))

        return self.calc_name == other.calc_name

    def __lt__(self, other):
        if not isinstance(other, Recorder):
            return TypeError("Cannot compare with Recorder and {}".format(type(other)))

        return self.calc_name < other.calc_name


@tc.type_check
def generate_excel(file_name: str, recoders: list[Recorder]):
    """generate excel file from recoders
    Args:
        file_name (str): file name
        recoders (list[Recorder]): recoders

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


@tc.type_check
def mixing_recorders(
    base_recorder_list: list[Recorder], recorder_list: list[Recorder]
) -> list[Recorder]:
    """mixing two recoders
    Args:
        base_recorder_list (list[Recorder]): base recoders
        recorder_list (list[Recorder]): recoders to add
    """
    new_recorders = copy.deepcopy(base_recorder_list)

    for recorder in new_recorders:
        for add_recorder in recorder_list:
            if recorder.calc_name == add_recorder.calc_name:
                recorder.concat(add_recorder)

    return new_recorders


@tc.type_check
def perocess_files(
    calculation_basedir: str,
    last_calc,
    work: Callable[[str], Recorder | None],
    do_parallel: bool = False,
) -> list[Recorder]:
    """

    Args:
        calculation_basedir (str): base directory of calculations
        last_calc (str): last calculation name (e.g. 4_md_main)
        work (Callable[[str], Recorder]): function to work on each calculation
        do_parallel (bool): whether to do parallel processing or not
    Returns:
        list[Recorder]: list of recorders
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


@tc.type_check
def _analyze_trj_inner(
    path: str, work: Callable[[mda.Universe, str], Recorder]
) -> Recorder | None:
    grofile = os.path.join(path, "output.gro")
    xtcfile = os.path.join(path, "output.xtc")
    if not os.path.exists(grofile):
        cui_utils.warning(f"{grofile} does not exist")
        return None
    u = mda.Universe(grofile)
    if xtcfile is not None:
        u.load_new(xtcfile)
    return work(u, path)


@tc.type_check
def analyze_trj(
    calculation_basedir: str,
    last_calc,
    work: Callable[[mda.Universe, str], Recorder | None],
    do_parallel: bool = False,
) -> list[Recorder]:
    return perocess_files(
        calculation_basedir,
        last_calc,
        partial(_analyze_trj_inner, work=work),
        do_parallel,
    )


def get_calcname(calc_path: str) -> str:
    """
    calcpath "...../iQuin/4_md_main" -> "iQuin"
    """

    splitedpath = os.path.split(calc_path)

    if not splitedpath[1][0].isdigit():
        cui_utils.warning(f"{calc_path} may not be calculation path.")

    return os.path.split(splitedpath[0])[1]


@tc.type_check
def grouping[T](groups: list[T], group_size: int) -> Iterator[list[T]]:
    """
    Args:
        groups (list[T]): list to group
        group_size (int): size of each group
    Returns:
        Iterator[list[T]]: grouped list

    example:
    for group in grouping([1,2,3,4,5,6,7,8,9], 3):
        print(group)
        # [[1, 2, 3],
        # [4, 5, 6],
        # [7, 8, 9]]
    """
    for i in range(0, len(groups), group_size):
        yield groups[i : i + group_size]


class _Addable[T](Protocol):
    def __add__(self: T, other: T) -> T: ...


@tc.type_check
def concat(group: list[_Addable]) -> _Addable:
    temp = group[0]
    for i in range(1, len(group)):
        temp = temp + group[i]
    return temp


@tc.type_check
def make_analyze_command_script(
    basedir: str, calcname: str, command: str, commandname: str
):
    """make script file to analyze calculation result
    Args:
        basedir (str): base directory
        calcname (str): calculation directory name
        command (str): command to analyze
        commandname (str): script file name

    example:
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
