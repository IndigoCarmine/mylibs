

from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum
import os
import shutil
from typing import override

from core.cui_utils import format_return_char
import numpy as np


def defaut_file_content(name:str) -> str:
    DefaultFile_dir = os.path.join(os.path.dirname(__file__), "DefaultFiles")
    if not os.path.exists(DefaultFile_dir):
        raise FileNotFoundError("DefaultFiles directory not found")
    with open(os.path.join(DefaultFile_dir, name), "r") as f:
        return f.read()


class Calclation(ABC):
    def __init__(self):
        raise Exception("Abstract class cannot be instantiated")

    @abstractmethod
    def generate(self) -> dict[str,str]:
        Exception("This method must be implemented. Abstract method was called")
        '''
        dict[str,str] : key is the name of the file, value is the content of the file 
        '''
        return {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass



class EM(Calclation):
    '''
    energy minimization
    '''

    def __init__(self, nsteps:int=3000, emtol:float=300, name:str = "em"):
        self.nsteps = nsteps
        self.emtol = emtol
        self.calculation_name = name
    
    @property
    def name(self) -> str:
        return self.calculation_name
    
    @override
    def generate(self) -> dict[str,str]:
        return {
            "setting.mdp": defaut_file_content("em.mdp").format(nsteps = self.nsteps, emtol = self.emtol),
            "grommp.sh": defaut_file_content("grommp.sh"),
            "mdrun.sh": defaut_file_content("mdrun.sh"),
            "ovito.sh": defaut_file_content("em_ovito.sh")
        }


class MDType(enum.Enum):
    v_rescale_c_rescale = 1
    nose_hoover_parinello_rahman = 2
    berendsen = 3

@dataclass
class MD(Calclation):
    '''
    molecular dynamics
    '''
    type:MDType
    calculation_name:str
    nsteps:int = 10000
    gen_vel:str = "yes"
    temperature:float = 300

    
    @property
    def name(self) -> str:
        return self.calculation_name
    
    @override
    def generate(self) -> dict[str,str]:
        match self.type:
            case MDType.v_rescale_c_rescale:
                return {
                    "setting.mdp": defaut_file_content("v_rescale_c_rescale.mdp")
                    .format(nsteps = self.nsteps, gen_vel = self.gen_vel, temperature = self.temperature),   
                    "grommp.sh": defaut_file_content("grommp.sh"),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh")
                }
            case MDType.nose_hoover_parinello_rahman:
                raise NotImplementedError("parrameters are not linked to the mdp file")
                return {
                    "setting.mdp": defaut_file_content("nose_hoover_parinello_rahman.mdp"),
                    "grommp.sh": defaut_file_content("grommp.sh"),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh")
                }
            case MDType.berendsen:
                raise NotImplementedError("parrameters are not linked to the mdp file")
                return {
                    "setting.mdp": defaut_file_content("berendsen.mdp"),
                    "grommp.sh": defaut_file_content("grommp.sh"),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh")
                }
            
class RuntimeSolvation(Calclation):
    '''
    solvation (calculate number of molecules at runtime from the cell size)
    '''
    def __init__(self, solvent:str = "MCH", name:str = "solvation", rate:float = 1.0, ntry:int = 300):
        match solvent:
            case "MCH":
                self.solvent = "MCH"
            case _:
                raise ValueError("Invalid solvent")
            
        self.calculation_name = name
        self.rate = rate
        self.ntry = ntry

    @override
    def generate(self) -> dict[str,str]:
        return {
            "mdrun.sh"  : defaut_file_content("runtime_solvation.sh")
                .replace("SOLVENT", self.solvent)
                .replace("RATE", str(self.rate))
                .replace("TRY", str(self.ntry)),
            f"{self.solvent}.itp": defaut_file_content(f"{self.solvent}.itp"),
            f"{self.solvent}.gro": defaut_file_content(f"{self.solvent}.gro"),
            "runtime_solvation.py": defaut_file_content("runtime_solvation.py"),
            "grommp.sh": 'echo "this is a dummy file for automation"',
        }
    
    @override
    @property
    def name(self) -> str:
        return self.calculation_name
    


class Solvation(Calclation):
    '''
    solvation
    '''
    def __init__(self, solvent:str = "MCH", name:str = "solvation",nmol:int = 100, ntry:int = 300):
        match solvent:
            case "MCH":
                self.solvent = "MCH"
            case _:
                raise ValueError("Invalid solvent")
            
        self.calculation_name = name
        self.nmol = nmol
        self.ntry = ntry
    

    
    @classmethod
    def from_cell_size(cls, cell_size:np.ndarray,name:str = "solvation", solvent:str = "MCH",  rate:float = 1.0):
        '''
        try to fill the cell with the solvent as much as density of the solvent
        rate : the rate of the solvent filling in the cell (when 1.0, the cell is filled with the solvent as much as the density of the solvent)
        '''
        volume = np.prod(cell_size) # nm^3
        print("The volume of the cell is", volume, "nm^3")
        match solvent:
            case "MCH":
                density = 0.77 # g/cm^3
                mass = 98.186 # g/mol
                mass_den = mass / density # cm^3/mol = nm^3/mol * 10e21
                print("MCH is", mass_den, "cm^3/mol")
                print("is", mass_den * 10e21, "nm^3/mol")

            case _:
                raise ValueError("Invalid solvent")
            
        Na = 6.022 * 100  # *10e21 # Avogadro's number
        nmol = int(volume / mass_den * rate * Na) # number of molecules

        print("I will fill the cell with", nmol, "molecules")

        return cls(solvent,name=name, nmol = nmol, ntry = 300)

        


    @property
    def name(self) -> str:
        return self.calculation_name
    @override
    def generate(self) -> dict[str,str]:
        return {
            "mdrun.sh"  : defaut_file_content("solvation.sh")
                .replace("SOLVENT", self.solvent)
                .replace("NMOL", str(self.nmol))
                .replace("TRY", str(self.ntry)),
            f"{self.solvent}.itp": defaut_file_content(f"{self.solvent}.itp"),
            f"{self.solvent}.gro": defaut_file_content(f"{self.solvent}.gro"),
            "solvation.py": defaut_file_content("solvation.py"),
            "grommp.sh": 'echo "this is a dummy file for automation"',
        }

def copy_file_script(extension:str, destination:str) -> str:
    return f"cp *.{extension} ../{destination}"

def copy_inherited_files_script(destination:str) -> str:
    scripts =[
        copy_file_script("top", destination),
        copy_file_script("itp", destination),
        f"cp output.gro ../{destination}/input.gro"
    ]
    return "\n".join(scripts)
    



def launch(calculations:list[Calclation], input_gro:str,working_dir:str,overwrite:bool = False):
    names = [calculation.name for calculation in calculations]
    # check if there are any duplicate names
    if len(names) != len(set(names)):
        raise ValueError("Duplicate names")


    for i in range(len(calculations)):
        calculation = calculations[i]
        dirname = os.path.join(working_dir, str(i) + "_" + calculation.name)
        # create folder in the working directory
        if os.path.exists(os.path.join(dirname)):
            if overwrite:
                # remove the folder and its content
                for file in os.listdir(dirname):
                    os.remove(os.path.join(dirname, file))
                os.rmdir(dirname)
            else:
                raise ValueError("Working directory already exists", dirname)
    
        os.mkdir(dirname)
        
        # generate files
        files = calculation.generate()
        for name, content in files.items():
            with open(os.path.join(dirname, name), "w",newline="\n") as f:
                f.write(content)
        
        if i == len(calculations) - 1:
            '''last calculation'''
            break


        # create a script to copy inherited files
        with open(os.path.join(dirname, "copy.sh"), "w", newline="\n") as f:
            f.write(copy_inherited_files_script(str(i + 1) + "_" + (calculations[i+1].name)))
            f.write(f"\necho {calculation.name} is done")
            f.write(f"\necho Next calculation is {calculations[i+1].name}")

    # copy input file to the first calculation
    shutil.copy2(input_gro, os.path.join(working_dir, "0_" + calculations[0].name, "input.gro"))       


    #create a script to all the calculations

    with open(os.path.join(working_dir, "run.sh"), "w", newline="\n") as f:
        for i,calc in enumerate(calculations):
            f.write(f"cd {str(i) + '_' + calc.name}\n")
            f.write(f"sh grommp.sh\n")
            f.write(f"sh mdrun.sh\n")
            if i != len(calculations) - 1:
                f.write(f"sh copy.sh\n")
            f.write("cd ..\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
        f.write("echo All calculations are done")

def test():
    calculations = [
        EM(name= "firstEM",nstep = 3000, e = 1000),
        Solvation(solvent="MCH",nmol = 100, ntry = 300),
        EM(name= "solvetedEM",nstep = 3000, e = 1000),
        MD(name = "firstMD", type = MDType.v_rescale_c_rescale ),
    ]
    #remove all files in the working directory on all platforms
    # working_dir = r"C:\Users\taman\Dropbox\python\gromacs\worktest"
    # for file in os.listdir(working_dir):
    #     os.remove(os.path.join(working_dir, file))

    launch(calculations, "input.gro",r"C:\Users\taman\Dropbox\python\gromacs\worktest", overwrite = True)


def main():
    test()

if __name__ == "__main__":
    main()