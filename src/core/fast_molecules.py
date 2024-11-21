import gc
from scipy.spatial import distance_matrix
import core.molecules as mol
import numpy as np
from scipy.spatial.transform import Rotation
from copy import deepcopy

class FastStructWapper[T: mol.IMolecule]():
    def __init__(self, molecule: T):
        self.molecule = molecule
        self.n_atoms = len(molecule.get_children())
        self.coordinates = [atom.coordinate for atom in molecule.get_children()]
        self.n_mols = 1


    def translate(self, coordinate: np.ndarray):
        self.coordinates += coordinate
    
    def rotate(self, rotation: Rotation):
        self.coordinates = rotation.apply(self.coordinates)
    
    def generate_as_molecule(self,Type: type) -> T:
        old_atoms = self.molecule.get_children()
        n_old = len(old_atoms)
        atoms =[]
        for i,coordinate in enumerate(self.coordinates):
            atoms.append(deepcopy(old_atoms[i%n_old]))
            atoms[-1].coordinate = coordinate
        return Type.make(atoms)
    def reset(self):
        self.coordinates = [atom.coordinate for atom in self.molecule.get_children()]
        self.n_mols = 1
    def replicate(self, n: int):
        self.coordinates = np.tile(self.coordinates, (n,1))
        gc.collect()
        self.n_mols = n

    def translate_one(self, index: int, coordinate: np.ndarray):
        self.coordinates[index*self.n_atoms:(index+1)*self.n_atoms] += coordinate
    
    def rotate_one(self, index: int, rotation: Rotation):
        self.coordinates[index*self.n_atoms:(index+1)*self.n_atoms] = rotation.apply(self.coordinates[index*self.n_atoms:(index+1)*self.n_atoms])

    def linear_translate(self, vector: np.ndarray):
        for i in range(self.n_mols):
            self.translate_one(i, vector*i)
    
    def linear_rotate(self, rotation: Rotation):
        for i in range(self.n_mols):
            self.rotate_one(i, rotation**i)
    
    # def reset(self):
    #     for i in range(self.n_mols):
    #         self.co
    
    def is_too_close(self, distance: float=0.09953893710503443) -> bool:
        #pairwise distance
        dist:np.ndarray = distance_matrix(self.coordinates,self.coordinates)
        #check if there is any distance less than 0.1 except for the same atom
        a = dist.flatten()
        a = a[a!=0]
        # print(np.min(a))
        return np.any(a<distance)


        
