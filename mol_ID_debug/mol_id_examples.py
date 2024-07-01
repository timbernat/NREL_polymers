'''Minimally-reproducing debug examples for non-unique molecule IDs issue in OpenFF Interchange LAMMPS atom writer'''

from copy import deepcopy
from pathlib import Path
from itertools import product as cartesian_product

from openff.toolkit import Molecule, Topology, ForceField


## basic water box example (all waters will be assigned the same mol ID)
waterbox_path = Path('waterbox.pdb')
watertop = Topology.from_pdb(waterbox_path)

ff = ForceField('tip3p.offxml')
winc = ff.create_interchange(watertop)
winc.to_lammps('water.lammps')


## More complex tiling example (closer to my use case here)
# create prototype Molecule w/ partial charges and coordinates
offmol = Molecule.from_smiles('CC(=O)C')
offmol.generate_conformers(n_conformers=1)
offmol.assign_partial_charges(partial_charge_method='gasteiger')

# determine effective radius of base Molecule to guarantee no overlap
conf = offmol.conformers[0]
COM  = conf.mean(axis=0)
conf_origin = conf - COM # conformer centered at origin

R2 = (conf_origin)**2
radii = R2.sum(axis=1)**0.5
r_eff = radii.max()

# Craete new topology consisting of tiled copies of the Molecule onto a lattice
mols : list[Molecule] = []

s = 3 # sidelength of cubic lattice
one_axis_mults = [i for i in range(s)] 
for coord_mults in cartesian_product(*(one_axis_mults for _ in range(3))): # 3 is for a 3-dimensional lattice
    mol = deepcopy(offmol) # deepcopy to ensure no attributes point to the same objects
    mol.conformers[0] = conf_origin + (2*r_eff*coord_mults) # affine transformation of conformer to each point scaled by effective diameter
    mols.append(mol)
tiled_offtop = Topology.from_molecules(mols)
tiled_offtop.to_file('tiled.pdb', file_format='pdb')

# Create Interchange from FF, write to LAMMPS
ff = ForceField('openff-2.0.0.offxml')
inc = ff.create_interchange(tiled_offtop, charge_from_molecules=mols)
inc.to_lammps('tiled.lammps')