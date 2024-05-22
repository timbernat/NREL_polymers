'''Collection of functions useful throughout the polymer building process'''

from typing import Optional

import re
import numpy as np
from pathlib import Path

from rich.progress import Progress
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Group

from rdkit import Chem
from openmm import Integrator, Context

from openff.interchange import Interchange
from openff.interchange.components.mdconfig import MDConfig
from openff.interchange.interop.openmm._positions import to_openmm_positions

from polymerist.maths.lattices.integral import CubicIntegerLattice
from polymerist.polymers.monomers import specification, MonomerGroup
from polymerist.rdutils.reactions.reactors import PolymerizationReactor

from polymerist.mdtools.openmmtools import serialization
from polymerist.mdtools.openmmtools.forcegroups import impose_unique_force_groups


# PROGRESS TRACKING
def initialize_polymer_progress(num_compounds : int) -> tuple[Group, tuple[int, int, int]]:
    '''Initialize a custom rich Progress Group'''
    # status of individual task
    status_readout = Progress(
        'STATUS:',
        TextColumn(
            '[purple]{task.fields[action]}'
        ),
        '...'
    )
    status_id = status_readout.add_task('[green]Current compound:', action='')

    # textual display of the name of the curent polymer
    compound_readout = Progress(
        'Current compound:',
        TextColumn(
            '[blue]{task.fields[polymer_name]} ({task.fields[mechanism]})',
            justify='right'
        ),
    )
    curr_compound_id  = compound_readout.add_task('[green]Compound:', polymer_name='', mechanism='', total=num_compounds)

    # progress over individual compounds (irrespective of mechanism)
    compound_progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TextColumn(
            '({task.completed} / {task.total})'
        ),
    )
    comp_progress_id = compound_progress.add_task('[blue]Unique compound(s)   ', total=num_compounds)

    # combine progess readouts into unified live console
    group = Group(
        status_readout,
        compound_readout,
        compound_progress,
    )

    return group, (status_id, curr_compound_id, comp_progress_id)


# POLYMERIZATION
def generate_smarts_fragments(reactants_dict : dict[str, Chem.Mol], reactor : PolymerizationReactor) -> MonomerGroup:
    '''Takes a labelled dict of reactant Mols and a PolymerizationReactor object with predefined rxn mechanism
    Returns a MonomerGroup containing all fragments enumerated by the provided rxn'''
    monogrp = MonomerGroup()
    initial_reactants = [reactants for reactants in reactants_dict.values()] # must convert to list to pass to ChemicalReaction
    
    for intermediates, frags in reactor.propagate(initial_reactants):
        for assoc_group_name, rdfragment in zip(reactants_dict.keys(), frags):
            # generate spec-compliant SMARTS
            raw_smiles = Chem.MolToSmiles(rdfragment)
            exp_smiles = specification.expanded_SMILES(raw_smiles)
            spec_smarts = specification.compliant_mol_SMARTS(exp_smiles)

            # record to monomer group
            affix = 'TERM' if MonomerGroup.is_terminal(rdfragment) else 'MID'
            monogrp.monomers[f'{assoc_group_name}_{affix}'] = [spec_smarts]

    return monogrp


# TOPOLOGY PACKING
HILL_REGEX = re.compile(r'([A-Z][a-z]?)[0-9]*?') # break apart hill formula into just unique elements (one capital letter, one or no lowercase letters, any (including none) digits)

def generate_uniform_subpopulated_lattice(max_num_atoms : int, num_atoms_in_mol : int, dimension : int=3) -> CubicIntegerLattice:
    '''Create an integer lattice which accomodates a number of sites while minimizing the size of consecutive voids between empty sites'''
    num_mols = max_num_atoms // num_atoms_in_mol # NOTE: key that this is floor division and not ordinary division
    sidelen = np.ceil(num_mols**(1/dimension)).astype(int) # needed to bypass float-typing for integer-valued quantity
    sidelens = np.array([sidelen]*dimension)
    full_lattice = CubicIntegerLattice(sidelens)

    # determine how many odd and even sublattice sites to sample
    num_even_sites = full_lattice.even_idxs.size
    num_even_to_take = min(num_mols, num_even_sites)     # lower bound on occupancy in d-dims is 0.5**(d-1) (=0.25 when d=3), meaning half lattice is not guaranteed to be occupied
    num_odd_to_take  = max(0, num_mols - num_even_sites) # only choose odd sites if there are any remaining once filling the even sites

    # randomly subsample appropriate amounts of each sublattice
    even_idxs_to_keep = np.random.permutation(full_lattice.even_idxs)[:num_even_to_take] # if the even lattice is unfilled, this improves spread, and if it is full this doesn't matter
    odd_idxs_to_keep  = np.random.permutation(full_lattice.odd_idxs )[:num_odd_to_take ] # populate interstices randmoly to avoid bias towards any part of the box
    idxs_to_keep = np.concatenate([even_idxs_to_keep, odd_idxs_to_keep])
    full_lattice.points = full_lattice.points[idxs_to_keep]

    return full_lattice # TOSELF: naming here no longer makes sense as lattice is not technically full anymore: worth fixing?


# INTERCHANGE AND MD FILE EXPORT
def interchange_to_lammps(interchange : Interchange, lmp_data_path : Path, lmp_input_path : Path) -> None:
    '''Produce LAMMPS input and data files from an OpenFF Interchange'''
    interchange.to_lammps(lmp_data_path) # MD data file
    mdc = MDConfig.from_interchange(interchange)
    # mdc.write_lammps_input(lmp_input_path) # input directive file
    mdc.write_lammps_input(input_file=lmp_input_path, interchange=interchange) # input directive file

    # replacing generic lmp file with data file from above
    with lmp_input_path.open('r') as in_file:
        in_file_block = in_file.read()

    with lmp_input_path.open('w') as in_file:
        in_file.write(
            in_file_block.replace('out.lmp', f'"{lmp_data_path}"') # need surrounding double quotes to allow LAMMPS to read special symbols in filename (if present)
        )

def interchange_to_openmm(interchange : Interchange, integrator : Integrator, omm_top_path : Path, omm_sys_path : Path, omm_state_path : Path, state_params : Optional[dict[str, bool]]=None) -> Context:
    '''Produce OpenMM System and State .xml files from an OpenFF Interchange'''
    if state_params is None:
        state_params = {
        'getPositions'  : True,
        'getVelocities' : True,
        'getForces'     : True,
        'getEnergy'     : True,
        'getParameters' : True,
        'getParameterDerivatives' : True,
        'getIntegratorParameters' : True
    }
    
    system  = interchange.to_openmm(combine_nonbonded_forces=False)
    topology = interchange.to_openmm_topology()
    positions = to_openmm_positions(interchange, include_virtual_sites=True)

    impose_unique_force_groups(system)
    context = Context(system, integrator)
    context.setPositions(positions)

    ## writing OpenMM files
    serialization.serialize_system(omm_sys_path, system)
    serialization.serialize_state_from_context(omm_state_path, context, state_params=state_params)
    serialization.serialize_openmm_pdb(omm_top_path, topology, positions)

    return context