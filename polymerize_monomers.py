# Imports

## Custom Imports
from polysaccharide import logutils

from polysaccharide.molutils import reactions
# from polysaccharide.molutils.rdmol.rdtypes import *
from polysaccharide.molutils.rdmol import rdconvert, rdlabels

from polysaccharide.polymer import monomer as monoutils
from polysaccharide.polymer.monomer import MonomerInfo

## Logging and Shell
import logging
logging.basicConfig(
    level=logging.INFO,
    format=logutils.LOG_FORMATTER._fmt,
    datefmt=logutils.LOG_FORMATTER.datefmt,
    force=True
)

## Numeric imports
import pandas as pd

## File I/O
from pathlib import Path
import json

## Cheminformatics
from rdkit import Chem


# Input parameters
mono_data_path : Path = Path('processed_monomer_data') / 'clean_smarts_digroup.csv'
rxn_mech_dir   : Path = Path('rxn_smarts')
mono_info_dir  : Path = Path('monomer_files')

rxns_from_smarts : bool = True#False


# main code
if __name__ == '__main__':
    # Load processed monomer starting structures
    logging.info(f'Loading processed data from {mono_data_path.stem}')
    digroup_table = pd.read_csv(mono_data_path, index_col=[0])
    tables_by_chem = {
        chemistry : digroup_table[digroup_table['Chemistry'] == chemistry].dropna(axis=1).reset_index(drop=True)
            for chemistry in set(digroup_table['Chemistry'])
    }

    # Defining rxn mechanisms
    reaction_pairs = {
        'NIPU' : ('cyclocarbonate', 'amine'),
        'urethane' : ('isocyanate', 'hydroxyl')
    }

    logging.info(f'Loading reaction mechanisms (from {"SMARTS" if rxns_from_smarts else "MDL files"})')
    if rxns_from_smarts:
        with (rxn_mech_dir / 'rxn_smarts.json').open('r') as rxn_file:
            rxns = {
                chemistry : reactions.AnnotatedReaction.from_smarts(rxn_SMARTS)
                    for chemistry, rxn_SMARTS in json.load(rxn_file).items()
            }
    else:
        # from files
        rxns = {
            chemistry : reactions.AnnotatedReaction.from_rxnfile(rxn_mech_dir / f'{chemistry}.rxn')
                for chemistry in reaction_pairs.keys()
        }

    # Polymerize into well-specified fragments with ports
    mono_info_dir.mkdir(exist_ok=True)
    cvtr = rdconvert.SMILESConverter()

    for chemistry, smarts_table in tables_by_chem.items():
        chem_dir = mono_info_dir / chemistry
        chem_dir.mkdir(exist_ok=True)

        for i, sample in smarts_table.iterrows():
            logging.info(f'Generating fragments for {chemistry} #{i}')
            # look up reactive groups and pathway by chemistry
            rxn_group_names = reaction_pairs[chemistry]
            rxn = rxns[chemistry]

            # read reactant monomers from digroup_table
            initial_reactants = []
            for j, group_name in enumerate(rxn_group_names):
                reactant = Chem.MolFromSmarts(sample[group_name])
                for atom in reactant.GetAtoms():
                    atom.SetProp('reactant_group', group_name)
                
                initial_reactants.append(reactant)
            mono_info = MonomerInfo()

            # first round of polymerization (initiation)
            reactor = reactions.PolymerizationReactor(rxn)
            for dimer, frags in reactor.propagate(initial_reactants):
                for assoc_group_name, rdfragment in zip(rxn_group_names, frags):
                    rdfragment = cvtr.convert(rdfragment) # hacky workaround for RDKit nitrogen bond order SMARTS bug
                    rdlabels.clear_atom_isotopes(rdfragment, in_place=True)

                    affix = 'TERM' if monoutils.is_term_by_rdmol(rdfragment) else 'MID'
                    mono_info.monomers[f'{assoc_group_name}_{affix}'] = Chem.MolToSmarts(rdfragment)

            # monomer post-processing and saving
            for monomer_tag, smarts in mono_info.monomers.items():
                monomer = Chem.MolFromSmarts(smarts)
                monomer = cvtr.convert(monomer) # must convert to-and-from SMILES (since SMARTS-Mols aren't be Kekulized properly)
                Chem.SanitizeMol(monomer) # need to Sanitize, otherwise Kekulize raises unexpected valence errors (despite valence never changing)
                
                Chem.Kekulize(monomer, clearAromaticFlags=True) # convert aromatic bonds to single-double
                rdlabels.assign_ordered_atom_map_nums(monomer, in_place=True) # number monomers

                mono_info.monomers[monomer_tag] = Chem.MolToSmarts(monomer).replace('#0', '*') # ensure wild atoms are marked correctly (rather than as undefined atoms)
            mono_info.to_file(chem_dir / f'{chemistry}_{i}.json')