import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from openmm.app import Simulation
from openmm import Force, NonbondedForce, CustomNonbondedForce
from openmm import MonteCarloBarostat, LangevinMiddleIntegrator

from openmm.unit import atmosphere, kelvin, nanometer
from openmm.unit import femtosecond, picosecond
from openmm.unit import kilojoule_per_mole, kilocalorie_per_mole

NULL_ENERGY = 0.0*kilojoule_per_mole

from openff.toolkit import Molecule, Topology, ForceField
from openff.units import unit as offunit
from openforcefields.openforcefields import get_forcefield_dirs_paths


# ===========================================
# PARAMETERS
# ===========================================

# Directories
topo_dir = Path('Topologies')
energy_df_path = Path('OpenMM_energies.csv')

# forcefield
# ff_name = 'openff-2.0.0.offxml'
ff_name = 'openff_unconstrained-2.0.0.offxml'

# Box sizes
BOX_VECS = np.eye(3) * 10 * nanometer

# Long-range parameters
CUTOFF = 2.0 * nanometer
# CUTOFF_METHOD = NonbondedForce.NoCutoff
# CUTOFF_METHOD = NonbondedForce.CutoffNonPeriodic
CUTOFF_METHOD = NonbondedForce.CutoffPeriodic

DISPERSION = False
SWITCHING  = False

# Thermodynamic/integrator parameters
T = 300*kelvin
P = 1*atmosphere

timestep = 2*femtosecond
friction = 1*picosecond**-1

E_PRECISION : int = 4

# ===========================================
# PREPROCESSING
# ===========================================

# loading forcefield
OPENFF_DIR = Path(get_forcefield_dirs_paths()[0])
ff_path = OPENFF_DIR / ff_name
forcefield = ForceField(ff_path)

# defining reacting functional groups
reaction_pairs = {
    'NIPU' : ('cyclocarbonate', 'amine'),
    'urethane' : ('isocyanate', 'hydroxyl')
}
chemistries = [i for i in reaction_pairs.keys()]

# labels for OpenMM forces
force_names = (
    'vdW pairwise',
    'Electrostatic',
    '1-4 LJ',
    '1-4 Coulomb',
    'Torsion',
    'Angle',
    'Bond'
)

# ===========================================
# EXECUTION
# ===========================================

if __name__ == '__main__':
    data_dicts = []

    for chemistry in chemistries:
        chem_dir = topo_dir / chemistry
        progress = tqdm([path for path in chem_dir.iterdir()]) # unpack into list for progress bar
        for sdf_path in progress:
            mol_name = sdf_path.stem
            progress.set_postfix_str(f'{chemistry} : {mol_name}')

            # ===========================================
            # INTERCHANGE GENERATION
            # ===========================================

            progress.set_description('Loading OpenFF Interchange...')
            offmol = Molecule.from_file(sdf_path)
            offtop = Topology.from_molecules(offmol) 
            offtop.box_vectors = BOX_VECS

            try:
                interchange = forcefield.create_interchange(offtop, charge_from_molecules=[offmol])
            except Exception as e:
                pass

            # specifying thermo/baro to determine ensemble
            integrator = LangevinMiddleIntegrator(T, friction, timestep)
            # extra_forces = [MonteCarloBarostat(P, T, baro_freq)]
            extra_forces = None

            # ===========================================
            # OPENMM CONFIG
            # ===========================================

            progress.set_description('Building OpenMM Simulation...')
            omm_top = interchange.topology.to_openmm()
            omm_sys = interchange.to_openmm(combine_nonbonded_forces=False)
            omm_pos = interchange.positions.m_as(offunit.nanometer)

            ## Setting box vectors for periodic forces
            omm_top.setPeriodicBoxVectors(BOX_VECS)
            omm_sys.setDefaultPeriodicBoxVectors(*BOX_VECS)

            # configuring bound Force objects
            if extra_forces:
                for force in extra_forces:
                    omm_sys.addForce(force)

            ## number all forces into separate force groups for separability
            for i, force in enumerate(omm_sys.getForces()):
                force.setForceGroup(i)

            ## Add labels to default forces
            for force, name in zip(omm_sys.getForces(), force_names):
                force.setName(name)

            ## reconfiguring non-bonded forces
            ### Custom nonbonded (vdW)
            nonbond_custom = omm_sys.getForce(0)
            assert(isinstance(nonbond_custom, CustomNonbondedForce))

            nonbond_custom.setCutoffDistance(CUTOFF)
            nonbond_custom.setUseSwitchingFunction(SWITCHING)
            nonbond_custom.setNonbondedMethod(CUTOFF_METHOD)
            nonbond_custom.setUseLongRangeCorrection(DISPERSION)
    
            ### Default nonbonded (Electrostatics)
            nonbond = omm_sys.getForce(1)
            assert(isinstance(nonbond, NonbondedForce))

            nonbond.setCutoffDistance(CUTOFF)
            nonbond.setNonbondedMethod(CUTOFF_METHOD)
            nonbond.setUseSwitchingFunction(SWITCHING)
            nonbond.setUseDispersionCorrection(DISPERSION)

            # create and register simulation
            sim = Simulation(omm_top, omm_sys, integrator)
            sim.context.setPositions(omm_pos)

            # ===========================================
            # ENERGY EVAL
            # ===========================================

            # extract total and component energies from OpenMM force groups
            data_dict = {
                'Chemistry' : chemistry,
                'Molecule'  : mol_name
            }
            omm_energies = {}

            ## Total Potential
            progress.set_description('Evaluating Potential energy...')
            overall_state = sim.context.getState(getEnergy=True) # get total potential energy
            PE = overall_state.getPotentialEnergy()
            omm_energies['Potential'] = PE

            ## Total Kinetic (to verify no integration is being done)
            progress.set_description('Evaluating Kinetic energy...')
            KE = overall_state.getKineticEnergy()
            omm_energies['Kinetic'] = KE
            assert(KE == NULL_ENERGY)

            ## Individual force contributions
            for i, force in enumerate(sim.system.getForces()):
                progress.set_description(f'Evaluating {force.getName()} energy...')
                state = sim.context.getState(getEnergy=True, groups={i})
                omm_energies[force.getName()] = state.getPotentialEnergy()

            # reformat to desired units and precision
            omm_energies_kcal = {}
            for contrib_name, energy_kj in omm_energies.items():
                energy_kcal = energy_kj.in_units_of(kilocalorie_per_mole)
                omm_energies_kcal[f'{contrib_name} ({energy_kcal.unit.get_symbol()})'] = round(energy_kcal._value, E_PRECISION)

            # compile data
            data_dict = {**data_dict, **omm_energies_kcal}
            data_dicts.append(data_dict)

    # convert records to table, saving if necessary
    omm_table = pd.DataFrame.from_records(data_dicts)
    omm_table.sort_values('Molecule', inplace=True)
    omm_table.set_index(['Chemistry', 'Molecule'], inplace=True)

    if energy_df_path:
        assert(energy_df_path.suffix == '.csv')
        omm_table.to_csv(energy_df_path)