'''Define simulation parameters to be used for OpenMM validation'''

from polymerist.openmmtools.parameters import SimulationParameters, IntegratorParameters, ThermoParameters, ReporterParameters
from openmm.unit import kelvin, atmosphere
from openmm.unit import femtosecond, picosecond, nanosecond


state_data_props : dict[str, bool] = {
    'step'            : True,
    'time'            : True,
    'potentialEnergy' : True,
    'kineticEnergy'   : True,
    'totalEnergy'     : True,
    'temperature'     : True,
    'volume'          : True,
    'density'         : True,
    'speed'           : True,
    'progress'        : False,
    'remainingTime'   : False,
    'elapsedTime'     : False
}

sim_params = SimulationParameters(
    integ_params = IntegratorParameters(
        time_step=2*femtosecond,
        total_time=1*nanosecond,
        num_samples=10
    ),
    thermo_params = ThermoParameters(
        ensemble='NVT', #'NPT,
        temperature=300*kelvin,
        pressure=1*atmosphere,
        friction_coeff=1*picosecond**-1,
        barostat_freq=25
    ),
    reporter_params = ReporterParameters(
        report_checkpoint=True,
        report_state     =True,
        report_trajectory=True,
        report_state_data=True,
        traj_ext='dcd',
        state_data=state_data_props,
    )
)
sim_params.to_file('sim_params.json')