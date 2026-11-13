# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Generic ddsim / npsim steering for all SoLID configurations.

    source <prefix>/bin/thissolid.sh solid_JPsi_lite        # selects DETECTOR_CONFIG
    ddsim --steeringFile ${DETECTOR_PATH}/steering/solid_sim.py \
          --compactFile ${DETECTOR_PATH}/${DETECTOR_CONFIG}.xml -N 100 --outputFile out.edm4hep.root

Environment switches (all optional):
    SOLID_OPTICAL=0|1   enable Cherenkov + optical-photon physics and photon-counting readout for the
                        gas Cherenkovs (default 1; set 0 for fast tracking/calorimetry studies)
    SOLID_GUN=0|1       use the built-in particle gun (default 1) - disable when using --inputFiles
    SOLID_SCENE=FILE    write the geometry scene for drawings (scene.npz for solid_lineview / make_solid_views)
                        when the run is initialised; the same file as `solid_scene_dump -o FILE` writes without
                        ddsim. SOLID_SCENE_SEGMENTS=N sets the line segments per circle (default 72).
    SOLID_TRAJ=FILE|1   record trajectory points with time and kinetic energy for event displays (solid_evexport,
                        solid_evview): the SoLIDTrajectoryRecorder stepping action writes FILE (1: solid_traj.bin).
                        SOLID_TRAJ_MINEKIN=E keeps secondaries born with at least E MeV (default 5);
                        SOLID_TRAJ_MAXDIST (mm, default 20) and SOLID_TRAJ_MAXANGLE (rad, default 0.04) set the
                        point spacing. Works with -j N: all threads write the one file.
                        SOLID_TRAJ_EVERY=N (or N:R) records only the events whose number is 0 (or R) modulo N.
                        SOLID_TRAJ_TIMING=FILE appends the CPU time of every event to FILE
                        (scripts/event_display/recorder_overhead.py).
Command-line options of ddsim override everything set here.
"""
import os

from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import GeV, cm, deg, mm

SIM = DD4hepSimulation()

# --------------------------------------------------------------------------------------
# geometry: default to $DETECTOR_PATH/$DETECTOR_CONFIG.xml (thissolid.sh sets both)
_path = os.environ.get('DETECTOR_PATH', '.')
_cfg = os.environ.get('DETECTOR_CONFIG', 'solid')
SIM.compactFile = [os.path.join(_path, _cfg + '.xml')]

# --------------------------------------------------------------------------------------
# physics and field tracking
SIM.physics.list = 'FTFP_BERT'
SIM.physics.rangecut = 0.7 * mm
SIM.field.stepper = 'ClassicalRK4'
SIM.field.equation = 'Mag_UsualEqRhs'
SIM.field.eps_min = 5e-05 * mm
SIM.field.eps_max = 0.001 * mm
SIM.field.min_chord_step = 0.01 * mm
SIM.field.delta_chord = 0.25 * mm
SIM.field.delta_intersection = 0.001 * mm
SIM.field.delta_one_step = 0.01 * mm
SIM.field.largest_step = 10 * mm

# keep hits with zero energy deposit in trackers (needed for photon-counting readouts too)
SIM.filter.tracker = 'edep0'

# --------------------------------------------------------------------------------------
# gas Cherenkovs: count optical photons on the photocathode planes
CHERENKOVS = ['LightGasCherenkov', 'HeavyGasCherenkov', 'LiteCherenkov']
if os.environ.get('SOLID_OPTICAL', '1') != '0':

    def setup_optical(kernel):
        from DDG4 import PhysicsList
        seq = kernel.physicsList()
        cerenkov = PhysicsList(kernel, 'Geant4CerenkovPhysics/CerenkovPhys')
        cerenkov.MaxNumPhotonsPerStep = 10
        cerenkov.MaxBetaChangePerStep = 10.0
        cerenkov.TrackSecondariesFirst = False
        cerenkov.VerboseLevel = 0
        cerenkov.enableUI()
        seq.adopt(cerenkov)
        optical = PhysicsList(kernel, 'Geant4OpticalPhotonPhysics/OpticalGammaPhys')
        optical.addParticleConstructor('G4OpticalPhoton')
        optical.VerboseLevel = 0
        optical.enableUI()
        seq.adopt(optical)
        return None

    SIM.physics.setupUserPhysics(setup_optical)
    SIM.filter.filters['opticalphotons'] = dict(
        name='ParticleSelectFilter/OpticalPhotonSelector',
        parameter={'particle': 'opticalphoton'},
    )
    for det in CHERENKOVS:
        SIM.filter.mapDetFilter[det] = 'opticalphotons'
        SIM.action.mapActions[det] = 'Geant4OpticalTrackerAction'
    # keep the photon -> hit association
    SIM.part.userParticleHandler = ''

# --------------------------------------------------------------------------------------
# geometry scene for drawings: the SoLIDSceneDump plugin (libsolid_ddg4) walks the Geant4 geometry of this job
_scene = os.environ.get('SOLID_SCENE', '')
if _scene:

    # ddsim executes this file and calls the function later, without the module globals: bind the values now
    def setup_scene_dump(kernel, output=os.path.abspath(_scene), config=_cfg,
                         segments=int(os.environ.get('SOLID_SCENE_SEGMENTS', '72'))):
        import DDG4
        master = DDG4.Geant4(kernel).master()
        dump = DDG4.DetectorConstruction(master, 'SoLIDSceneDump/SceneDump')
        dump.Output = output
        dump.Segments = segments
        dump.Config = config
        master.detectorConstruction(True).adopt(dump)
        return None

    SIM.physics.setupUserPhysics(setup_scene_dump)

# --------------------------------------------------------------------------------------
# trajectories for event displays: the SoLIDTrajectoryRecorder stepping action (libsolid_ddg4). It does not
# touch the particle handler, so the MCParticle content of the output is the same with and without it.
_traj = os.environ.get('SOLID_TRAJ', '')
if _traj and _traj != '0':
    _every = (os.environ.get('SOLID_TRAJ_EVERY', '1') + ':0').split(':')
    _timing = os.environ.get('SOLID_TRAJ_TIMING', '')
    SIM.action.step = {
        'name': 'SoLIDTrajectoryRecorder/TrajectoryRecorder',
        'parameter': {
            'OutputFile': os.path.abspath('solid_traj.bin' if _traj == '1' else _traj),
            'MinEkin': float(os.environ.get('SOLID_TRAJ_MINEKIN', '5')),
            'MaxDistance': float(os.environ.get('SOLID_TRAJ_MAXDIST', '20')),
            'MaxAngle': float(os.environ.get('SOLID_TRAJ_MAXANGLE', '0.04')),
            'Every': int(_every[0]),
            'Remainder': int(_every[1]),
            'TimingFile': os.path.abspath(_timing) if _timing else '',
        },
    }

# --------------------------------------------------------------------------------------
# default particle gun: electrons from the target center into the SoLID acceptance
_target_z = {
    'solid': -350, 'solid_SIDIS_He3': -350, 'solid_SIDIS_NH3': -350, 'solid_SIDIS_He3_lite': -350,
    'solid_JPsi': -315, 'solid_DDVCS': -315, 'solid_JPsi_lite': -315, 'solid_DDVCS_lite': -315,
    'solid_PVDIS': 10,
}.get(_cfg, -350)
if os.environ.get('SOLID_GUN', '1') != '0':
    SIM.enableGun = True
    SIM.gun.particle = 'e-'
    SIM.gun.energy = 4 * GeV
    SIM.gun.position = (0.0, 0.0, _target_z * cm)
    SIM.gun.distribution = 'uniform'
    SIM.gun.thetaMin = (22 if _cfg == 'solid_PVDIS' else 8) * deg
    SIM.gun.thetaMax = (35 if _cfg == 'solid_PVDIS' else 24) * deg
    SIM.gun.phiMin = 0 * deg
    SIM.gun.phiMax = 360 * deg

SIM.numberOfEvents = 10
SIM.outputFile = 'solid_sim.edm4hep.root'
