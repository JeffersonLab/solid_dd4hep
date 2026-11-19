# Analysis of simulated events: particle tables and acceptance

The first layer between the simulation and the physics projections. It works on simulated hits with the Monte
Carlo truth, not on reconstructed quantities.

| step | tool | output |
|---|---|---|
| single particles, flat in momentum, polar angle, azimuth and along the target | `scripts/analysis/gun_events.py` | HepMC3 file |
| simulation | `ddsim` with `steering/solid_sim.py` | EDM4hep file |
| one row per generated particle with what it left in every detector | `scripts/analysis/particle_table.py` | `table.npz` |
| acceptance in (θ, p), with or without azimuthal segments, compared with GEMC | `scripts/analysis/acceptance.py` | maps, figure |

```
gun_events.py --pdg 11 -n 10000 --p 0.5:8 --theta 6:28 --z=-370:-330 -o e.hepmc
SOLID_GUN=0 SOLID_OPTICAL=0 ddsim --steeringFile $DETECTOR_PATH/steering/solid_sim.py --inputFiles e.hepmc -N 10000 \
        --outputFile sim.edm4hep.root
particle_table.py sim.edm4hep.root -o table.npz
acceptance.py table.npz --config full --plot acceptance.png --gemc acceptance_solid_SIDIS_He3_electron_1e7_201701_output_final.root
```

This is the method of the GEMC studies (`solid_gemc/analysis/acceptance`): particles flat in P and θ, even along
the target, and a map of the detected fraction. The maps of both simulations can therefore be compared bin by
bin, and physics events can be folded with them in the same way as before.

## The particle table

`python/solid_ana/table.py`. For every generator particle (generatorStatus 1):

- kinematics at the vertex: `pdg`, `p`, `theta`, `phi`, `vx`, `vy`, `vz` (GeV, rad, mm);
- for every tracker-type collection (GEM planes, scintillator pads, MRPC) and layer: whether the particle left a
  hit, the position and time of its first hit, and the azimuthal element of that hit (the GEM sector, the
  scintillator pad);
- for every calorimeter collection: the visible energy from the particle and all its descendants, the position
  of the cell with most of it, and the energy of the whole event.

Two properties of ddsim matter here. Showers that start inside the tracking region, which holds the large-angle
calorimeter, are stored as particles of their own: the table adds the energy of all descendants. And the hits of
secondaries that are not stored are booked on their stored ancestor: shower leakage can therefore add tracker
hits behind a calorimeter to a particle's row.

## What counts as detected

`python/solid_ana/accept.py`. A particle is detected in an arm when it left

- hits in enough GEM planes of the arm,
- a hit of its own in the scintillator pads in front of the arm's calorimeter, and
- visible energy in the calorimeter, shower plus preshower, from itself and its descendants.

| arm | GEM planes | default requirement |
|---|---|---|
| forward, SoLID-Lite | 5 forward planes | 4 planes, forward pad, more than 5 MeV visible |
| large angle, SoLID-Lite | 4 large-angle planes | 3 planes, large-angle pad, more than 5 MeV visible |
| forward, full SoLID | planes 2–6 of the six | the same |
| large angle, full SoLID | planes 1–4 | the same |

The pad hit says that the particle itself arrived at the calorimeter front; both electron triggers of the pre-CDR
ask for the pads as well. The calorimeter energy says that a module is there. The shashlik modules show about
20 % of an electron's energy; 5 MeV visible is what a minimum-ionizing particle leaves in 17 of the 194 layers, so
the requirement holds for full showers and for the fast mode below alike. It gives acceptance, not efficiency. No
Cherenkov is asked for, since optical photons are off in these runs.

Without the pad hit (`--no-spd`) the arms come out wider, because shower debris alone can then mark a
calorimeter: 7.2°–15.8° and 15.2°–25.9° for electrons in the full SoLID instead of the 7.3°–15.2° and 16.0°–25.8°
quoted below.

## Azimuthal segments as a selection

For SoLID-Lite the forward detectors may cover only segments of the azimuth
([lite_options.md](lite_options.md)). `acceptance.py --fa-segments 4x24` keeps a particle when the GEM sectors
it hit, its scintillator pad and its calorimeter cell lie inside the segments, which is what a detector built
only there would record. One simulation of a full-azimuth detector then serves every layout of segments. The
table stores detector elements, not the particle's azimuth at the vertex, because the solenoid turns positive
and negative particles in opposite directions on their way to the detectors.

This selection is exact as long as the removed elements were not needed for the particle to reach the kept ones.
The segmented geometry (`solid_SIDIS_He3_lite.xml`) exists to check that.

## Check against GEMC: electrons in the full SoLID

33 486 electrons, flat in 0.5–8 GeV and 6°–28° and along the 40 cm ³He cell, through `solid_SIDIS_He3` with all
materials and the v4c field map, against GEMC's map of 2017 (`solid_gemc/analysis/acceptance/result_SIDIS_He3/201701`:
10⁷ electrons, vacuum-and-absorber geometry, field CLEOv8, particles counted at virtual planes in front of the
calorimeters within 105–220 cm and 83–127 cm). With the same radii (`--ec-r-fa 105:220 --ec-r-la 83:127`):

| | forward arm | large-angle arm |
|---|---|---|
| edges at half of the plateau, this simulation | 8.2° and 15.2° | 16.8° and 24.9° |
| the same, GEMC | 8.2° and 15.1° | 16.7° and 24.7° |
| mean difference where both are above 50 % | −1.8 % | +0.1 % |
| scatter of the difference per bin (1° × 0.5 GeV, about 100 electrons) | 5.9 % | 5.2 % |

The edges agree within 0.2° and the plateau within 2 %; the scatter is the statistical error of this sample. The
forward arm loses about 2 % to the material that GEMC's simple geometry does not have.

![electron acceptance against GEMC](images/acceptance_electron_SIDIS_He3_vs_gemc.svg)

Two things to know when comparing:

- **Where the radius is taken matters.** A cut on the calorimeter cell with most energy instead of on the position
  at the front moves the large-angle window down by 1.2°, because the shower maximum lies deeper and, for an
  inclined track, further out.
- **The calorimeters are larger than those radii.** Without the cut, with every module of the present layout, the
  arms span 7.3°–15.2° and 16.0°–25.8°. How much of the edge modules is usable is a question for the shower
  containment, not for the acceptance.

### Pions

20 000 pions of each charge (fast mode) against GEMC's pion maps have the same edges and, at first, a plateau
11–13 % higher. With real tungsten the collimators of the ³He target are not opaque to hadrons: 35–50 % of
few-GeV pions arrive from the part of the target where electrons are fully blocked. A crosser loses 150–200 MeV;
a pion that misses the collimators loses 3 MeV up to the last large-angle GEM plane and 15 MeV up to the forward
pads. `--p-loss-max 0.05` removes the crossers:

| pions, loss below 50 MeV | forward arm | large-angle arm |
|---|---|---|
| edges, this simulation | 8.1° and 15.1° | 16.8° and 24.8° |
| edges, GEMC | 8.2° and 15.1° | 16.7° and 24.7° |
| mean difference in the plateau, π⁺ / π⁻ | −6 % / −7 % | +1 % / −1 % |

The forward arm is lower because about one pion in ten interacts in the material in front of the forward pads.

## Fast mode for acceptance runs

A shower in the shashlik modules takes most of the simulation time of an electron, nearly all of it in particles
of a few MeV crossing thin layers. An acceptance map only asks whether the particle arrived. With `SOLID_FAST=1`
the `SoLIDShowerStop` stacking action drops the secondaries that are born inside the shower modules with less than
20 MeV (`SOLID_FAST_MIN_EKIN`); the energetic core of the shower still develops and marks the calorimeter.
Everything in front of the shower modules is simulated in full, the preshower with its lead radiator included.

10 000 electrons in the full SoLID in fast mode against the 33 486 with full showers:

| | fast mode | full showers |
|---|---|---|
| time per electron (one core) | 0.09 s | 0.34 s |
| file size per electron | 38 kB | 100 kB |
| edges of the forward arm | 7.3° and 15.3° | 7.3° and 15.2° |
| edges of the large-angle arm | 16.1° and 25.8° | 16.0° and 25.8° |
| pass the calorimeter requirement, of those with GEM planes and pad hit, forward | 98.6 ± 0.3 % | 99.5 % |
| the same, large angle | 98.5 ± 0.2 % | 98.8 % |
| difference per bin of (θ, p) in units of its error: mean, spread | −0.2 and −0.1, 1.0 | – |

So a fast-mode map is right in shape and up to 1 % low in the forward arm. The particles it misses are
electrons that leave the lead radiator only as soft photons. What was tried on the way:

- Raising the Geant4 production cut from 0.7 to 50 mm saves 10 % of the time, no more.
- Dropping every secondary born in a calorimeter loses about 40 % of the electrons: the electron itself often
  ends in the radiator.
- Dropping secondaries in the radiator as well, or only those below 50 MeV, leaves 2–3 % of the arriving
  electrons without a signal, mostly below 1 GeV.

A fast-mode file has **no calorimeter response**: no energy, no E/p, no trigger thresholds. The log says so in its
last lines.

## Trapped particles

A low-energy electron born in a vacuum volume inside the solenoid field can be trapped between two regions of
stronger field. It loses no energy, every step completes, and Geant4's looper protection does not act: one such
event stopped a production job for good. `steering/solid_sim.py` therefore adds the `SoLIDTrackTimeLimit`
stepping action, which stops charged particles still being tracked 100 µs after the event
(`SOLID_MAX_TIME=<ns>`, 0 switches it off). The log reports each stopped track and their number.

## Test

ctest `analysis` (`scripts/analysis/test_analysis.py`): 80 electrons through `ddsim`; the table against the
generated kinematics; GEM sectors and scintillator pads against the hit positions; the segment selection against
the azimuth of the hits; the acceptance tool; the time limit.
