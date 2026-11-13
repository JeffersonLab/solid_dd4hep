#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Beam electrons on the target, as HepMC3 events for ddsim: the source of the beam background.

usage:
    beam_on_target.py -n EVENTS -N ELECTRONS -o beam.hepmc [--energy 11] [--z -350] [--raster 0.25]
                      [--window NS] [--seed 1] [--first-event 0]
    SOLID_GUN=0 ddsim --steeringFile $DETECTOR_PATH/steering/solid_sim.py --inputFiles beam.hepmc -N EVENTS ...

Every event holds ELECTRONS beam electrons of --energy GeV along +z. They start at z = --z (cm; the default
-350 is in the vacuum of the entrance pipe of the J/psi and DDVCS configurations, 27.5 cm upstream of the
LH2 cell), spread uniformly over a disk of radius --raster (cm) and, with --window, uniformly over that many
ns. Geant4 does the rest: Moller and bremsstrahlung in the target and the windows, and what these make in
the detector. This is how the SoLID GEMC background studies start ("beam on target").

The number of electrons per event is a free choice, not a time window of the experiment: 3 uA are 1.9e4
electrons per ns, so 2000 electrons are the beam of 0.1 ns. Scale accordingly.

Overlay on signal events: give ddsim both files, `--inputFiles signal.hepmc beam.hepmc`; event i of one
is simulated together with event i of the other (keep the event numbers of the two files equal: both
count from --first-event).

HepMC3 layout: each electron is a beam particle (status 4) that ends in its own vertex, from which the same
electron leaves with status 1; ddsim starts primaries at the end vertex of the particles without parents.
Needs numpy only.
"""
import argparse
import math

import numpy as np

M_E = 0.00051099895          # GeV
C_MM_NS = 299.792458


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-n', '--events', type=int, default=100)
    ap.add_argument('-N', '--electrons', type=int, default=1000, help='beam electrons per event (default 1000)')
    ap.add_argument('-o', '--output', default='beam.hepmc')
    ap.add_argument('--energy', type=float, default=11.0, help='GeV (default 11)')
    ap.add_argument('--z', type=float, default=-350.0, help='start z in cm (default -350)')
    ap.add_argument('--raster', type=float, default=0.25, help='raster radius in cm (default 0.25)')
    ap.add_argument('--window', type=float, default=0.0, help='spread the start times over this many ns '
                    '(default 0: all at t = 0)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--first-event', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pz = math.sqrt(a.energy ** 2 - M_E ** 2)
    n = a.electrons
    with open(a.output, 'w') as out:
        out.write('HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\n')
        for i in range(a.events):
            r = 10.0 * a.raster * np.sqrt(rng.random(n))
            phi = 2.0 * math.pi * rng.random(n)
            t = C_MM_NS * a.window * rng.random(n) if a.window > 0 else np.zeros(n)
            out.write('E %d %d %d\nU GEV MM\n' % (a.first_event + i, n, 2 * n))
            for k in range(n):
                out.write('P %d 0 11 0 0 %.9e %.9e %.9e 4\n' % (2 * k + 1, pz, a.energy, M_E))
                out.write('V %d 0 [%d] @ %.6e %.6e %.6e %.6e\n' % (-(k + 1), 2 * k + 1, r[k] * math.cos(phi[k]),
                                                                 r[k] * math.sin(phi[k]), 10.0 * a.z, t[k]))
                out.write('P %d %d 11 0 0 %.9e %.9e %.9e 1\n' % (2 * k + 2, -(k + 1), pz, a.energy, M_E))
        out.write('HepMC::Asciiv3-END_EVENT_LISTING\n')
    print('%d events of %d beam electrons (%g GeV, z = %g cm, raster radius %g cm%s) written to %s' % (
        a.events, n, a.energy, a.z, a.raster, ', %g ns window' % a.window if a.window > 0 else '', a.output))


if __name__ == '__main__':
    main()
