#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single particles for acceptance maps: flat in momentum, polar angle, azimuth and along the target.

usage:
    gun_events.py --pdg 11 -n 100000 --p 0.5:11 --theta 5:30 --z=-370:-330 -o e_minus.hepmc [--seed 1]
    SOLID_GUN=0 ddsim --steeringFile $DETECTOR_PATH/steering/solid_sim.py --inputFiles e_minus.hepmc -N 100000 ...

This is the sampling of the GEMC acceptance studies (solid_gemc/analysis/acceptance: flat in P, theta and phi, even
along the target), so that the maps can be compared bin by bin. One particle per event, HepMC3 ASCII, GeV and mm.

  --pdg ID            particle (11 e-, -11 e+, 211 pi+, -211 pi-, 321 K+, -321 K-, 2212 p, 22 photon, 13 mu-)
  --p MIN:MAX         momentum range, GeV
  --theta MIN:MAX     polar angle range, degrees (flat in theta, not in cos theta)
  --phi MIN:MAX       azimuth range, degrees (default 0:360)
  --z=MIN:MAX         vertex range along the beam, cm (hall frame; the 40 cm 3He cell is -370:-330,
                      the 15 cm LH2 cell -322.5:-307.5). Write it with '=': a value that starts with '-' is
                      otherwise read as an option
  --raster R          vertex spread across the beam: uniform in a disk of radius R cm (default 0.2)
Needs numpy only.
"""
import argparse
import sys

import numpy as np

MASS = {11: 0.000510999, 13: 0.1056584, 22: 0.0, 211: 0.13957039, 321: 0.493677, 2212: 0.93827209, 2112: 0.93956542}


def rng_pair(text):
    a, b = (float(v) for v in text.split(":"))
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdg", type=int, required=True)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--p", type=rng_pair, required=True)
    ap.add_argument("--theta", type=rng_pair, required=True)
    ap.add_argument("--phi", type=rng_pair, default=(0.0, 360.0))
    ap.add_argument("--z", type=rng_pair, required=True)
    ap.add_argument("--raster", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--first-event", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    if abs(a.pdg) not in MASS:
        ap.error("no mass known for PDG %d" % a.pdg)
    m = MASS[abs(a.pdg)]
    rng = np.random.default_rng(a.seed)
    p = rng.uniform(a.p[0], a.p[1], a.n)
    th = np.radians(rng.uniform(a.theta[0], a.theta[1], a.n))
    ph = np.radians(rng.uniform(a.phi[0], a.phi[1], a.n))
    z = rng.uniform(a.z[0], a.z[1], a.n) * 10.0
    r = a.raster * 10.0 * np.sqrt(rng.uniform(0.0, 1.0, a.n))
    psi = rng.uniform(0.0, 2.0 * np.pi, a.n)
    x, y = r * np.cos(psi), r * np.sin(psi)
    px, py, pz = p * np.sin(th) * np.cos(ph), p * np.sin(th) * np.sin(ph), p * np.cos(th)
    e = np.sqrt(p * p + m * m)
    with open(a.output, "w") as out:
        out.write("HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\n")
        for i in range(a.n):
            # an incoming copy of the particle (status 4) and the particle itself (status 1) at one vertex
            out.write("E %d 1 2\nU GEV MM\n" % (a.first_event + i))
            out.write("P 1 0 %d %.9e %.9e %.9e %.9e %.9e 4\n" % (a.pdg, px[i], py[i], pz[i], e[i], m))
            out.write("V -1 0 [1] @ %.6e %.6e %.6e 0\n" % (x[i], y[i], z[i]))
            out.write("P 2 -1 %d %.9e %.9e %.9e %.9e %.9e 1\n" % (a.pdg, px[i], py[i], pz[i], e[i], m))
        out.write("HepMC::Asciiv3-END_EVENT_LISTING\n")
    print("wrote %s: %d events of PDG %d, p %g-%g GeV, theta %g-%g deg, z %g-%g cm"
          % (a.output, a.n, a.pdg, a.p[0], a.p[1], a.theta[0], a.theta[1], a.z[0], a.z[1]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
