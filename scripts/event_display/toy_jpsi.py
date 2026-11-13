#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Toy generator of J/psi -> e+ e- (or mu+ mu-) events for tests of the event display. NOT a physics
generator: for real events use lAger (docs/event_display.md, "Generators").

usage:
    toy_jpsi.py -n 100 -o jpsi_toy.hepmc [--seed 1] [--lepton e|mu] [--beam-energy 11] [--vertex-z -315]
                [--target-length 15] [--lepton-theta MIN MAX] [--first-event 0]
    SOLID_GUN=0 ddsim --steeringFile $DETECTOR_PATH/steering/solid_sim.py --inputFiles jpsi_toy.hepmc -N 100 ...

It writes HepMC3 ASCII events of exclusive photoproduction, gamma p -> J/psi p -> l+ l- p, with a kinematic
spread that looks like the SoLID J/psi measurement, so that displays and performance tests have realistic
track topologies (two leptons of a few GeV and a recoil proton). What it does:

    photon energy     bremsstrahlung-like, dN/dE ~ 1/E, from the J/psi threshold (8.21 GeV) to the beam energy
    momentum transfer dN/dt ~ exp(b t) between the kinematic limits, b = 1.5 / GeV^2 (--slope)
    J/psi decay       1 + cos^2(theta) in the helicity frame, uniform in azimuth
    vertex            uniform along the target (--vertex-z, --target-length, in cm), 1 mm beam spot

What it leaves out: the scattered beam electron (taken as lost down the beam line), the cross section and
its energy dependence, radiative effects, electroproduction (Q^2 > 0), Fermi motion, backgrounds. The event
weights are all 1. Do not use it for acceptance, rate or resolution studies.

--lepton-theta MIN MAX keeps only events with both leptons inside that range of polar angles (degrees), to
avoid simulating events that miss the detector; the default keeps all events.

The file lists the photon and the target proton as incoming particles and the two leptons and the recoil
proton as final-state particles; the J/psi itself is not written. Needs numpy only.
"""
import argparse
import math
import sys

import numpy as np

M_P, M_JPSI = 0.93827208816, 3.0969
M_LEPTON = {'e': 0.00051099895, 'mu': 0.1056583755}
PDG_LEPTON = {'e': 11, 'mu': 13}


def boost(p4, beta):
    """Lorentz boost of (E, px, py, pz) into the frame moving with velocity -beta (active boost by beta)."""
    b2 = float(beta @ beta)
    if b2 == 0.0:
        return p4.copy()
    g = 1.0 / math.sqrt(1.0 - b2)
    bp = float(beta @ p4[1:])
    out = np.empty(4)
    out[0] = g * (p4[0] + bp)
    out[1:] = p4[1:] + ((g - 1.0) * bp / b2 + g * p4[0]) * beta
    return out


def two_body(m, m1, m2):
    """momentum of the daughters in the rest frame of the mother"""
    return math.sqrt(max((m * m - (m1 + m2) ** 2) * (m * m - (m1 - m2) ** 2), 0.0)) / (2.0 * m)


def frame(z):
    """orthonormal x, y for a given unit z"""
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return x, np.cross(z, x)


def generate(rng, beam, slope, lepton):
    """one event: (photon, target, lepton-, lepton+, proton) four-vectors in the lab, beam along +z"""
    ml = M_LEPTON[lepton]
    e_thr = ((M_JPSI + M_P) ** 2 - M_P ** 2) / (2.0 * M_P)
    eg = e_thr * (beam / e_thr) ** rng.random()                  # dN/dE ~ 1/E
    s = M_P ** 2 + 2.0 * M_P * eg
    w = math.sqrt(s)
    beta_cm = np.array([0.0, 0.0, eg / (eg + M_P)])
    k = (s - M_P ** 2) / (2.0 * w)                               # photon momentum in the CM
    q = two_body(w, M_JPSI, M_P)                                 # J/psi momentum in the CM
    e_j = math.sqrt(q * q + M_JPSI ** 2)
    t0, t1 = M_JPSI ** 2 - 2.0 * k * (e_j - q), M_JPSI ** 2 - 2.0 * k * (e_j + q)     # t at 0 and 180 deg
    u = rng.random()
    t = t0 + math.log(1.0 - u * (1.0 - math.exp(slope * (t1 - t0)))) / slope
    cth = min(1.0, max(-1.0, (t - M_JPSI ** 2 + 2.0 * k * e_j) / (2.0 * k * q)))
    sth, phi = math.sqrt(1.0 - cth * cth), 2.0 * math.pi * rng.random()
    dj = np.array([sth * math.cos(phi), sth * math.sin(phi), cth])
    jpsi_cm = np.concatenate([[e_j], q * dj])
    prot_cm = np.concatenate([[math.sqrt(q * q + M_P ** 2)], -q * dj])
    # decay in the helicity frame: z along the J/psi direction in the CM; 1 + cos^2 by rejection
    while True:
        c = 2.0 * rng.random() - 1.0
        if 2.0 * rng.random() < 1.0 + c * c:
            break
    sd, pd = math.sqrt(1.0 - c * c), 2.0 * math.pi * rng.random()
    x, y = frame(dj)
    dl = sd * math.cos(pd) * x + sd * math.sin(pd) * y + c * dj
    pl = two_body(M_JPSI, ml, ml)
    el = math.sqrt(pl * pl + ml * ml)
    beta_j = jpsi_cm[1:] / jpsi_cm[0]
    lm = boost(boost(np.concatenate([[el], pl * dl]), beta_j), beta_cm)
    lp = boost(boost(np.concatenate([[el], -pl * dl]), beta_j), beta_cm)
    return (np.array([eg, 0.0, 0.0, eg]), np.array([M_P, 0.0, 0.0, 0.0]), lm, lp, boost(prot_cm, beta_cm))


def theta_deg(p4):
    return math.degrees(math.atan2(math.hypot(p4[1], p4[2]), p4[3]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-n', '--events', type=int, default=100)
    ap.add_argument('-o', '--output', default='jpsi_toy.hepmc')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lepton', choices=('e', 'mu'), default='e')
    ap.add_argument('--beam-energy', type=float, default=11.0, help='GeV (default 11)')
    ap.add_argument('--slope', type=float, default=1.5, help='t slope b in 1/GeV^2 (default 1.5)')
    ap.add_argument('--vertex-z', type=float, default=-315.0, help='target centre in cm (default -315, the '
                    'J/psi and DDVCS configurations)')
    ap.add_argument('--target-length', type=float, default=15.0, help='cm (default 15)')
    ap.add_argument('--lepton-theta', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                    help='keep events with both leptons in this polar-angle range, degrees')
    ap.add_argument('--first-event', type=int, default=0, help='event number of the first event (default 0)')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ml, code = M_LEPTON[a.lepton], PDG_LEPTON[a.lepton]
    tried = 0
    with open(a.output, 'w') as out:
        out.write('HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\n')
        for i in range(a.events):
            while True:
                tried += 1
                g, p, lm, lp, pr = generate(rng, a.beam_energy, a.slope, a.lepton)
                if a.lepton_theta is None or all(a.lepton_theta[0] <= theta_deg(v) <= a.lepton_theta[1]
                                                 for v in (lm, lp)):
                    break
                if tried > 100000 * (i + 1):
                    sys.exit('toy_jpsi: no event passes --lepton-theta')
            vx, vy = rng.normal(0.0, 1.0, 2)
            vz = 10.0 * (a.vertex_z + a.target_length * (rng.random() - 0.5))
            out.write('E %d 1 5\nU GEV MM\n' % (a.first_event + i))
            rows = [(1, 0, 22, g, 0.0, 4), (2, 0, 2212, p, M_P, 4), (3, -1, -code, lp, ml, 1),
                    (4, -1, code, lm, ml, 1), (5, -1, 2212, pr, M_P, 1)]
            for pid, vtx, pdg, v, m, status in rows[:2]:
                out.write('P %d %d %d %.9e %.9e %.9e %.9e %.9e %d\n' % (pid, vtx, pdg, v[1], v[2], v[3], v[0], m,
                                                                      status))
            out.write('V -1 0 [1,2] @ %.6e %.6e %.6e 0\n' % (vx, vy, vz))
            for pid, vtx, pdg, v, m, status in rows[2:]:
                out.write('P %d %d %d %.9e %.9e %.9e %.9e %.9e %d\n' % (pid, vtx, pdg, v[1], v[2], v[3], v[0], m,
                                                                      status))
        out.write('HepMC::Asciiv3-END_EVENT_LISTING\n')
    print('%d events written to %s (%d generated, toy J/psi -> %s+ %s-)' % (a.events, a.output, tried, a.lepton,
                                                                          a.lepton))


if __name__ == '__main__':
    main()
