#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Convert a LUND event file (GEMC input format, as written by clasdis and the SoLID generators) to HepMC3.

usage:
    lund2hepmc.py IN.dat [IN2.dat ...] -o OUT.hepmc [--max-events N] [--first-event K]
    SOLID_GUN=0 ddsim --steeringFile $DETECTOR_PATH/steering/solid_sim.py --inputFiles OUT.hepmc ...

LUND format: per event one header line (number of particles first; with 10 columns the beam energy is the
7th) and one line per particle:

    index  charge  type  PDG  parent  daughter  px py pz E m [GeV]  vx vy vz [cm]

type 1 marks the particles to be tracked. What is written for each event (GeV, mm, event numbers counting
from --first-event):

    a beam electron and a target proton (status 4): the particles 1 and 2 if they are an electron and a
        nucleon of type 21 (clasdis), else an electron of the header's beam energy and a proton at rest
    one vertex at the position of the first type-1 particle
    every type-1 particle that starts at this vertex, with status 1
    for a type-1 particle that starts elsewhere (clasdis lets JETSET decay K0S, Lambda and other strange
        hadrons in flight): its first ancestor that starts at the vertex, with status 1 and once. Geant4
        then decays it itself, at a place and into a channel of its own choice. A displaced particle without
        such an ancestor in the file is written at the main vertex, and counted in the summary.

Quarks, diquarks and strings of the event record are left out. Weights and cross sections are not carried
over: all events have weight 1. Needs nothing beyond the Python standard library.
"""
import argparse
import sys

M_E, M_P = 0.00051099895, 0.93827208816


def read_events(path):
    """yield (header fields, [particle rows]) per event; a row is (index, type, pdg, parent, p4+m, vertex)"""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            head = line.split()
            if not head:
                continue
            try:
                n = int(float(head[0]))
            except ValueError:
                raise SystemExit('lund2hepmc: %s: unexpected header line: %s' % (path, line.strip()))
            rows = []
            for _ in range(n):
                c = f.readline().split()
                if len(c) < 14:
                    raise SystemExit('lund2hepmc: %s: truncated event (%d of %d particles)' % (path, len(rows), n))
                rows.append((int(c[0]), int(float(c[2])), int(c[3]), int(c[4]),
                             tuple(float(v) for v in c[6:11]), tuple(float(v) for v in c[11:14])))
            yield head, rows


def convert(head, rows):
    """(beam e, target, vertex [cm], [(pdg, px, py, pz, E, m)], number of displaced particles kept as they are)"""
    by_index = {r[0]: r for r in rows}
    final = [r for r in rows if r[1] == 1]
    if not final:
        return None
    vtx = final[0][5]

    def at_vertex(r):
        return max(abs(a - b) for a, b in zip(r[5], vtx)) < 1e-4          # 1 um

    out, seen, loose = [], set(), 0
    for r in final:
        if not at_vertex(r):
            a, hops = r, 0
            while a[3] in by_index and a[3] != a[0] and hops < 50:
                a, hops = by_index[a[3]], hops + 1
                if at_vertex(a) and abs(a[2]) > 100 and a[1] != 21:      # a hadron born at the main vertex
                    break
            else:
                a = None
            if a is None:
                loose += 1
            else:
                r = a
        if r[0] not in seen:
            seen.add(r[0])
            out.append((r[2],) + r[4])
    beam = target = None
    if len(rows) >= 2 and rows[0][1] == 21 and rows[0][2] == 11 and rows[1][1] == 21:
        beam, target = (rows[0][2],) + rows[0][4], (rows[1][2],) + rows[1][4]
    if beam is None:
        e = float(head[6]) if len(head) >= 10 else 11.0
        beam, target = (11, 0.0, 0.0, (e * e - M_E * M_E) ** 0.5, e, M_E), (2212, 0.0, 0.0, 0.0, M_P, M_P)
    return beam, target, vtx, out, loose


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', nargs='+', help='LUND file(s)')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--first-event', type=int, default=0)
    a = ap.parse_args()
    nout = nloose = npart = 0
    with open(a.output, 'w') as out:
        out.write('HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\n')
        for path in a.input:
            for head, rows in read_events(path):
                ev = convert(head, rows)
                if ev is None:
                    continue
                beam, target, vtx, parts, loose = ev
                nloose += loose
                npart += len(parts)
                out.write('E %d 1 %d\nU GEV MM\n' % (a.first_event + nout, len(parts) + 2))
                fmt = 'P %d %d %d %.9e %.9e %.9e %.9e %.9e %d\n'
                for i, p in enumerate((beam, target)):
                    out.write(fmt % ((i + 1, 0) + p + (4,)))
                out.write('V -1 0 [1,2] @ %.6e %.6e %.6e 0\n' % tuple(10.0 * v for v in vtx))
                for i, p in enumerate(parts):
                    out.write(fmt % ((i + 3, -1) + p + (1,)))
                nout += 1
                if a.max_events is not None and nout >= a.max_events:
                    break
            if a.max_events is not None and nout >= a.max_events:
                break
        out.write('HepMC::Asciiv3-END_EVENT_LISTING\n')
    if nout == 0:
        sys.exit('lund2hepmc: no event with a type-1 particle in %s' % ', '.join(a.input))
    print('%d events written to %s (%.1f particles per event%s)' % (
        nout, a.output, npart / nout,
        '; %d displaced particles without an ancestor at the vertex were moved to it' % nloose if nloose else ''))


if __name__ == '__main__':
    main()
