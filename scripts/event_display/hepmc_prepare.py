#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prepare a generator's HepMC3 file for ddsim and the event display: mm, unique event numbers, vertex shift.

usage:
    hepmc_prepare.py IN.hepmc -o OUT.hepmc [--shift X Y Z] [--raster R] [--seed N] [--first-event K]
                     [--skip N] [--max-events N]

What it does to every event, and why:

    length unit mm    DD4hep 1.38 (eic_xl 26.10) reads the position of the primary vertex without the length
                      unit of the file: a vertex written in cm at z = -315 lands at z = -315 mm. Files in mm
                      are read correctly. (Fixed on the DD4hep master branch; harmless to keep.)
    event numbers     0, 1, 2, ... (from --first-event; with --skip N --first-event N the events keep the
                      numbers they have without --skip). ddsim copies the HepMC3 event number into the
                      EDM4hep EventHeader, and solid_evexport names and matches events by it. lAger 3.7.0
                      writes event number 0 for every event.
    --shift X Y Z     moves all vertices by this vector, in cm (for generators that produce events at the
                      origin; lAger can generate along the target directly, see lager_solid_jpsi.json)
    --raster R        moves each event by a point drawn uniformly from a disk of radius R cm in x-y (the
                      beam raster; lAger and most generators put every vertex on the beam axis)

Momenta are converted to GeV if they are not already. Particles, vertices, status codes, weights and
attributes are kept as they are. Needs pyHepMC3 (in eic_xl) and nothing else.
"""
import argparse
import math
import random
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='HepMC3 or HepMC2 ASCII file')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--shift', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'), help='cm')
    ap.add_argument('--raster', type=float, default=0.0, metavar='R', help='raster radius in cm (default 0)')
    ap.add_argument('--seed', type=int, default=1, help='seed of the raster (default 1)')
    ap.add_argument('--first-event', type=int, default=0, help='number of the first event written (default 0)')
    ap.add_argument('--skip', type=int, default=0, help='skip this many events of the input')
    ap.add_argument('--max-events', type=int, default=None)
    a = ap.parse_args()
    try:
        from pyHepMC3 import HepMC3 as hm
    except ImportError:
        sys.exit('hepmc_prepare: pyHepMC3 not found (it is part of eic_xl)')
    reader = hm.deduce_reader(a.input)
    if reader is None or reader.failed():
        sys.exit('hepmc_prepare: cannot read %s' % a.input)
    writer = hm.WriterAscii(a.output)
    rng = random.Random(a.seed)
    nread = nout = 0
    while not reader.failed():
        ev = hm.GenEvent()
        reader.read_event(ev)
        if reader.failed():
            break
        nread += 1
        # one raster point per event read, so that an event gets the same one with and without --skip
        r, phi = 10.0 * a.raster * math.sqrt(rng.random()), 2.0 * math.pi * rng.random()
        if nread <= a.skip:
            continue
        ev.set_units(hm.Units.GEV, hm.Units.MM)
        dx, dy, dz = [10.0 * v for v in a.shift] if a.shift else (0.0, 0.0, 0.0)
        dx, dy = dx + r * math.cos(phi), dy + r * math.sin(phi)
        if dx or dy or dz:
            ev.shift_position_by(hm.FourVector(dx, dy, dz, 0.0))
        ev.set_event_number(a.first_event + nout)
        writer.write_event(ev)
        nout += 1
        if a.max_events is not None and nout >= a.max_events:
            break
    writer.close()
    reader.close()
    if nout == 0:
        sys.exit('hepmc_prepare: no event written (%d read from %s)' % (nread, a.input))
    print('%d events written to %s (GeV, mm, event numbers %d to %d%s%s)' % (
        nout, a.output, a.first_event, a.first_event + nout - 1,
        ', shifted by %g %g %g cm' % tuple(a.shift) if a.shift else '',
        ', raster radius %g cm' % a.raster if a.raster > 0 else ''))


if __name__ == '__main__':
    main()
