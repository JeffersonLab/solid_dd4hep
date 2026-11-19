#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hit rates of the tracker-type detectors from beam-on-target simulations.

usage:
    rates.py SIM.edm4hep.root [...] --electrons-per-event N --current UA [-o RATES.npz] [--encodings-from FILE]
             [--gem-emin 26e-9] [--r-step 5] [--window NS]

The input is a simulation of beam electrons on the target (scripts/event_display/beam_on_target.py, N electrons
per event, full physics, not the fast mode). For every tracker-type collection (GEM planes, scintillator pads,
MRPC) and layer the hits are counted in rings of --r-step cm and turned into a rate at the beam current --current
(in uA; 1 uA = 6.2415e12 electrons per second):

    rate per area   hits / (electrons x ring area) x electrons per second          kHz/cm2
    rate per element, for collections with pads (ring field in the cell ID): hits per pad                 kHz
    occupancy       with --window NS: rate per element x window (pads), for comparison with the pre-CDR tables

A GEM hit counts when it left at least --gem-emin GeV in the drift gap (default 26 eV, one primary ionization);
every other hit counts as stored. Several steps of one particle in one volume are one hit in ddsim.

Statistical errors are sqrt(hits); a ring with fewer than about 100 hits says little. The simulation has the
electromagnetic background and Geant4's photo- and electro-nuclear processes; the hadron background of the GEMC
studies (a dedicated generator) is not in it.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

from solid_ana.table import _collections, decode, parse_encoding, read_encodings  # noqa: E402

E_PER_UA = 6.241509e12


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--electrons-per-event", type=float, required=True)
    ap.add_argument("--current", type=float, required=True)
    ap.add_argument("--gem-emin", type=float, default=26e-9)
    ap.add_argument("--r-step", type=float, default=5.0)
    ap.add_argument("--window", type=float, default=0.0)
    ap.add_argument("--encodings-from")
    ap.add_argument("-o", "--output")
    a = ap.parse_args()

    import awkward as ak
    import uproot

    enc_text = {}
    if a.encodings_from:
        enc_text = read_encodings(uproot.open(a.encodings_from))
    redges = np.arange(0.0, 300.0 + a.r_step, a.r_step)
    area = np.pi * (redges[1:] ** 2 - redges[:-1] ** 2)
    hist, pads, nev = {}, {}, 0
    for path in a.files:
        f = uproot.open(path)
        tree = f["events"]
        enc = dict(read_encodings(f))
        enc.update(enc_text)
        nev += tree.num_entries
        for name, kind in _collections(tree).items():
            if kind != "tracker" or "Cherenkov" in name:
                continue
            e = parse_encoding(enc[name]) if enc.get(name) else {}
            for start in range(0, tree.num_entries, 200):
                arr = tree.arrays(["%s/%s.cellID" % (name, name), "%s/%s.eDep" % (name, name), "%s/%s.position.x" % (name, name),
                                   "%s/%s.position.y" % (name, name)], entry_start=start, entry_stop=start + 200, library="ak")
                cid = ak.to_numpy(ak.flatten(arr["%s/%s.cellID" % (name, name)]))
                if len(cid) == 0:
                    continue
                ed = ak.to_numpy(ak.flatten(arr["%s/%s.eDep" % (name, name)]))
                x = ak.to_numpy(ak.flatten(arr["%s/%s.position.x" % (name, name)])) / 10.0
                y = ak.to_numpy(ak.flatten(arr["%s/%s.position.y" % (name, name)])) / 10.0
                ok = ed >= a.gem_emin if "GEM" in name else np.ones(len(cid), bool)
                layer = decode(cid, e, "layer") if "layer" in e else np.ones(len(cid), np.int64)
                r = np.hypot(x, y)
                for l in np.unique(layer[ok]):
                    m = ok & (layer == l)
                    h, _ = np.histogram(r[m], bins=redges)
                    key = (name, int(l))
                    hist[key] = hist.get(key, 0) + h
                if "ring" in e and "sector" in e:
                    ring, sec = decode(cid, e, "ring"), decode(cid, e, "sector")
                    for rg in np.unique(ring):
                        cnt = np.bincount(sec[ring == rg], minlength=sec.max() + 1)
                        key = (name, int(rg))
                        old = pads.get(key, np.zeros(0))
                        n = max(len(old), len(cnt))
                        pads[key] = np.pad(old, (0, n - len(old))) + np.pad(cnt, (0, n - len(cnt)))
    electrons = nev * a.electrons_per_event
    per_s = a.current * E_PER_UA
    print("%d events x %g electrons = %.3g beam electrons; rates at %g uA" % (nev, a.electrons_per_event, electrons, a.current))
    out = {"r_edges": redges, "electrons": electrons, "current_uA": a.current}
    print("%-22s %5s %10s %9s %14s %12s" % ("collection", "layer", "hits", "r range", "highest ring", "kHz/cm2 there"))
    for (name, l), h in sorted(hist.items()):
        rate = h / (electrons * area) * per_s / 1e3
        err = np.sqrt(h) / (electrons * area) * per_s / 1e3
        nz = np.nonzero(h)[0]
        big = np.nonzero(h >= 25)[0]
        k = big[np.argmax(rate[big])] if len(big) else nz[np.argmax(rate[nz])]
        print("%-22s %5d %10d %4.0f-%-4.0f %6.0f-%-4.0f cm %8.1f +- %.1f"
              % (name, l, h.sum(), redges[nz[0]], redges[nz[-1] + 1], redges[k], redges[k + 1], rate[k], err[k]))
        out["%s_L%d_rate_kHz_cm2" % (name, l)] = rate
        out["%s_L%d_hits" % (name, l)] = h
    if pads:
        print("%-22s %5s %10s %16s %16s%s" % ("pads", "ring", "hits", "mean kHz per pad", "max kHz per pad", "   occupancy in %g ns" % a.window if a.window else ""))
        for (name, rg), cnt in sorted(pads.items()):
            c = cnt[1:] if len(cnt) > 1 else cnt          # sector numbers start at 1
            khz = c / electrons * per_s / 1e3
            occ = "   %.3f mean" % (khz.mean() * 1e3 * a.window * 1e-9) if a.window else ""
            print("%-22s %5d %10d %16.1f %16.1f%s" % (name, rg, c.sum(), khz.mean(), khz.max(), occ))
            out["%s_ring%d_kHz" % (name, rg)] = khz
    if a.output:
        np.savez_compressed(a.output, **out)
        print("wrote %s" % a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
