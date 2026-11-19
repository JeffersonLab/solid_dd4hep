#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Convert events of SIDIS-RC EvGen (sidisgen) to HepMC3 for ddsim.

usage:
    sidisgen2hepmc.py EVENT.root -o OUT.hepmc --z=-370:-330 [--raster 0.2] [-n N] [--first N] [--first-event N]
                      [--seed 1] [--summary FILE.json]
    SOLID_GUN=0 ddsim --steeringFile $DETECTOR_PATH/steering/solid_sim.py --inputFiles OUT.hepmc -N N ...

SIDIS-RC EvGen (D. Byer et al., Comput. Phys. Commun. 287 (2023) 108702; github.com/duanebyer/sidis, GPL-3.0) is
the SIDIS generator with lowest-order QED radiative corrections. Its event file must hold the particle momenta:
set `file.write_momenta on` in the parameter file. docs/generators.md has the parameter file for SoLID.

What is written, per event (GeV and mm, hall frame: beam along +z):
    incoming   beam lepton and target nucleon (status 4); the nucleon is at rest
    outgoing   scattered lepton, hadron, and for a radiative event the photon (status 1)
    vertex     uniform along the target (--z=MIN:MAX in cm) and in a disk of radius --raster cm
    weight     cross section carried by the event in nb: the generator's weight divided by the norm of its event
               type, times 0.389379e6 nb GeV^2. The weights of all generated events add up to the cross section
               inside the generator's cuts, per target nucleon. ddsim stores the weight in the event header.

Polarized 3He is not a target of the generator. Generate on the neutron and on the proton (setup.target n, p)
and combine afterwards: the asymmetry comes from the neutron with its effective polarization, the two protons
dilute it.

  -n N              number of events to write (default: all)
  --first N         first event of the file to use (default 0)
  --first-event N   number of the first event written (default 0); keep numbers unique across files
  --summary FILE    write counts, the cross section and the generator parameters as JSON
Needs uproot, awkward, numpy.
"""
import argparse
import json
import sys

import numpy as np

GEV2_TO_NB = 0.3893793721e6
TARGET_PDG = {0: 2212, 1: 2112, 2: 1000010020}
HADRON_PDG = {0: 111, 1: 211, 2: -211, 3: 311, 4: 321, 5: -321, 6: 333, 7: 443}
BEAM_PDG = {0: 11, 1: 13, 2: 15, 3: 11}


def vec(tree, name, start, stop):
    import awkward as ak
    a = tree[name].array(library="ak", entry_start=start, entry_stop=stop)
    return np.stack([ak.to_numpy(a["fP"]["fX"]), ak.to_numpy(a["fP"]["fY"]), ak.to_numpy(a["fP"]["fZ"]),
                     ak.to_numpy(a["fE"])], axis=1)


def rng_pair(text):
    a, b = (float(v) for v in text.split(":"))
    return a, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--z", type=rng_pair, required=True)
    ap.add_argument("--raster", type=float, default=0.2)
    ap.add_argument("-n", type=int)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--first-event", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--summary")
    a = ap.parse_args()

    import uproot
    f = uproot.open(a.file)
    tree = f["events"]
    if "k2" not in tree.keys():
        sys.exit("%s has no momenta: generate with 'file.write_momenta on'" % a.file)
    par = {}
    for k in f.keys():
        if k.startswith("params/"):
            o, name = f[k], k.split("/")[1].split(";")[0]
            cls = getattr(o, "classname", "")
            par[name] = (o.member("fVal") if "TParameter" in cls else
                         [float(v) for v in o] if "TArray" in cls else str(o))
    norm = [float(v) for v in f["stats/norm"]]
    total = tree.num_entries
    start = a.first
    stop = total if a.n is None else min(total, start + a.n)
    if start >= stop:
        sys.exit("no events in the requested range (the file has %d)" % total)
    typ = tree["type"].array(library="np", entry_start=start, entry_stop=stop)
    w = tree["weight"].array(library="np", entry_start=start, entry_stop=stop)
    k1, p, k2, ph, k = (vec(tree, n_, start, stop) for n_ in ("k1", "p", "k2", "ph", "k"))
    # cross section per event; the norms are those of the whole file, so a part of the file carries its share
    sigma = w / np.array(norm)[typ] * GEV2_TO_NB * (total / float(stop - start))
    pdg_beam = BEAM_PDG.get(int(par.get("setup.beam", 0)), 11)
    pdg_target = TARGET_PDG[int(par["setup.target"])]
    pdg_hadron = HADRON_PDG[int(par["setup.hadron"])]
    mass = lambda v: np.sqrt(np.clip(v[:, 3] ** 2 - (v[:, :3] ** 2).sum(axis=1), 0.0, None))
    m_k1, m_p, m_k2, m_ph = mass(k1), mass(p), mass(k2), mass(ph)

    n = stop - start
    rng = np.random.default_rng(a.seed)
    z = rng.uniform(a.z[0], a.z[1], n) * 10.0
    r = a.raster * 10.0 * np.sqrt(rng.uniform(0.0, 1.0, n))
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    x, y = r * np.cos(psi), r * np.sin(psi)
    rad = (typ == 2) & (k[:, 3] > 0.0)
    line = "P %d %d %d %.9e %.9e %.9e %.9e %.9e %d\n"
    with open(a.output, "w") as out:
        out.write("HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\nW sigma_nb\n")
        for i in range(n):
            out.write("E %d 1 %d\nU GEV MM\nW %.9e\n" % (a.first_event + i, 5 if rad[i] else 4, sigma[i]))
            out.write(line % (1, 0, pdg_beam, k1[i, 0], k1[i, 1], k1[i, 2], k1[i, 3], m_k1[i], 4))
            out.write(line % (2, 0, pdg_target, p[i, 0], p[i, 1], p[i, 2], p[i, 3], m_p[i], 4))
            out.write("V -1 0 [1,2] @ %.6e %.6e %.6e 0\n" % (x[i], y[i], z[i]))
            out.write(line % (3, -1, pdg_beam, k2[i, 0], k2[i, 1], k2[i, 2], k2[i, 3], m_k2[i], 1))
            out.write(line % (4, -1, pdg_hadron, ph[i, 0], ph[i, 1], ph[i, 2], ph[i, 3], m_ph[i], 1))
            if rad[i]:
                out.write(line % (5, -1, 22, k[i, 0], k[i, 1], k[i, 2], k[i, 3], 0.0, 1))
        out.write("HepMC::Asciiv3-END_EVENT_LISTING\n")
    summary = {"input": a.file, "output": a.output, "events": int(n), "first": int(start), "events_in_file": int(total),
               "radiative": int(rad.sum()), "cross_section_nb": float(sigma.sum()),
               "cross_section_error_nb": float(np.sqrt((sigma ** 2).sum())), "beam_pdg": pdg_beam, "target_pdg": pdg_target,
               "hadron_pdg": pdg_hadron, "parameters": par}
    print("wrote %s: %d events (%d radiative), target %d, hadron %d; cross section inside the cuts %.4g +- %.2g nb"
          % (a.output, n, rad.sum(), pdg_target, pdg_hadron, summary["cross_section_nb"],
             summary["cross_section_error_nb"]), file=sys.stderr)
    if a.summary:
        with open(a.summary, "w") as fo:
            json.dump(summary, fo, indent=1)
            fo.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
