# SPDX-License-Identifier: LGPL-3.0-or-later
"""solid_evexport: EDM4hep output + trajectory file + geometry scene -> one compact event.npz per event.

usage:
    solid_evexport SIM.edm4hep.root --scene scene.npz [--traj traj.bin] [-o OUTDIR]
                   [--event N ...] [--select pair|pair-mu] [--min-energy GEV] [--max-events K]

The three inputs must come from the same ddsim run (SOLID_SCENE and SOLID_TRAJ in steering/solid_sim.py), or
at least from the same geometry.  Needs uproot and numpy; no DD4hep, no ROOT.

What is written (OUTDIR/event_NNNNNN.npz, NNNNNN = EDM4hep event number; schema 1; mm, ns, MeV, momenta GeV):

    meta            JSON: schema, config, geometry_hash, event, run, entry, files, trajectory cuts,
                    collections [{name, kind (tracker | calorimeter), photon, drawn, hits, energy}]
    trk_id, trk_parent, trk_pdg, trk_ekin      recorded tracks (Geant4 track ID, parent ID, PDG code, kinetic
                    energy at the first point); trk_offset (N + 1): track k owns points offset[k]:offset[k+1]
    pt_pos (P, 3), pt_time, pt_ekin            trajectory points in time order within a track
    hit_collection, hit_cellid, hit_node, hit_energy, hit_time, hit_pos (H, 3), hit_particle
                    every hit of every collection: index into meta collections, cellID, scene node of the
                    most specific drawn volume (-1 none, -2 tie), energy, time (calorimeter: earliest
                    contribution), position, MCParticle index (tracker hits; -1 otherwise)
    dep_hit, dep_time, dep_energy, dep_pdg     calorimeter contributions (dep_hit: row in hit_*)
    vol_node, vol_collection, vol_energy, vol_hits   energy and number of hits per drawn volume and collection
    mc_pdg, mc_gen_status, mc_sim_status, mc_charge, mc_mass, mc_time, mc_vertex (M, 3), mc_momentum (M, 3),
    mc_parent       the MCParticles ddsim kept (first parent index, -1 for none)

Hit -> volume: among the drawn volumes of the hit's readout with (cellID & mask) == volume ID, the one with
the most mask bits (solid_evdisplay.hitmap).  The exporter checks, per collection, that the energies per
volume add up to the hit energies, and reports hits without a unique volume; such hits make the exit status 1.
A readout whose volumes are all invisible (the muon observer planes of the DDVCS configurations) has nothing
to map to: its hits are kept with hit_node -1, the collection is flagged drawn = false, and that is no error.

Photon-counting readouts (collections with 'Cherenkov' in the name; filled only when the simulation ran with
optical physics, SOLID_OPTICAL=1) are flagged photon = true: their hits are single photons, and a display
counts them instead of summing energies.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from . import __version__, trajfile
from .event import SCHEMA, load_event, passes
from .hitmap import NO_MATCH, TIE, VolumeMap

GEV = 1000.0     # EDM4hep energies are in GeV, event files in MeV


def _concat_ranges(begin, end):
    """indices begin[i]..end[i]-1 for all i, and the i each belongs to"""
    n = (end - begin).astype(np.int64)
    owner = np.repeat(np.arange(len(n)), n)
    if len(owner) == 0:
        return np.zeros(0, np.int64), owner
    start = np.repeat(begin.astype(np.int64) - np.concatenate([[0], np.cumsum(n)[:-1]]), n)
    return start + np.arange(len(owner)), owner


class SimFile:
    """The hit collections and MCParticles of an EDM4hep file written by ddsim, read with uproot."""

    def __init__(self, path):
        import uproot
        self.path = path
        self.tree = uproot.open(path)["events"]
        self.collections = []          # (name, kind)
        for name, br in self.tree.items(recursive=False):
            tn = br.typename
            if "SimTrackerHitData" in tn:
                self.collections.append((name, "tracker"))
            elif "SimCalorimeterHitData" in tn:
                self.collections.append((name, "calorimeter"))
        self.collections.sort()
        self.entries = self.tree.num_entries
        hdr = self.tree.arrays(["EventHeader/EventHeader.eventNumber", "EventHeader/EventHeader.runNumber"],
                               library="np")
        self.event_number = np.array([int(v[0]) if len(v) else i for i, v in
                                      enumerate(hdr["EventHeader/EventHeader.eventNumber"])], np.int64)
        self.run_number = np.array([int(v[0]) if len(v) else 0 for v in
                                    hdr["EventHeader/EventHeader.runNumber"]], np.int64)
        self._cache = {}

    def _read(self, branches, entry):
        """branch arrays of one entry; read in blocks of 50 entries"""
        block = entry // 50
        key = (tuple(branches), block)
        if key not in self._cache:
            for k in [k for k in self._cache if k[0] == key[0]]:
                del self._cache[k]
            self._cache[key] = self.tree.arrays(list(branches), entry_start=block * 50,
                                                entry_stop=min((block + 1) * 50, self.entries), library="np")
        arr = self._cache[key]
        return [arr[b][entry - block * 50] for b in branches]

    def hits(self, name, kind, entry):
        """dict of arrays for one collection of one entry (energies in MeV)"""
        p = "%s/%s." % (name, name)
        pos = [p + "position.x", p + "position.y", p + "position.z"]
        if kind == "tracker":
            cid, e, t, x, y, z, part = self._read(
                [p + "cellID", p + "eDep", p + "time"] + pos + ["_%s_particle/_%s_particle.index" % (name, name)],
                entry)
            n = len(cid)
            return dict(cellid=cid.astype(np.uint64), energy=e.astype(np.float64) * GEV, time=t.astype(np.float32),
                        pos=np.stack([x, y, z], 1).astype(np.float32).reshape(n, 3),
                        particle=part.astype(np.int32) if len(part) == n else np.full(n, -1, np.int32))
        c = "%sContributions/%sContributions." % (name, name)
        cid, e, x, y, z, b0, b1, idx, ce, ct, cpdg = self._read(
            [p + "cellID", p + "energy"] + pos + [p + "contributions_begin", p + "contributions_end",
                                                 "_%s_contributions/_%s_contributions.index" % (name, name),
                                                 c + "energy", c + "time", c + "PDG"], entry)
        n = len(cid)
        rows, owner = _concat_ranges(b0, b1)
        rows = idx[rows].astype(np.int64) if len(idx) else rows
        tmin = np.full(n, np.inf)
        np.minimum.at(tmin, owner, ct[rows])
        return dict(cellid=cid.astype(np.uint64), energy=e.astype(np.float64) * GEV,
                    time=np.where(np.isfinite(tmin), tmin, 0.0).astype(np.float32),
                    pos=np.stack([x, y, z], 1).astype(np.float32).reshape(n, 3),
                    particle=np.full(n, -1, np.int32),
                    dep_hit=owner.astype(np.int32), dep_time=ct[rows].astype(np.float32),
                    dep_energy=ce[rows].astype(np.float64) * GEV, dep_pdg=cpdg[rows].astype(np.int32))

    def mcparticles(self, entry):
        p = "MCParticles/MCParticles."
        names = ["PDG", "generatorStatus", "simulatorStatus", "charge", "mass", "time", "vertex.x", "vertex.y",
                 "vertex.z", "momentum.x", "momentum.y", "momentum.z", "parents_begin", "parents_end"]
        vals = self._read([p + n for n in names] + ["_MCParticles_parents/_MCParticles_parents.index"], entry)
        v, pidx = dict(zip(names, vals)), vals[-1]
        n = len(v["PDG"])
        parent = np.full(n, -1, np.int32)
        has = v["parents_end"] > v["parents_begin"]
        if has.any() and len(pidx):
            parent[has] = pidx[v["parents_begin"][has].astype(np.int64)]
        return dict(mc_pdg=v["PDG"].astype(np.int32), mc_gen_status=v["generatorStatus"].astype(np.int32),
                    mc_sim_status=v["simulatorStatus"].astype(np.int32), mc_charge=v["charge"].astype(np.float32),
                    mc_mass=v["mass"].astype(np.float64), mc_time=v["time"].astype(np.float32),
                    mc_vertex=np.stack([v["vertex.x"], v["vertex.y"], v["vertex.z"]], 1).reshape(n, 3),
                    mc_momentum=np.stack([v["momentum.x"], v["momentum.y"], v["momentum.z"]], 1).reshape(n, 3),
                    mc_parent=parent)


def is_photon_readout(name):
    return "Cherenkov" in name


def build_event(sim, entry, vmap, traj_points, meta):
    """The arrays of one event.npz and its per-collection check rows.

    Returns (arrays, rows): rows are (collection, hits, unique, no match, tie, hit energy, volume energy,
    readout has drawn volumes).
    """
    out = dict(sim.mcparticles(entry))
    hit = {k: [] for k in ("collection", "cellid", "node", "energy", "time", "pos", "particle")}
    dep = {k: [] for k in ("hit", "time", "energy", "pdg")}
    vol = {k: [] for k in ("node", "collection", "energy", "hits")}
    colls, rows, nhit = [], [], 0
    for ci, (name, kind) in enumerate(sim.collections):
        h = sim.hits(name, kind, entry)
        n = len(h["cellid"])
        node, _ = vmap.match(name, h["cellid"]) if n else (np.zeros(0, np.int64), None)
        hit["collection"].append(np.full(n, ci, np.int16))
        hit["node"].append(node.astype(np.int32))
        for k in ("cellid", "energy", "time", "pos", "particle"):
            hit[k].append(h[k])
        if kind == "calorimeter":
            dep["hit"].append(h["dep_hit"] + nhit)
            for k in ("time", "energy", "pdg"):
                dep[k].append(h["dep_" + k])
        ok = node >= 0
        un, inv = np.unique(node[ok], return_inverse=True)
        ve = np.bincount(inv, weights=h["energy"][ok], minlength=len(un))
        vol["node"].append(un.astype(np.int32))
        vol["collection"].append(np.full(len(un), ci, np.int16))
        vol["energy"].append(ve)
        vol["hits"].append(np.bincount(inv, minlength=len(un)).astype(np.int32))
        esum = float(h["energy"].sum())
        drawn = vmap.is_drawn(name)
        colls.append(dict(name=name, kind=kind, photon=is_photon_readout(name), drawn=drawn, hits=int(n),
                          energy=esum))
        rows.append((name, n, int(ok.sum()), int((node == NO_MATCH).sum()), int((node == TIE).sum()), esum,
                     float(ve.sum()), drawn))
        nhit += n

    def cat(parts, dtype, shape=(0,)):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(shape, dtype)

    out.update(hit_collection=cat(hit["collection"], np.int16), hit_cellid=cat(hit["cellid"], np.uint64),
               hit_node=cat(hit["node"], np.int32), hit_energy=cat(hit["energy"], np.float32),
               hit_time=cat(hit["time"], np.float32), hit_pos=cat(hit["pos"], np.float32, (0, 3)),
               hit_particle=cat(hit["particle"], np.int32),
               dep_hit=cat(dep["hit"], np.int32), dep_time=cat(dep["time"], np.float32),
               dep_energy=cat(dep["energy"], np.float32), dep_pdg=cat(dep["pdg"], np.int32),
               vol_node=cat(vol["node"], np.int32), vol_collection=cat(vol["collection"], np.int16),
               vol_energy=cat(vol["energy"], np.float32), vol_hits=cat(vol["hits"], np.int32))
    if traj_points is not None and len(traj_points):
        trk, offset, p = trajfile.tracks_of(traj_points)
        out.update(trk_id=trk["track"], trk_parent=trk["parent"], trk_pdg=trk["pdg"], trk_ekin=trk["ekin"],
                   trk_offset=offset, pt_pos=np.stack([p["x"], p["y"], p["z"]], 1), pt_time=p["t"],
                   pt_ekin=p["ekin"])
    else:
        out.update(trk_id=np.zeros(0, np.int32), trk_parent=np.zeros(0, np.int32), trk_pdg=np.zeros(0, np.int32),
                   trk_ekin=np.zeros(0, np.float32), trk_offset=np.zeros(1, np.int64),
                   pt_pos=np.zeros((0, 3), np.float32), pt_time=np.zeros(0, np.float32),
                   pt_ekin=np.zeros(0, np.float32))
    m = dict(meta, schema=SCHEMA, event=int(sim.event_number[entry]), run=int(sim.run_number[entry]),
             entry=int(entry), collections=colls)
    out["meta"] = np.array(json.dumps(m))
    return out, rows


def build_parser():
    ap = argparse.ArgumentParser(prog="solid_evexport", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sim", help="EDM4hep ROOT file written by ddsim")
    ap.add_argument("--scene", required=True, help="scene.npz of the same geometry (SOLID_SCENE or solid_scene_dump)")
    ap.add_argument("--traj", default=None, help="trajectory file of the same run (SOLID_TRAJ); without it the "
                    "events have hits but no tracks")
    ap.add_argument("-o", "--outdir", default="events")
    ap.add_argument("--event", type=int, action="append", default=None, metavar="N",
                    help="export only this event number (repeatable)")
    ap.add_argument("--select", choices=("pair", "pair-mu"), default=None,
                    help="keep only events with an e+e- (pair) or mu+mu- (pair-mu) pair in acceptance: both "
                    "generator leptons leave at least --min-tracker-hits GEM hits")
    ap.add_argument("--min-tracker-hits", type=int, default=3)
    ap.add_argument("--min-energy", type=float, default=None, metavar="GEV",
                    help="keep only events with at least this calorimeter energy deposit")
    ap.add_argument("--max-events", type=int, default=None, help="stop after this many exported events")
    ap.add_argument("--sensitive-only", action="store_true",
                    help="map hits to sensitive drawn volumes only (not needed for the SoLID geometries)")
    ap.add_argument("--allow-unmatched", action="store_true",
                    help="exit with status 0 even if hits have no unique drawn volume")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version", version="solid_evexport " + __version__)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    sim = SimFile(a.sim)
    if len(set(sim.event_number.tolist())) != sim.entries:
        # event files are named by this number, and trajectories are matched to hits by it
        print("solid_evexport: the event numbers in %s are not unique (%d events, %d distinct numbers).\n"
              "ddsim copies them from the generator file: number its events first, for example with\n"
              "scripts/event_display/hepmc_prepare.py (lAger 3.7.0 writes 0 for every event)." % (
                  a.sim, sim.entries, len(set(sim.event_number.tolist()))), file=sys.stderr)
        return 1
    scene = np.load(a.scene, allow_pickle=False)
    vmap = VolumeMap(scene, sensitive_only=a.sensitive_only)
    try:
        smeta = json.loads(str(scene["meta"]))
    except (KeyError, ValueError):
        smeta = {}
    meta = dict(config=smeta.get("config", ""), geometry_hash=smeta.get("geometry_hash", ""),
                sim_file=os.path.abspath(a.sim), scene_file=os.path.abspath(a.scene),
                exporter="solid_evexport " + __version__, units="mm, ns, MeV; mc_momentum and mc_mass in GeV")
    by_event = {}
    if a.traj:
        meta.update(traj_file=os.path.abspath(a.traj), traj=trajfile.read_header(a.traj))
        by_event = trajfile.split_events(trajfile.read_trajectories(a.traj))
    os.makedirs(a.outdir, exist_ok=True)
    want = set(a.event) if a.event else None
    total = {}
    written, nbad, npts, nvirtual = [], 0, 0, 0
    for entry in range(sim.entries):
        evn = int(sim.event_number[entry])
        if want is not None and evn not in want:
            continue
        arrays, rows = build_event(sim, entry, vmap, by_event.get(evn), meta)
        if a.select or a.min_energy is not None:
            ev = load_event(arrays)
            if not passes(ev, a.select, a.min_energy, a.min_tracker_hits):
                continue
        path = os.path.join(a.outdir, "event_%06d.npz" % evn)
        np.savez_compressed(path, **arrays)
        written.append(path)
        npts += len(arrays["pt_time"])
        for name, n, uni, none, tie, eh, ev_, drawn in rows:
            t = total.setdefault(name, [0, 0, 0, 0, 0.0, 0.0, drawn])
            for i, v in enumerate((n, uni, none, tie, eh, ev_)):
                t[i] += v
            if drawn:
                nbad += none + tie
            else:
                nvirtual += none
        if a.max_events and len(written) >= a.max_events:
            break
    if a.traj and want is None and not a.select and a.min_energy is None and not a.max_events:
        missing = sorted(set(by_event) - set(int(v) for v in sim.event_number))
        if missing:
            print("warning: the trajectory file has %d events that are not in %s" % (len(missing), a.sim))
    ebad = 0
    if not a.quiet:
        print("%-24s %8s %8s %8s %6s %14s %14s" % ("collection", "hits", "unique", "no match", "tie",
                                                    "hit E [MeV]", "volume E [MeV]"))
    for name in sorted(total):
        n, uni, none, tie, eh, ev_, drawn = total[name]
        if abs(eh - ev_) > 1e-6 * max(abs(eh), 1e-12) and none + tie == 0:
            ebad += 1
        if not a.quiet and n:
            print("%-24s %8d %8d %8d %6d %14.6f %14.6f%s" % (
                name, n, uni, none, tie, eh, ev_, "" if drawn else "  (readout without drawn volumes)"))
    size = sum(os.path.getsize(p) for p in written)
    print("%d events written to %s (%.1f kB per event, %d trajectory points per event) in %.1f s" % (
        len(written), a.outdir, size / 1024.0 / max(len(written), 1), npts // max(len(written), 1),
        time.perf_counter() - t0))
    print("%d hits, %d without exactly one most specific drawn volume; energy per volume %s the hit energy" % (
        sum(t[0] for t in total.values()) - nvirtual, nbad, "differs from" if ebad else "adds up to"))
    if nvirtual:
        print("%d more hits are in readouts without drawn volumes (invisible planes): kept, not shown" % nvirtual)
    if not written:
        print("no event selected")
        return 1
    return 1 if (ebad or (nbad and not a.allow_unmatched)) else 0


if __name__ == "__main__":
    sys.exit(main())
