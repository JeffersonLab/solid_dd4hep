# SPDX-License-Identifier: LGPL-3.0-or-later
"""Event files (event.npz, written by solid_evexport): reader, selection, overlay of several events.

The arrays are described in solid_evdisplay.export.  Units: mm, ns, MeV (mc_momentum and mc_mass in GeV).
"""
import glob
import json
import os

import numpy as np

SCHEMA = 1
_ARRAYS = ("trk_id", "trk_parent", "trk_pdg", "trk_ekin", "trk_offset", "pt_pos", "pt_time", "pt_ekin",
           "hit_collection", "hit_cellid", "hit_node", "hit_energy", "hit_time", "hit_pos", "hit_particle",
           "dep_hit", "dep_time", "dep_energy", "dep_pdg", "vol_node", "vol_collection", "vol_energy", "vol_hits",
           "mc_pdg", "mc_gen_status", "mc_sim_status", "mc_charge", "mc_mass", "mc_time", "mc_vertex",
           "mc_momentum", "mc_parent")


class Event:
    """One event: the arrays of an event.npz as attributes, and `meta` (dict)."""

    def __init__(self, arrays, meta, path=None):
        self.meta, self.path = meta, path
        for k in _ARRAYS:
            setattr(self, k, np.asarray(arrays[k]))
        self.collections = meta.get("collections", [])
        self.number = meta.get("event", -1)
        self.numbers = [self.number]

    # ---- collections
    def collection_mask(self, kind=None, photon=None):
        """bool per collection index"""
        return np.array([(kind is None or c["kind"] == kind) and (photon is None or bool(c.get("photon")) == photon)
                         for c in self.collections], bool)

    def calorimeter_energy(self):
        """total energy deposit in calorimeter collections [MeV]"""
        m = self.collection_mask(kind="calorimeter")
        return float(self.hit_energy[m[self.hit_collection]].sum()) if len(m) else 0.0

    # ---- tracks
    def track(self, k):
        """(positions (n, 3), times (n,)) of track k"""
        a, b = int(self.trk_offset[k]), int(self.trk_offset[k + 1])
        return self.pt_pos[a:b], self.pt_time[a:b]

    def generator_particles(self):
        """indices of the MCParticles with generator status 1"""
        return np.nonzero(self.mc_gen_status == 1)[0]


def load_event(src):
    """Event from an event.npz path, or from the dict of arrays solid_evexport builds."""
    if isinstance(src, dict):
        arrays, path = src, None
    else:
        with np.load(src, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        path = src
    try:
        meta = json.loads(str(arrays["meta"]))
    except (KeyError, ValueError):
        raise SystemExit("%s is not an event file (no meta)" % (path or "input"))
    if meta.get("schema", SCHEMA) > SCHEMA:
        print("warning: %s has event schema %s, this reader knows %d" % (path, meta.get("schema"), SCHEMA))
    return Event(arrays, meta, path)


def find_event_files(paths):
    """event.npz files named directly or found as event_*.npz in directories, in name order"""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "event_*.npz")))
        else:
            out.append(p)
    return out


def lepton_pair(ev, pdg=11, min_tracker_hits=3):
    """True if the generator particles hold a lepton and an antilepton from the same parent that both leave
    at least min_tracker_hits hits in the GEM collections (hits linked to the generator particle itself).
    The common parent keeps the scattered beam electron of an electroproduction event from standing in for
    the decay electron."""
    gem = np.array(["GEM" in c["name"] for c in ev.collections], bool)
    if not gem.any() or len(ev.hit_collection) == 0:
        return False
    part = ev.hit_particle[gem[ev.hit_collection]]
    found = {}
    for i in ev.generator_particles():
        code = int(ev.mc_pdg[i])
        if abs(code) == pdg and int((part == i).sum()) >= min_tracker_hits:
            found.setdefault(code, set()).add(int(ev.mc_parent[i]))
    return bool(found.get(pdg, set()) & found.get(-pdg, set()))


def passes(ev, select=None, min_energy=None, min_tracker_hits=3):
    """Event selection shared by solid_evexport and solid_evview.

    select: None, 'pair' (e+e- in acceptance) or 'pair-mu' (mu+mu-); min_energy: calorimeter energy [GeV].
    """
    if min_energy is not None and ev.calorimeter_energy() < 1000.0 * min_energy:
        return False
    if select == "pair":
        return lepton_pair(ev, 11, min_tracker_hits)
    if select == "pair-mu":
        return lepton_pair(ev, 13, min_tracker_hits)
    return True


def select_events(paths, numbers=None, select=None, min_energy=None, min_tracker_hits=3, max_events=None):
    """Load the event files under `paths` and keep those that pass; numbers: event numbers to keep."""
    out = []
    for f in find_event_files(paths):
        ev = load_event(f)
        if numbers and ev.number not in numbers:
            continue
        if passes(ev, select, min_energy, min_tracker_hits):
            out.append(ev)
            if max_events and len(out) >= max_events:
                break
    return out


def merge_events(events):
    """Overlay: one Event holding the tracks and hits of all, energies per volume summed.

    Track IDs and MCParticle indices are only unique within their original event; the merged event keeps
    them as they are, which is enough for drawing.
    """
    if len(events) == 1:
        return events[0]
    first = events[0]
    arr = {}
    npt = nhit = nmc = 0
    parts = {k: [] for k in _ARRAYS}
    offs = [np.zeros(1, np.int64)]
    for ev in events:
        if [c["name"] for c in ev.collections] != [c["name"] for c in first.collections]:
            raise SystemExit("cannot overlay events with different hit collections")
        for k in _ARRAYS:
            v = getattr(ev, k)
            if k == "trk_offset":
                offs.append(v[1:] + npt)
                continue
            if k == "dep_hit":
                v = v + nhit
            elif k == "hit_particle" or k == "mc_parent":
                v = np.where(v >= 0, v + nmc, v)
            parts[k].append(v)
        npt += len(ev.pt_time)
        nhit += len(ev.hit_node)
        nmc += len(ev.mc_pdg)
    for k in _ARRAYS:
        arr[k] = np.concatenate(offs) if k == "trk_offset" else np.concatenate(parts[k])
    # energies per (volume, collection) summed over the events
    key = arr["vol_node"].astype(np.int64) * 65536 + arr["vol_collection"]
    uk, inv = np.unique(key, return_inverse=True)
    arr["vol_energy"] = np.bincount(inv, weights=arr["vol_energy"], minlength=len(uk)).astype(np.float32)
    arr["vol_hits"] = np.bincount(inv, weights=arr["vol_hits"], minlength=len(uk)).astype(np.int32)
    arr["vol_node"] = (uk // 65536).astype(np.int32)
    arr["vol_collection"] = (uk % 65536).astype(np.int16)
    meta = dict(first.meta)
    meta["collections"] = [dict(c, hits=sum(e.collections[i]["hits"] for e in events),
                                energy=sum(e.collections[i]["energy"] for e in events))
                           for i, c in enumerate(first.collections)]
    out = Event(arr, meta)
    out.numbers = [e.number for e in events]
    return out
