# SPDX-License-Identifier: LGPL-3.0-or-later
"""Particle table: one row per generated particle, with what it left in every detector.

The table is the common input of the acceptance, resolution and rate studies. It is made once per simulated file
(ddsim, EDM4hep) and holds, for every generator particle (generatorStatus 1):

  event, pdg, p, theta, phi, vx, vy, vz           at the vertex (GeV, rad, mm)
  parent                                          PDG code of the particle's first parent in the event record (0: none)
  for every tracker-type collection C (GEM planes, scintillator pads, MRPC), per layer:
      C_hit[row, layer]      True if the particle itself left a hit in that layer
      C_x, C_y, C_z, C_t     position (mm) and time (ns) of its first hit there
      C_p[row, layer]        its momentum (GeV) at that hit: what is left after the material in front
      C_elem[row, layer]     azimuthal element of that hit: the "module" or "sector" field of the cell ID
      C_ring[row, layer]     the "ring" field, where the read-out has one
  for every calorimeter collection C:
      C_e[row]               energy (GeV) deposited by the particle and all its descendants
      C_x, C_y               position (mm) of the cell with the largest such energy
      C_etot[row]            energy of the whole event in that collection

"Itself" means the hit's MC-particle link points to the generator particle; hits of secondaries that the
simulation keeps as separate particles are not counted, but ddsim books the hits of secondaries it does not store
on their stored ancestor, so shower leakage can add hits behind a calorimeter. Calorimeter energy is summed over the particle and its
descendants (ddsim stores secondaries born inside the tracking region, which holds the large-angle calorimeter,
as particles of their own).

A selection of azimuthal segments is applied to a table afterwards (solid_ana.accept), on the elements and cell
positions stored here, so one simulation of a full-azimuth detector serves every choice of segments.
"""
from __future__ import annotations

import json

import numpy as np

TRACKER_TYPES = ("edm4hep::SimTrackerHitData",)
CALO_TYPES = ("edm4hep::SimCalorimeterHitData",)


def parse_encoding(text):
    """'system:8,layer:4,module:6' or 'name:offset:width' -> {name: (shift, width, signed)}"""
    out, shift = {}, 0
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            name, width = parts[0], int(parts[1])
        else:
            name, shift, width = parts[0], int(parts[1]), int(parts[2])
        out[name] = (shift, abs(width), width < 0)
        shift += abs(width)
    return out


def decode(cell_id, enc, name):
    shift, width, signed = enc[name]
    v = (cell_id >> np.uint64(shift)) & np.uint64((1 << width) - 1)
    v = v.astype(np.int64)
    if signed:
        v = np.where(v >= (1 << (width - 1)), v - (1 << width), v)
    return v


def read_encodings(f):
    """cell-ID encodings from the file's metadata: {collection: encoding string}"""
    out = {}
    for tree in ("metadata", "podio_metadata", "runs"):
        if tree not in f:
            continue
        t = f[tree]
        keys = [k for k in t.keys() if k.endswith("GPStringKeys")]
        vals = [k for k in t.keys() if k.endswith("GPStringValues")]
        if not keys or not vals or t.num_entries == 0:
            continue
        ks = t[keys[0]].array(library="np")
        vs = t[vals[0]].array(library="ak")
        for i in range(len(ks)):
            for j, k in enumerate(list(ks[i])):
                if str(k).endswith("__CellIDEncoding"):
                    v = vs[i][j]
                    out[str(k)[:-len("__CellIDEncoding")]] = str(v[0]) if len(v) else ""
    return out


def _collections(tree):
    """{name: 'tracker' | 'calo'} from the branch types"""
    out = {}
    for name in tree.keys(recursive=False):
        if name.startswith("_"):
            continue
        tn = tree[name].typename
        if any(t in tn for t in TRACKER_TYPES):
            out[name] = "tracker"
        elif any(t in tn for t in CALO_TYPES):
            out[name] = "calo"
    return out


def _read_flat(tree, branch, start=None, stop=None):
    import awkward as ak
    a = tree[branch].array(library="ak", entry_start=start, entry_stop=stop)
    return ak.to_numpy(ak.num(a)), ak.to_numpy(ak.flatten(a))


def build_table(path, encodings=None, step=2000):
    """Read one EDM4hep file; returns (arrays, info): dict of numpy arrays and a description (JSON-able).

    The file is read `step` events at a time, which bounds the memory. `encodings` ({collection: encoding string})
    replaces or supplies the cell-ID encodings: a file that was not closed (a recovered file) has no metadata."""
    import uproot

    f = uproot.open(path)
    tree = f["events"]
    enc_text = dict(read_encodings(f))
    enc_text.update(encodings or {})
    cols = _collections(tree)
    parts, info = [], None
    n = tree.num_entries
    for start in range(0, max(n, 1), step):
        t, i = _build_range(tree, enc_text, cols, start, min(n, start + step), path)
        parts.append(t)
        if info is None:
            info = i
        else:
            for name, c in i["collections"].items():
                if c.get("layers", 0) > info["collections"][name].get("layers", 0):
                    info["collections"][name] = c
    arrays = parts[0] if len(parts) == 1 else concat_tables(parts, renumber=False)
    info.update(events=int(n), rows=int(len(arrays["pdg"])))
    return arrays, info


def _build_range(tree, enc_text, cols, start, stop, path):
    def _flat(tree, branch):
        return _read_flat(tree, branch, start, stop)

    n_mc, pdg = _flat(tree, "MCParticles/MCParticles.PDG")
    _, status = _flat(tree, "MCParticles/MCParticles.generatorStatus")
    mom = [_flat(tree, "MCParticles/MCParticles.momentum.%s" % c)[1] for c in "xyz"]
    vtx = [_flat(tree, "MCParticles/MCParticles.vertex.%s" % c)[1] for c in "xyz"]
    off_mc = np.concatenate(([0], np.cumsum(n_mc)))
    event_of_mc = np.repeat(np.arange(len(n_mc)), n_mc)

    prim = np.nonzero(status == 1)[0]
    row_of = np.full(len(pdg), -1, np.int64)
    row_of[prim] = np.arange(len(prim))
    nrow = len(prim)
    # generator ancestor of every particle (first parent, followed upwards): showers that start inside the tracking
    # region are stored as separate particles, and their calorimeter energy belongs to the generator particle
    _, pb = _flat(tree, "MCParticles/MCParticles.parents_begin")
    _, pe = _flat(tree, "MCParticles/MCParticles.parents_end")
    n_par, par = _flat(tree, "_MCParticles_parents/_MCParticles_parents.index")
    off_par = np.repeat(np.concatenate(([0], np.cumsum(n_par)))[:-1], n_mc)
    has = pe > pb
    parent = np.full(len(pdg), -1, np.int64)
    parent[has] = par[(pb + off_par)[has]] + np.repeat(off_mc[:-1], n_mc)[has]
    anc = np.where(status == 1, np.arange(len(pdg)), parent)
    for _ in range(200):
        up = (anc >= 0) & (status[np.clip(anc, 0, None)] != 1)
        if not up.any():
            break
        anc[up] = parent[anc[up]]
    row_of_anc = np.where(anc >= 0, row_of[np.clip(anc, 0, None)], -1)
    parent_pdg = np.where(parent >= 0, pdg[np.clip(parent, 0, None)], 0)
    p = np.sqrt(mom[0][prim] ** 2 + mom[1][prim] ** 2 + mom[2][prim] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.arccos(np.clip(mom[2][prim] / np.where(p > 0, p, 1.0), -1.0, 1.0))
    out = {"event": (event_of_mc[prim] + start).astype(np.int32), "pdg": pdg[prim].astype(np.int32), "p": p.astype(np.float32),
           "theta": theta.astype(np.float32), "phi": np.arctan2(mom[1][prim], mom[0][prim]).astype(np.float32),
           "vx": vtx[0][prim].astype(np.float32), "vy": vtx[1][prim].astype(np.float32), "vz": vtx[2][prim].astype(np.float32),
           "parent": parent_pdg[prim].astype(np.int32)}
    info = {"file": str(path), "events": int(len(n_mc)), "rows": int(nrow), "collections": {}}

    for name, kind in sorted(cols.items()):
        enc = parse_encoding(enc_text[name]) if enc_text.get(name) else {}
        if kind == "tracker":
            n_hit, cid = _flat(tree, "%s/%s.cellID" % (name, name))
            if len(cid) == 0:
                info["collections"][name] = {"kind": kind, "layers": 0, "encoding": enc_text.get(name, "")}
                continue
            off = np.repeat(off_mc[:-1], n_hit)
            _, pidx = _flat(tree, "_%s_particle/_%s_particle.index" % (name, name))
            row = np.where(pidx >= 0, row_of[np.clip(pidx + off, 0, len(row_of) - 1)], -1)
            _, t = _flat(tree, "%s/%s.time" % (name, name))
            xyz = [_flat(tree, "%s/%s.position.%s" % (name, name, c))[1] for c in "xyz"]
            pm = [_flat(tree, "%s/%s.momentum.%s" % (name, name, c))[1] for c in "xyz"]
            pmag = np.sqrt(pm[0] ** 2 + pm[1] ** 2 + pm[2] ** 2)
            layer = decode(cid, enc, "layer") if "layer" in enc else np.ones(len(cid), np.int64)
            elem_field = next((k for k in ("module", "sector") if k in enc), None)
            elem = decode(cid, enc, elem_field) if elem_field else np.zeros(len(cid), np.int64)
            ring = decode(cid, enc, "ring") if "ring" in enc else None
            nlay = int(layer.max()) + 1
            ok = row >= 0
            key = row[ok] * nlay + layer[ok]
            order = np.lexsort((t[ok], key))
            ks, first = np.unique(key[order], return_index=True)
            sel = np.nonzero(ok)[0][order][first]               # first hit in time of each (row, layer)
            r_, l_ = ks // nlay, ks % nlay
            hit = np.zeros((nrow, nlay), bool)
            hit[r_, l_] = True
            out[name + "_hit"] = hit
            for c, a in zip("xyz", xyz):
                arr = np.full((nrow, nlay), np.nan, np.float32)
                arr[r_, l_] = a[sel]
                out["%s_%s" % (name, c)] = arr
            arr = np.full((nrow, nlay), np.nan, np.float32)
            arr[r_, l_] = t[sel]
            out[name + "_t"] = arr
            arr = np.full((nrow, nlay), np.nan, np.float32)
            arr[r_, l_] = pmag[sel]
            out[name + "_p"] = arr
            arr = np.zeros((nrow, nlay), np.int16)
            arr[r_, l_] = elem[sel]
            out[name + "_elem"] = arr
            if ring is not None:
                arr = np.zeros((nrow, nlay), np.int16)
                arr[r_, l_] = ring[sel]
                out[name + "_ring"] = arr
            info["collections"][name] = {"kind": kind, "layers": nlay, "element_field": elem_field,
                                         "encoding": enc_text.get(name, "")}
        else:
            n_hit, e_hit = _flat(tree, "%s/%s.energy" % (name, name))
            etot = np.bincount(np.repeat(np.arange(len(n_hit)), n_hit), weights=e_hit, minlength=len(n_hit))
            out[name + "_etot"] = etot[event_of_mc[prim]].astype(np.float32)
            out[name + "_e"] = np.zeros(nrow, np.float32)
            out[name + "_x"] = np.full(nrow, np.nan, np.float32)
            out[name + "_y"] = np.full(nrow, np.nan, np.float32)
            info["collections"][name] = {"kind": kind, "encoding": enc_text.get(name, "")}
            if len(e_hit) == 0:
                continue
            hx = _flat(tree, "%s/%s.position.x" % (name, name))[1]
            hy = _flat(tree, "%s/%s.position.y" % (name, name))[1]
            _, begin = _flat(tree, "%s/%s.contributions_begin" % (name, name))
            _, end = _flat(tree, "%s/%s.contributions_end" % (name, name))
            n_rel, rel = _flat(tree, "_%s_contributions/_%s_contributions.index" % (name, name))
            cname = name + "Contributions"
            n_con, e_con = _flat(tree, "%s/%s.energy" % (cname, cname))
            _, cp = _flat(tree, "_%s_particle/_%s_particle.index" % (cname, cname))
            counts = (end - begin).astype(np.int64)
            if counts.sum() != len(rel):
                raise RuntimeError("%s: the hits do not tile their contribution list" % name)
            hit_of_pos = np.repeat(np.arange(len(e_hit)), counts)
            off_con = np.repeat(np.concatenate(([0], np.cumsum(n_con)))[:-1], n_rel)
            hit_of_con = np.full(len(e_con), -1, np.int64)
            hit_of_con[rel + off_con] = hit_of_pos
            off_p = np.repeat(off_mc[:-1], n_con)
            row = np.where(cp >= 0, row_of_anc[np.clip(cp + off_p, 0, len(row_of) - 1)], -1)
            ok = (row >= 0) & (hit_of_con >= 0)
            if not ok.any():
                continue
            out[name + "_e"] = np.bincount(row[ok], weights=e_con[ok], minlength=nrow).astype(np.float32)
            key = row[ok] * (len(e_hit) + 1) + hit_of_con[ok]
            ks, inv = np.unique(key, return_inverse=True)
            ek = np.bincount(inv, weights=e_con[ok])
            r_, h_ = ks // (len(e_hit) + 1), ks % (len(e_hit) + 1)
            order = np.lexsort((-ek, r_))                       # per row, the cell with the largest energy first
            rr, first = np.unique(r_[order], return_index=True)
            best = h_[order][first]
            out[name + "_x"][rr] = hx[best]
            out[name + "_y"][rr] = hy[best]
    return out, info


def save_table(path, arrays, info):
    np.savez_compressed(path, __info__=np.array(json.dumps(info)), **arrays)


def load_table(path):
    """-> (arrays, info) of one file written by save_table"""
    with np.load(path, allow_pickle=False) as z:
        info = json.loads(str(z["__info__"]))
        return {k: z[k] for k in z.files if k != "__info__"}, info


def concat_tables(tables, renumber=True):
    """Join the arrays of several tables of the same configuration; with `renumber` the events are renumbered to
    stay unique (tables of different files)."""
    out, shift = {}, 0
    keys = set(tables[0])
    for t in tables[1:]:
        keys &= set(t)
    for k in sorted(keys):
        parts = []
        for t in tables:
            a = t[k]
            parts.append(a)
        if parts[0].ndim == 2:                                  # layers can differ in number: pad to the widest
            w = max(a.shape[1] for a in parts)
            parts = [np.pad(a, ((0, 0), (0, w - a.shape[1])), constant_values=(np.nan if a.dtype.kind == "f" else 0))
                     for a in parts]
        out[k] = np.concatenate(parts)
    if renumber:
        ev = []
        for t in tables:
            ev.append(t["event"].astype(np.int64) + shift)
            shift += int(t["event"].max()) + 1 if len(t["event"]) else 0
        out["event"] = np.concatenate(ev)
    return out
