# SPDX-License-Identifier: LGPL-3.0-or-later
"""solid_evview: draw simulated events on the SoLID line drawings, as figures and animations.

usage:
    solid_evview EVENTS... --scene scene.npz [-o OUTDIR] [--views side,zoom,beam,cutaway] [--formats svg,png]
                 [--event N ...] [--select pair|pair-mu] [--min-energy GEV] [--max-events K] [--overlay]
                 [--animate [VIEWS]] [--frames 120] [--anim-formats gif,svg,webp,mp4,webm]

EVENTS are event.npz files written by solid_evexport, or directories holding event_*.npz.  The scene is the
scene.npz of the same geometry (default: the scene named in the event file).  Needs numpy; PNG and animation
frames need pycairo, GIF and WebP need Pillow.

Views
    side     section through the beam axis in the event plane; tracks and hits in r-z projection (each keeps
             its radius and z and is turned about the beam axis into the plane), hit volumes that the plane
             misses drawn translucent, turned into the plane the same way
    zoom     the same, around the volume with the largest energy deposit
    beam     beam-eye section just behind the front face of that volume (or at --zsec); tracks and hits
             upstream of the section, hits as rings
    cutaway  3-D hidden-line drawing, cut at the event plane; tracks hidden behind solids; hit volumes are
             coloured where the cut goes through them
The event plane is the plane through the beam axis that the energy deposits lie closest to (their principal
axis in x-y); --phi sets its azimuth by hand.

What is drawn
    tracks   recorded trajectories, coloured by particle type; primaries wider.  Secondaries are drawn from
             --emin (charged, 5 MeV), --emin-gamma (20 MeV), --emin-neutral (50 MeV) kinetic energy at birth.
    hits     markers at the tracker-type hits (GEM, SPD, MRPC, muon chambers)
    volumes  every drawn volume with hits is coloured by its energy deposit on a log scale (one red ramp,
             --emin-cal to the largest deposit); volumes below the scale minimum stay as they are.
             Photon-counting readouts (Cherenkov) are counted, not summed: a sensor volume with photon hits
             is violet and has a violet dot at its centre (the photocathode tiles are thin).

Event selection: --event N (repeatable), --min-energy GEV (calorimeter deposit), --select pair (generator
e+ and e- both with at least --min-tracker-hits GEM hits; pair-mu for muons), --max-events K.  --overlay
draws all selected events in one figure, energies summed (the plane is then x = 0 unless --phi is given).

Animation (--animate, optionally with a comma list of views; default side): the camera is fixed, tracks grow
with the simulation time, hits appear and volumes fill with energy as it is deposited.  The last frame is the
static figure.  Always written: GIF and animated SVG.  WebP if Pillow can write it.  MP4 and WebM only if an
ffmpeg program is found ($SOLID_FFMPEG, or ffmpeg on PATH); otherwise a message says so and the run goes on.

Output files are OUTDIR/<name>.<format>, name from --name with {config}, {event} and {view}; animations get
_anim appended.
"""
import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

from solid_lineview import core as lv

from . import __version__
from .event import merge_events, select_events


def hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


# particle types: distinct hues, readable on white; neutral inks for the rest
STYLE = {
    "e-": dict(rgb=hex2rgb("#2a78d6"), dash=None, label="e⁻"),
    "e+": dict(rgb=hex2rgb("#eb6834"), dash=None, label="e⁺"),
    "gamma": dict(rgb=hex2rgb("#1baf7a"), dash=(2.0, 1.6), label="photon"),
    "hadron": dict(rgb=hex2rgb("#52514e"), dash=None, label="hadron / muon"),
    "neutral": dict(rgb=hex2rgb("#898781"), dash=(0.6, 1.4), label="n / other neutral"),
}
KINDS = ["e-", "e+", "gamma", "hadron", "neutral"]
INK = hex2rgb("#0b0b0b")
INK2 = hex2rgb("#52514e")
PHOTON_RGB = hex2rgb("#8f6fd8")          # volumes of photon-counting readouts that saw photons
# one sequential hue for energy (ColorBrewer Reds 9): blue would collide with e- and the EC module fill
REDS = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"]
REDS_RGB = np.array([hex2rgb(h) for h in REDS])


def particle_kind(pdg):
    if pdg == 11:
        return "e-"
    if pdg == -11:
        return "e+"
    if pdg == 22:
        return "gamma"
    if abs(pdg) in (13, 211, 321, 2212) or abs(pdg) > 1000000000:
        return "hadron"
    return "neutral"


def nice_ceil(x):
    p = 10.0 ** math.floor(math.log10(x))
    return next(m * p for m in (1, 2, 5, 10) if m * p >= x * 0.999)


def energy_color(E, emin, emax):
    """log scale, MeV -> rgb; starts at the 2nd ramp step so that small deposits stay visible on white"""
    u = (math.log10(max(E, emin)) - math.log10(emin)) / (math.log10(emax) - math.log10(emin))
    u = min(max(u, 0.0), 1.0)
    x = 1 + u * (len(REDS_RGB) - 2)
    i = min(int(x), len(REDS_RGB) - 2)
    f = x - i
    return tuple(REDS_RGB[i] * (1 - f) + REDS_RGB[i + 1] * f)


class Track:
    __slots__ = ("parent", "pdg", "P", "T", "ekin0", "kind")

    def __init__(self, parent, pdg, P, T, ekin0, kind=None):
        self.parent, self.pdg, self.P, self.T, self.ekin0 = parent, pdg, P, T, ekin0
        self.kind = kind or particle_kind(pdg)

    @property
    def primary(self):
        return self.parent == 0


def generator_tracks(ev):
    """bool per recorded track: a particle of the generator.  These are the Geant4 primaries (parent 0) and
    the decay products of a primary that the generator already decayed: Geant4 gets a J/psi with its e+ e-
    as one primary, tracks it over no distance and starts the leptons as its secondaries."""
    n = len(ev.trk_id)
    gen = np.asarray(ev.trk_parent) == 0
    if not gen.any() or gen.all():
        return gen
    index = {int(t): k for k, t in enumerate(ev.trk_id)}
    off = ev.trk_offset
    nolength = np.zeros(n, bool)
    for k in np.nonzero(gen)[0]:
        P = ev.pt_pos[int(off[k]):int(off[k + 1])]
        nolength[k] = len(P) < 2 or float(np.abs(P - P[0]).max()) < 1e-3          # mm
    while True:                                                                    # chains of resonances
        new = [k for k in range(n) if not gen[k] and index.get(int(ev.trk_parent[k]), -1) >= 0
               and gen[index[int(ev.trk_parent[k])]] and nolength[index[int(ev.trk_parent[k])]]]
        if not new:
            return gen
        for k in new:
            gen[k] = True
            P = ev.pt_pos[int(off[k]):int(off[k + 1])]
            nolength[k] = len(P) < 2 or float(np.abs(P - P[0]).max()) < 1e-3


def event_tracks(ev, a):
    """the recorded tracks worth drawing: all generator particles (drawn as primaries), secondaries above
    the thresholds of their type"""
    out = []
    emin = {"gamma": a.emin_gamma, "neutral": a.emin_neutral}
    npts = np.diff(ev.trk_offset)
    gen = generator_tracks(ev)
    for k in range(len(ev.trk_id)):
        if npts[k] < 2:
            continue
        kind = particle_kind(int(ev.trk_pdg[k]))
        if not gen[k] and ev.trk_ekin[k] < emin.get(kind, a.emin):
            continue
        P, T = ev.track(k)
        out.append(Track(0 if gen[k] else int(ev.trk_parent[k]), int(ev.trk_pdg[k]), P.astype(float),
                         T.astype(float), float(ev.trk_ekin[k]), kind))
    return out


# =====================================================================================================
#  projections
# =====================================================================================================

def rhoz_points(P, phi, sign=None):
    """rho-z projection into the plane through the beam axis at azimuth phi: every point keeps its radius and z
    and is turned about the axis onto the half plane phi (sign +1) or phi + 180 deg (sign -1).  sign None: per
    point, from its own azimuth."""
    P = np.asarray(P, float)
    if len(P) == 0:
        return P.reshape(0, 3)
    rho = np.hypot(P[:, 0], P[:, 1])
    if sign is None:
        sign = np.where(P[:, 0] * math.cos(phi) + P[:, 1] * math.sin(phi) >= 0, 1.0, -1.0)
    return np.stack([sign * rho * math.cos(phi), sign * rho * math.sin(phi), P[:, 2]], 1)


def rhoz_track(t, phi):
    """a track goes to one side as a whole (no jumps between the half planes): the side its points lie on,
    weighted by their radius"""
    proj = t.P[:, 0] * math.cos(phi) + t.P[:, 1] * math.sin(phi)
    sign = 1.0 if proj.sum() >= 0 else -1.0
    return Track(t.parent, t.pdg, rhoz_points(t.P, phi, sign), t.T, t.ekin0, t.kind)


def clip_track_z(t, zmax):
    """the part of a track upstream of z = zmax (cut at its first crossing, time interpolated)"""
    above = np.nonzero(t.P[:, 2] > zmax)[0]
    if len(above) == 0:
        return t
    k = int(above[0])
    if k == 0:
        return None
    f = (zmax - t.P[k - 1, 2]) / (t.P[k, 2] - t.P[k - 1, 2])
    P = np.vstack([t.P[:k], t.P[k - 1] + f * (t.P[k] - t.P[k - 1])])
    T = np.append(t.T[:k], t.T[k - 1] + f * (t.T[k] - t.T[k - 1]))
    return Track(t.parent, t.pdg, P, T, t.ekin0, t.kind)


def clip_track_time(P, T, tmax):
    """points with time <= tmax plus the interpolated head point"""
    if tmax is None or T[-1] <= tmax:
        return P
    k = int(np.searchsorted(T, tmax, side="right"))
    if k == 0:
        return P[:0]
    if k < len(T) and T[k] > T[k - 1]:
        f = (tmax - T[k - 1]) / (T[k] - T[k - 1])
        return np.vstack([P[:k], P[k - 1] + f * (P[k] - P[k - 1])])
    return P[:k]


# =====================================================================================================
#  what the event did to the drawn volumes
# =====================================================================================================

class Deposits:
    """Per drawn solid of the scene: energy deposits with their times, and photon-hit times."""

    def __init__(self, sc, ev):
        solid_of = np.full(int(sc.data["node_mesh"].shape[0]), -1, np.int64)
        solid_of[sc.node] = np.arange(len(sc.node))
        photon_coll = ev.collection_mask(photon=True)
        calo_coll = ev.collection_mask(kind="calorimeter")
        hc = ev.hit_collection.astype(np.int64)
        nc = len(ev.collections)
        is_photon = photon_coll[hc] if nc else np.zeros(0, bool)
        is_calo = calo_coll[hc] if nc else np.zeros(0, bool)
        ok = ev.hit_node >= 0
        hsolid = np.where(ok, solid_of[np.maximum(ev.hit_node, 0)], -1)
        drawn_coll = np.array([c.get("drawn", True) for c in ev.collections], bool)
        self.unmapped = int(((hsolid < 0) & drawn_coll[hc]).sum()) if nc else 0
        # energy with time: calorimeter contributions, and the other hits at their hit time
        direct = ok & ~is_photon & ~is_calo
        sid = np.concatenate([hsolid[ev.dep_hit], hsolid[direct]])
        t = np.concatenate([ev.dep_time, ev.hit_time[direct]]).astype(float)
        e = np.concatenate([ev.dep_energy, ev.hit_energy[direct]]).astype(float)
        self.energy = self._group(sid, t, e, cumulative=True)
        ph = ok & is_photon
        self.photons = self._group(hsolid[ph], ev.hit_time[ph].astype(float), None, cumulative=False)
        # markers: tracker-type hits in drawn volumes that are not single photons
        m = ~is_photon & ~is_calo & ok
        self.hit_pos, self.hit_time = ev.hit_pos[m].astype(float), ev.hit_time[m].astype(float)
        # energy-weighted positions for the choice of the event plane (all hits with energy)
        w = np.where(is_photon, 0.0, ev.hit_energy.astype(float))
        self.w_pos, self.w = ev.hit_pos[w > 0].astype(float), w[w > 0]

    @staticmethod
    def _group(sid, t, e, cumulative):
        out = {}
        keep = sid >= 0
        sid, t = sid[keep], t[keep]
        e = e[keep] if e is not None else None
        order = np.lexsort((t, sid))
        sid, t = sid[order], t[order]
        e = e[order] if e is not None else None
        bounds = np.nonzero(np.diff(sid))[0] + 1
        for g in np.split(np.arange(len(sid)), bounds):
            if len(g):
                out[int(sid[g[0]])] = (t[g], np.cumsum(e[g])) if cumulative else t[g]
        return out

    def total(self, s):
        return float(self.energy[s][1][-1]) if s in self.energy else 0.0

    def energy_at(self, s, tmax):
        if s not in self.energy:
            return 0.0
        ts, ce = self.energy[s]
        if tmax is None:
            return float(ce[-1])
        k = int(np.searchsorted(ts, tmax, side="right"))
        return float(ce[k - 1]) if k > 0 else 0.0

    def photons_at(self, s, tmax):
        ts = self.photons.get(s)
        if ts is None:
            return 0
        return len(ts) if tmax is None else int(np.searchsorted(ts, tmax, side="right"))

    def dominant(self):
        """solid with the largest energy deposit, or None"""
        return max(self.energy, key=self.total) if self.energy else None

    def end_time(self, fraction=0.99):
        """time by which `fraction` of the energy is deposited"""
        if not self.energy:
            return 0.0
        t = np.concatenate([v[0] for v in self.energy.values()])
        e = np.concatenate([np.diff(v[1], prepend=0.0) for v in self.energy.values()])
        o = np.argsort(t)
        c = np.cumsum(e[o])
        return float(t[o][min(int(np.searchsorted(c, fraction * c[-1])), len(t) - 1)])


def event_plane_phi(dep, tracks):
    """azimuth of the plane through the beam axis closest to the energy deposits: the principal axis of
    their x-y positions, oriented so that the larger share of the energy is on the upper side.  Without
    deposits: the azimuth of the first primary track; else 90 deg (the y-z plane)."""
    if len(dep.w):
        x, y, w = dep.w_pos[:, 0], dep.w_pos[:, 1], dep.w
        sxx, syy, sxy = float((w * x * x).sum()), float((w * y * y).sum()), float((w * x * y).sum())
        phi = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        up = float((w * (x * math.cos(phi) + y * math.sin(phi))).sum())
        return phi if up >= 0 else phi + math.pi
    for t in tracks:
        if t.primary and len(t.P) >= 2:
            d = t.P[-1] - t.P[0]
            if math.hypot(d[0], d[1]) > 1e-6:
                return math.atan2(d[1], d[0])
    return 0.5 * math.pi


# =====================================================================================================
#  geometry layer (solid_lineview)
# =====================================================================================================

def lv_options(argv):
    o = lv.build_parser().parse_args(["scene"] + argv)
    o.width, o.height = [int(v) for v in o.size.lower().split("x")]
    o.lw_outline = o.lw if o.lw_outline is None else o.lw_outline
    o.lw_cut = 1.4 * o.lw if o.lw_cut is None else o.lw_cut
    return o


def scene_centre(sc):
    if sc.bbox is not None:
        return 0.5 * (sc.bbox[:3] + sc.bbox[3:])
    return 0.5 * (sc.V.min(0) + sc.V.max(0))


def make_view(kind, sc, dep, tracks, a):
    """camera, section or cut plane and caption of one view"""
    phi = math.radians(a.phi) if a.phi is not None else event_plane_phi(dep, tracks)
    n = np.array([-math.sin(phi), math.cos(phi), 0.0])
    ur = np.array([math.cos(phi), math.sin(phi), 0.0])
    dom = dep.dominant()
    deg = (math.degrees(phi) % 360, (math.degrees(phi) + 180) % 360)
    if kind in ("side", "zoom"):
        cam = lv.Camera(n, ur, (0, 0, 0))
        zoom = None
        if kind == "zoom":
            if dom is None:
                return None
            s = sc.solids[dom]
            c = 0.5 * (s.bmin + s.bmax)
            rc = math.copysign(math.hypot(c[0], c[1]), c[0] * ur[0] + c[1] * ur[1])
            hw = a.zoom_half
            zoom = (c[2] - 1.3 * hw, c[2] + 1.3 * hw, rc - hw, rc + hw)
        return dict(kind=kind, cam=cam, plane=(n, 0.0), phi=phi, zoom=zoom,
                    label="section through the beam axis at azimuth %.0f° / %.0f°; tracks, hits and hit volumes "
                          "off the plane (translucent) turned into it (r-z projection)" % deg)
    if kind == "cutaway":
        eye = lv._unit(0.75 * n + 0.48 * ur - 0.45 * np.array([0.0, 0.0, 1.0]))
        cam = lv.Camera(eye, ur, scene_centre(sc), a.perspective)
        keep = (-n, 0.0)                     # keep -n.P >= 0: the half behind the event plane
        return dict(kind=kind, cam=cam, plane=keep, cut=keep, phi=phi,
                    label="cut away at the plane through the beam axis at azimuth %.0f° / %.0f°, hidden lines removed"
                          % deg)
    if kind == "beam":
        if a.zsec is not None:
            zsec = a.zsec
        elif dom is not None:
            s = sc.solids[dom]
            zsec = float(s.bmin[2] + min(20.0, 0.5 * (s.bmax[2] - s.bmin[2])))
        else:
            return None
        cam = lv.Camera((0, 0, 1), (0, 1, 0), (0, 0, zsec))
        return dict(kind=kind, cam=cam, plane=(np.array([0.0, 0.0, 1.0]), -zsec), zsec=zsec,
                    label="beam view from downstream, section at z = %.1f cm; tracks and hits upstream of it"
                          % (zsec / 10))
    raise SystemExit("unknown view %s (side, zoom, beam, cutaway)" % kind)


def geometry_layer(sc, view, a, title):
    """the detector drawing of a view: (DrawList, Renderer, options)"""
    common = ["--size", a.size, "--dpi", str(a.dpi), "--lw", str(a.lw), "--font-size", str(a.font_size)]
    if title:
        common += ["--title", title]
    common += ["--subtitle", view["label"]]          # the caption, in the band reserved above the drawing
    labels = a.labels and view["kind"] in ("side", "cutaway")
    if labels:
        common += ["--labels", "solid"]
    if view["kind"] == "cutaway":
        o = lv_options(common + ["--mode", "hlr", "--cap-fill", "color", "--perspective", "%g" % a.perspective,
                                 "--hlr-scale", "2"])
        rd = lv.Renderer(sc, o)
        rd._fitinfo = None
        orig_fit = rd._fit

        def fit_and_store(cam_, pts):
            r = orig_fit(cam_, pts)
            rd._fitinfo = r
            return r

        rd._fit = fit_and_store
        dl = rd.render(view["cam"], [view["cut"]], [view["cut"]], "hlr")
    else:
        argv = common + ["--mode", "section", "--fill", "--scale-bar"]
        if view.get("zoom"):
            argv += ["--fit-box"] + ["%g" % v for v in view["zoom"]]
        o = lv_options(argv)
        dl, rd = lv.render_section(sc, o, view["cam"], view["plane"], [])
    if labels:
        found = lv.label_list(o)
        lv.place_labels(dl, rd, o, found)
        lv.finish_canvas(dl, o)
    lv.draw_title(dl, o)
    return dl, rd, o


def section_rings(sc, rd, plane, sids, to_px):
    """closed section outlines (canvas px) of the given solids"""
    rings = defaultdict(list)
    if not sids:
        return rings
    mesh = rd.mesh
    fsel = np.zeros(len(sc.solids), bool)
    fsel[list(sids)] = True
    tsel = np.nonzero(fsel[sc.fsolid][mesh.tri_face])[0]
    if len(tsel) == 0:
        return rings
    V = lv.snap_vertices(sc, sc.V, mesh.tri, [plane], rd.o.plane_snap, True)
    Q0, Q1, ti, k0, k1 = lv.section_segments(V, mesh.tri[tsel], plane)
    sol = sc.fsolid[mesh.tri_face[tsel[ti]]]
    for s in np.unique(sol):
        m = sol == s
        for pts, closed in lv.chain_segments(k0[m], k1[m], Q0[m], Q1[m]):
            if closed and len(pts) >= 3:
                x, y, _ = to_px(pts)
                rings[int(s)].append(np.stack([x, y], 1))
    return rings


def projected_hull(sc, sid, to_px, phi_plane):
    """outline of a solid turned about the beam axis into the section plane (it keeps its radius; the half
    plane is chosen by its azimuth): convex hull of its projected vertices"""
    s = sc.solids[sid]
    V = sc.V[s.v0:s.v1]
    c = V[:, :2].mean(0)
    phm = math.atan2(c[1], c[0])
    tgt = phi_plane if math.cos(phm - phi_plane) >= 0 else phi_plane + math.pi
    d = tgt - phm
    cd, sd = math.cos(d), math.sin(d)
    V = np.stack([cd * V[:, 0] - sd * V[:, 1], sd * V[:, 0] + cd * V[:, 1], V[:, 2]], 1)
    x, y, _ = to_px(V)
    P = np.unique(np.round(np.stack([x, y], 1), 3), axis=0)
    if len(P) < 3:
        return None
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h

    lo, hi = half(P), half(P[::-1])
    hull = np.array(lo[:-1] + hi[:-1])
    return hull if len(hull) >= 3 else None


# =====================================================================================================
#  one view of one event: static layers and everything needed to draw the event at any time
# =====================================================================================================

class EventView:
    def __init__(self, sc, ev, kind, a, title=None, dep=None, tracks=None):
        t0 = time.perf_counter()
        self.sc, self.ev, self.a, self.kind = sc, ev, a, kind
        self.dep = dep if dep is not None else Deposits(sc, ev)
        tracks = tracks if tracks is not None else event_tracks(ev, a)
        self.view = view = make_view(kind, sc, self.dep, tracks, a)
        if view is None:
            raise ValueError("view %s needs an energy deposit in a drawn volume (or --zsec for beam)" % kind)
        dl, rd, o = geometry_layer(sc, view, a, title)
        self.rd, self.o = rd, o
        ox, oy = float(dl.origin[0]), float(dl.origin[1])     # shift by labels and margins

        def to_px(P):
            x, y, z = rd.to_px(P)
            return x + ox, y + oy, z

        self.to_px, self.origin = to_px, (ox, oy)
        pt = dl.dpi / 72.0
        self.dpi = dl.dpi
        self.band = int(165 * pt)                      # colour bar and legend to the right of the drawing
        self.W0 = dl.W
        self.W, self.H = dl.W + self.band, max(dl.H, int(330 * pt))
        first_line = next((i for i, it in enumerate(dl.items) if it[0] != "fill"), len(dl.items))
        self.fill_items = dl.items[:first_line]
        self.line_items = dl.items[first_line:]
        # volumes: energy scale and outlines
        lit = sorted(set(self.dep.energy) | set(self.dep.photons))
        totals = [self.dep.total(s) for s in self.dep.energy]
        self.emin = a.emin_cal
        self.emax = nice_ceil(max(max(totals), 10.0 * self.emin)) if totals else 1000.0 * self.emin
        self.rings = section_rings(sc, rd, view["plane"], lit, to_px)
        self.ghosts = {}
        if kind in ("side", "zoom") and a.project_volumes:
            for s in lit:
                if s not in self.rings and (self.dep.total(s) >= self.emin or s in self.dep.photons):
                    h = projected_hull(sc, s, to_px, view["phi"])
                    if h is not None:
                        self.ghosts[s] = h
        # tracks and hit markers in the projection of the view
        self.hitP, self.hitT = self.dep.hit_pos, self.dep.hit_time
        if kind in ("side", "zoom"):
            tracks = [rhoz_track(t, view["phi"]) for t in tracks]
            self.hitP = rhoz_points(self.hitP, view["phi"])
        # photon-counting volumes (thin photocathode tiles) also get a marker at their centre
        ps = sorted(self.dep.photons)
        self.photon_solids = ps
        pc = np.array([0.5 * (sc.solids[s].bmin + sc.solids[s].bmax) for s in ps], float).reshape(-1, 3)
        if kind in ("side", "zoom"):
            pc = rhoz_points(pc, view["phi"])
        if kind == "beam" and len(pc):
            keep = pc[:, 2] <= view["zsec"]
            pc, self.photon_solids = pc[keep], [s for s, k in zip(ps, keep) if k]
        self.photon_px = np.stack(to_px(pc)[:2], 1) if len(pc) else np.zeros((0, 2))
        self.ring_hits = kind == "beam"
        if kind == "beam":
            tracks = [u for u in (clip_track_z(t, view["zsec"]) for t in tracks) if u is not None and len(u.P) >= 2]
            m = self.hitP[:, 2] <= view["zsec"]
            self.hitP, self.hitT = self.hitP[m], self.hitT[m]
        self.tracks = sorted(tracks, key=lambda t: (t.primary, t.kind != "gamma", t.ekin0))
        self.kinds = [k for k in KINDS if any(t.kind == k for t in self.tracks)]
        self._pieces = None
        self.title = title
        self.title_band = int(math.ceil(lv.title_band_px(o)))
        self.t_build = time.perf_counter() - t0
        lv.log("view %s: canvas %dx%d, %d volumes with hits (%d cut, %d projected), %d tracks, %d hit markers, "
               "%.2f s" % (kind, self.W, self.H, len(lit), len(self.rings), len(self.ghosts), len(self.tracks),
                           len(self.hitT), self.t_build))

    # ---- time
    def end_time(self):
        """animation end: 99 % of the energy deposited and the primaries finished (at most 50 ns), plus 2 ns"""
        tp = [float(t.T[-1]) for t in self.tracks if t.primary]
        return max(self.dep.end_time(), min(max(tp), 50.0) if tp else 0.0, 1.0) + 2.0

    # ---- volume colours
    def volume_color(self, s, tmax):
        """(rgb, True) by energy, (violet, False) for photon hits, or None if nothing to show at tmax"""
        E = self.dep.energy_at(s, tmax)
        if E >= self.emin:
            return energy_color(E, self.emin, self.emax)
        if self.dep.photons_at(s, tmax) > 0:
            return PHOTON_RGB
        return None

    def volume_items(self, tmax=None):
        dl = lv.DrawList(self.W, self.H, self.dpi, None)
        for s in sorted(self.ghosts, key=self.dep.total):
            c = self.volume_color(s, tmax)
            if c is not None:
                dl.fill([self.ghosts[s]], c, stroke=INK2, lw_pt=0.25, alpha=0.55)
        for s, rings in self.rings.items():
            c = self.volume_color(s, tmax)
            if c is not None:
                dl.fill(rings, c, stroke=INK, lw_pt=0.3)
        return dl.items

    # ---- tracks
    def track_pieces(self):
        """The drawn pieces of the complete tracks, in drawing order (photons first, primaries last):
        [(kind, primary, polyline px (n, 2), times (n,))].  In the 3-D view a track is split into the parts
        that the cut geometry does not hide (from the renderer's depth buffer)."""
        if self._pieces is not None:
            return self._pieces
        out = []
        if self.kind != "cutaway":
            for t in self.tracks:
                x, y, _ = self.to_px(t.P)
                out.append((t.kind, t.primary, np.stack([x, y], 1), t.T))
            self._pieces = out
            return out
        rd = self.rd
        hs = rd.o.hlr_scale
        ox, oy = self.origin
        # the cut faces must not hide the event: they are pushed to infinite depth for this test
        cap = rd.zface == -2
        saved = rd.zA.copy(), rd.zB.copy(), rd.zC.copy()
        rd.zA[cap], rd.zB[cap], rd.zC[cap] = 0.0, 0.0, 1e30
        try:
            for t in self.tracks:
                xa, ya, za = rd.to_px(t.P[:-1])
                xb, yb, zb = rd.to_px(t.P[1:])
                m1 = np.full(len(xa), -9, np.int64)
                e, f0, f1 = rd._visible_runs(xa * hs, ya * hs, za, xb * hs, yb * hs, zb, m1, m1)
                if len(e) == 0:
                    continue
                o = np.lexsort((f0, e))
                e, f0, f1 = e[o], f0[o], f1[o]
                X0, Y0 = xa[e] + f0 * (xb[e] - xa[e]) + ox, ya[e] + f0 * (yb[e] - ya[e]) + oy
                X1, Y1 = xa[e] + f1 * (xb[e] - xa[e]) + ox, ya[e] + f1 * (yb[e] - ya[e]) + oy
                T0, T1 = t.T[e] + f0 * (t.T[e + 1] - t.T[e]), t.T[e] + f1 * (t.T[e + 1] - t.T[e])
                # a run continues the previous one if it starts where that one ended
                brk = np.nonzero((np.abs(X0[1:] - X1[:-1]) > 1e-3) | (np.abs(Y0[1:] - Y1[:-1]) > 1e-3))[0] + 1
                for g in np.split(np.arange(len(e)), brk):
                    xy = np.stack([np.append(X0[g], X1[g[-1]]), np.append(Y0[g], Y1[g[-1]])], 1)
                    out.append((t.kind, t.primary, xy, np.append(T0[g], T1[g[-1]])))
        finally:
            rd.zA, rd.zB, rd.zC = saved
        self._pieces = out
        return out

    def track_polylines(self, tmax=None):
        """[(kind, primary, polyline px)] of the tracks up to time tmax (None: complete)"""
        out = []
        frame = (0.0, float(self.title_band), float(self.W), float(self.H))    # below the title, inside the canvas
        for kind, prim, xy, T in self.track_pieces():
            P = clip_track_time(xy, T, tmax)
            if len(P) >= 2:
                out += [(kind, prim, q) for q in lv.clip_polyline_rect(P, frame)]
        return out

    def _add_tracks(self, dl, tmax):
        groups = defaultdict(list)
        for kind, prim, pl in self.track_polylines(tmax):
            groups[(kind, prim)].append(pl)
        for (kind, prim) in sorted(groups, key=lambda k: (k[1], k[0] != "gamma")):
            st = STYLE[kind]
            dl.lines(groups[(kind, prim)], st["rgb"], self.a.traj_lw * (2.6 if prim else 1.0),
                     dash=st["dash"] if not prim else None)

    def _add_hits(self, dl, tmax):
        pt = self.dpi / 72.0
        m = np.ones(len(self.hitT), bool) if tmax is None else self.hitT <= tmax
        if not m.any():
            return
        x, y, _ = self.to_px(self.hitP[m])
        r = self.a.hit_size * pt
        if self.ring_hits:               # on coloured modules: hollow rings
            ang = np.linspace(0, 2 * math.pi, 17)
            dl.lines([np.stack([xi + r * np.cos(ang), yi + r * np.sin(ang)], 1) for xi, yi in zip(x, y)], INK, 0.7)
            return
        for xi, yi in zip(x, y):
            dl.marker("circle", float(xi), float(yi), r + 0.6 * pt, (1, 1, 1))
            dl.marker("circle", float(xi), float(yi), r, INK)

    # ---- legend
    def legend_items(self):
        """white band on the right with the colour bar and the legend"""
        a = self.a
        dl = lv.DrawList(self.W, self.H, self.dpi, None)
        pt = self.dpi / 72.0
        fs = a.font_size
        x1 = self.W0
        dl.fill([np.array([[x1, 0], [self.W, 0], [self.W, self.H], [x1, self.H]], float)], (1, 1, 1))
        x0, y0 = x1 + 14 * pt, 40 * pt
        bw, bh = 12 * pt, 150 * pt
        emin, emax = self.emin, self.emax
        dl.text(x0, y0 - 6 * pt, "energy deposit per volume", fs, INK)
        n = 64
        for i in range(n):
            u0, u1 = i / n, (i + 1) / n
            E = 10 ** (math.log10(emin) + (u0 + u1) / 2 * (math.log10(emax) - math.log10(emin)))
            ya, yb = y0 + bh * (1 - u1), y0 + bh * (1 - u0)
            dl.fill([np.array([[x0, ya], [x0 + bw, ya], [x0 + bw, yb + 0.5], [x0, yb + 0.5]])],
                    energy_color(E, emin, emax))
        dl.lines([np.array([[x0, y0], [x0 + bw, y0], [x0 + bw, y0 + bh], [x0, y0 + bh], [x0, y0]])], INK2, 0.5)
        decades = math.log10(emax / emin)
        for p in range(int(math.floor(math.log10(emin))), int(math.ceil(math.log10(emax))) + 1):
            for mlt in (1, 2, 5):
                E = mlt * 10.0 ** p
                if E < emin * 0.999 or E > emax * 1.001:
                    continue
                u = (math.log10(E) - math.log10(emin)) / (math.log10(emax) - math.log10(emin))
                yy = y0 + bh * (1 - u)
                dl.lines([np.array([[x0 + bw, yy], [x0 + bw + 3 * pt, yy]])], INK2, 0.5)
                top = abs(E - emax) < 1e-6 * emax
                if (mlt == 1 and (top or u < 0.93)) or (mlt == 5 and decades < 2.5 and u < 0.93) or top:
                    lab = "%g GeV" % (E / 1000.0) if E >= 1000.0 else "%g MeV" % E
                    dl.text(x0 + bw + 5 * pt, yy + 0.35 * fs * pt, lab, fs * 0.9, INK2)
        yl = y0 + bh + 28 * pt
        dl.text(x0, yl - 6 * pt, "particles", fs, INK)
        row = 13 * pt
        yy = yl + 0.5 * row
        for k in self.kinds or KINDS[:2]:
            st = STYLE[k]
            dl.lines([np.array([[x0, yy], [x0 + 22 * pt, yy]])], st["rgb"], 1.2, dash=st["dash"])
            dl.text(x0 + 28 * pt, yy + 0.35 * fs * pt, st["label"], fs * 0.9, INK)
            yy += row
        dl.lines([np.array([[x0, yy], [x0 + 22 * pt, yy]])], STYLE[(self.kinds or KINDS)[0]]["rgb"],
                 a.traj_lw * 2.6)
        dl.text(x0 + 28 * pt, yy + 0.35 * fs * pt, "primary (wide line)", fs * 0.9, INK)
        yy += row
        if len(self.hitT):
            if self.ring_hits:
                ang = np.linspace(0, 2 * math.pi, 17)
                r = a.hit_size * pt
                dl.lines([np.stack([x0 + 11 * pt + r * np.cos(ang), yy + r * np.sin(ang)], 1)], INK, 0.7)
            else:
                dl.marker("circle", x0 + 11 * pt, yy, a.hit_size * pt, INK)
            dl.text(x0 + 28 * pt, yy + 0.35 * fs * pt, "tracker hit", fs * 0.9, INK)
            yy += row
        if self.dep.photons:
            dl.marker("circle", x0 + 11 * pt, yy, 3.2 * pt, INK)
            dl.marker("circle", x0 + 11 * pt, yy, 2.5 * pt, PHOTON_RGB)
            dl.text(x0 + 28 * pt, yy + 0.35 * fs * pt, "sensor with Cherenkov photons", fs * 0.9, INK)
        return dl.items

    def mask_items(self):
        """zoomed views: geometry outside the window reaches into the title band; cover it and redraw the title"""
        if not self.view.get("zoom"):
            return []
        dl = lv.DrawList(self.W0, self.H, self.dpi, None)
        pt = self.dpi / 72.0
        band = self.title_band + int(0.3 * self.a.font_size * pt)
        dl.fill([np.array([[0, 0], [self.W0, 0], [self.W0, band], [0, band]], float)], (1, 1, 1))
        lv.draw_title(dl, self._title_options())
        return dl.items

    def _title_options(self):
        class _O:
            pass
        o = _O()
        for k in ("dpi", "title", "subtitle", "title_size", "title_bold", "font_size"):
            setattr(o, k, getattr(self.o, k))
        return o

    def event_items(self, tmax=None):
        """tracks and hit markers up to time tmax (None: all)"""
        dl = lv.DrawList(self.W0, self.H, self.dpi, None)
        self._add_tracks(dl, tmax)
        self._add_hits(dl, tmax)
        pt = self.dpi / 72.0
        for s, (x, y) in zip(self.photon_solids, self.photon_px):
            if self.dep.photons_at(s, tmax) > 0:
                dl.marker("circle", float(x), float(y), 3.2 * pt, INK)
                dl.marker("circle", float(x), float(y), 2.5 * pt, PHOTON_RGB)
        return dl.items

    def clock_items(self, tmax):
        dl = lv.DrawList(self.W, self.H, self.dpi, None)
        pt, fs = self.dpi / 72.0, self.a.font_size
        dl.text(self.W0 + 14 * pt, self.H - 14 * pt, "t = %5.1f ns" % tmax, fs * 1.4, INK)
        return dl.items

    def top_items(self):
        """everything static above the event: masks and legend"""
        if not hasattr(self, "_top"):
            self._top = self.mask_items() + self.legend_items()
        return self._top

    def drawlist(self, tmax=None):
        dl = lv.DrawList(self.W, self.H, self.dpi, (1, 1, 1))
        dl.items = (self.fill_items + self.volume_items(tmax) + self.line_items + self.event_items(tmax)
                    + self.top_items() + (self.clock_items(tmax) if tmax is not None else []))
        return dl


def write_figure(dl, paths):
    """Write a DrawList to files.  PNG goes through cairo and Pillow when both are there (several times
    faster than the dependency-free PNG writer of solid_lineview, same pixels); the rest through
    solid_lineview."""
    for path in paths:
        t0 = time.perf_counter()
        done = False
        if path.lower().endswith(".png"):
            try:
                import cairo
                from PIL import Image
                surf = cairo.ImageSurface(cairo.FORMAT_RGB24, dl.W, dl.H)
                cr = cairo.Context(surf)
                cr.set_source_rgb(*(dl.bg or (1, 1, 1)))
                cr.paint()
                lv.cairo_draw_items(cr, dl.items, dl.dpi)
                surf.flush()
                bgra = np.ndarray((dl.H, dl.W, 4), np.uint8, surf.get_data(), 0, (surf.get_stride(), 4, 1))
                Image.fromarray(np.ascontiguousarray(bgra[:, :, 2::-1])).save(path, dpi=(dl.dpi, dl.dpi), compress_level=3)
                done = True
            except ImportError:
                pass
        if not done:
            lv.write_outputs(dl, [path], "auto")
        else:
            lv.log("wrote %s (%.1f kB, %.2f s)" % (path, os.path.getsize(path) / 1024.0, time.perf_counter() - t0))


# =====================================================================================================
#  command line
# =====================================================================================================

def build_parser():
    ap = argparse.ArgumentParser(prog="solid_evview", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="+", help="event.npz files or directories with event_*.npz (solid_evexport)")
    ap.add_argument("--scene", default=None, help="scene.npz of the geometry (default: the one named in the "
                    "event file)")
    ap.add_argument("-o", "--outdir", default=".")
    ap.add_argument("--views", default="side,beam", help="comma list of side, zoom, beam, cutaway (default "
                    "side,beam)")
    ap.add_argument("--formats", default="svg,png", help="comma list of svg, pdf, eps, png (default svg,png)")
    ap.add_argument("--name", default="{config}_event{event}_{view}",
                    help="output name without extension, with {config}, {event}, {view}")
    ap.add_argument("--title", default=None, help="figure title (default '<config>, event N'); '' for none")
    g = ap.add_argument_group("event selection")
    g.add_argument("-e", "--event", type=int, action="append", default=None, metavar="N",
                   help="event number (repeatable); default: all events given")
    g.add_argument("--select", choices=("pair", "pair-mu"), default=None,
                   help="e+e- (pair) or mu+mu- (pair-mu) pair in acceptance: both generator leptons with at "
                   "least --min-tracker-hits GEM hits")
    g.add_argument("--min-tracker-hits", type=int, default=3)
    g.add_argument("--min-energy", type=float, default=None, metavar="GEV", help="minimum calorimeter deposit")
    g.add_argument("--max-events", type=int, default=None, help="draw at most this many events")
    g.add_argument("--overlay", action="store_true", help="all selected events in one figure")
    g = ap.add_argument_group("views")
    g.add_argument("--phi", type=float, default=None, help="azimuth of the side/cutaway plane in degrees "
                   "(default: the event plane; 90 with --overlay)")
    g.add_argument("--zsec", type=lv.parse_length, default=None, help="z of the beam-view section (mm, or 440cm)")
    g.add_argument("--zoom-half", type=lv.parse_length, default=600.0, help="zoom view: half height of the "
                   "window (default 600 mm)")
    g.add_argument("--perspective", type=float, default=40000.0, help="cutaway camera distance in mm")
    g.add_argument("--labels", action="store_true", help="subsystem labels in the side and cutaway views")
    g.add_argument("--no-project-volumes", dest="project_volumes", action="store_false",
                   help="side views: do not draw hit volumes that the section plane misses")
    g = ap.add_argument_group("style")
    g.add_argument("--size", default="2400x1500", help="largest drawing size in px (default 2400x1500)")
    g.add_argument("--dpi", type=float, default=200.0)
    g.add_argument("--lw", type=float, default=0.35, help="geometry line width in pt")
    g.add_argument("--font-size", type=float, default=9.0)
    g.add_argument("--traj-lw", type=float, default=0.45, help="track line width in pt (primaries 2.6 times)")
    g.add_argument("--hit-size", type=float, default=1.6, help="hit marker radius in pt")
    g.add_argument("--emin", type=float, default=5.0, help="charged secondaries: minimum kinetic energy at "
                   "birth in MeV (default 5)")
    g.add_argument("--emin-gamma", type=float, default=20.0)
    g.add_argument("--emin-neutral", type=float, default=50.0)
    g.add_argument("--emin-cal", type=float, default=0.1, help="minimum of the energy colour scale in MeV "
                   "(default 0.1)")
    g = ap.add_argument_group("animation")
    g.add_argument("--animate", nargs="?", const="side", default=None, metavar="VIEWS",
                   help="animate these views (comma list, default side)")
    g.add_argument("--frames", type=int, default=120)
    g.add_argument("--fps", type=float, default=12.0)
    g.add_argument("--hold", type=float, default=2.0, help="seconds the last frame stays (default 2)")
    g.add_argument("--frame-scale", type=float, default=0.6, help="frame size relative to the static PNG "
                   "(default 0.6)")
    g.add_argument("--t-end", type=float, default=None, help="end of the animated time span in ns (default: "
                   "99 %% of the energy deposited and primaries finished, plus 2 ns)")
    g.add_argument("--anim-formats", default="gif,svg,webp,mp4",
                   help="comma list of gif, svg, webp, mp4, webm (default gif,svg,webp,mp4; webp and the video "
                   "formats are skipped with a message when no encoder is available)")
    g.add_argument("--gif-max-kb", type=float, default=1000.0, help="shrink the GIF until it is below this "
                   "size (default 1000 kB; 0 = keep the frame size)")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--version", action="version", version="solid_evview " + __version__)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    lv.VERBOSE = 0 if a.quiet else 1
    t_start = time.perf_counter()
    events = select_events(a.events, numbers=set(a.event) if a.event else None, select=a.select,
                           min_energy=a.min_energy, min_tracker_hits=a.min_tracker_hits, max_events=a.max_events)
    if not events:
        print("solid_evview: no event selected", file=sys.stderr)
        return 1
    scene = a.scene or events[0].meta.get("scene_file")
    if not scene or not os.path.isfile(scene):
        print("solid_evview: scene file %s not found (use --scene)" % scene, file=sys.stderr)
        return 1
    sc = lv.load_scene(scene)
    if not hasattr(sc, "data"):
        print("solid_evview: %s is not a scene.npz (a .prim file has no volume IDs)" % scene, file=sys.stderr)
        return 1
    shash = sc.data.meta.get("geometry_hash", "")
    for ev in events:
        if ev.meta.get("geometry_hash") and shash and ev.meta["geometry_hash"] != shash:
            print("solid_evview: event %s was exported with another geometry than %s" % (ev.number, scene),
                  file=sys.stderr)
            return 1
    config = events[0].meta.get("config") or sc.data.meta.get("config", "event")
    if a.overlay and len(events) > 1:
        if a.phi is None:
            a.phi = 90.0
        merged = merge_events(events)
        groups = [(merged, "s%d-%d" % (min(merged.numbers), max(merged.numbers)),
                   "%s, %d events" % (config, len(events)))]
    else:
        groups = [(ev, "%d" % ev.number, "%s, event %d" % (config, ev.number)) for ev in events]
    os.makedirs(a.outdir, exist_ok=True)
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    anim = [v.strip() for v in a.animate.split(",") if v.strip()] if a.animate else []
    formats = [f.strip() for f in a.formats.split(",") if f.strip()]
    status = 0
    for ev, tag, title in groups:
        t0 = time.perf_counter()
        title = title if a.title is None else a.title
        dep = Deposits(sc, ev)
        tracks = event_tracks(ev, a)
        if dep.unmapped:
            lv.log("warning: %d hits of event %s have no drawn volume" % (dep.unmapped, tag))
        for vk in dict.fromkeys(views + anim):
            try:
                view = EventView(sc, ev, vk, a, title or None, dep, tracks)
            except ValueError as exc:
                print("solid_evview: event %s: %s; view skipped" % (tag, exc), file=sys.stderr)
                continue
            base = os.path.join(a.outdir, a.name.format(config=config, event=tag, view=vk))
            if vk in views:
                write_figure(view.drawlist(None), [base + "." + f for f in formats])
            if vk in anim:
                from . import animate
                status |= animate.animate_view(view, base + "_anim", a)
        lv.log("event %s: %d views in %.2f s" % (tag, len(dict.fromkeys(views + anim)), time.perf_counter() - t0))
    lv.log("total %.2f s" % (time.perf_counter() - t_start))
    return status


if __name__ == "__main__":
    sys.exit(main())
