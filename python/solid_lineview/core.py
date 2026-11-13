#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solid_lineview -- clean vector line drawings of Geant4 geometry scenes.

A dependency-light replacement for the DAWN (Fukui Renderer) + DAWNCUT workflow.  The input is either
a scene file with the Geant4 polyhedra (scene.npz, from solid_scene_dump or ddsim with SOLID_SCENE)
or a DAWNFILE .prim:

    solid_scene_dump           ->  scene.npz     \
                                                  >  solid_lineview  ->  SVG / PDF / EPS / PNG
    Geant4 /vis/open DAWNFILE  ->  g4_XXXX.prim  /

Rendering modes
  hlr        3-D hidden-line removal: crease / boundary / silhouette edges, visibility from a
             high-resolution depth buffer, output as exact vector polylines   (DAWN mode 4)
  section    exact 2-D cut of every solid by a plane (closed outlines, optional colour fill;
             --field MAP draws the magnetic field of a 2-D map under it)
  wireframe  feature (or all) edges without hidden-line removal               (DAWN mode 1)
  surface    flat-shaded hidden-surface image + visible edges                 (DAWN mode 3)
Half-space cut-aways (DAWNCUT equivalent, any number of planes, capped) work with all modes;
--trim cuts only selected volumes (e.g. the beam line beyond the detector).
Annotations: callout labels (--label REGEX=Text, --labels solid) anchored on visible lines,
a beam-direction arrow (--beam-arrow), title and subtitle.

Hard dependency: numpy.  SVG, PDF and EPS are written by hand; PNG needs pycairo or
matplotlib (whichever is importable).  Greek letters in titles and labels are drawn as glyph
outlines taken through either of them when available.  Python >= 3.8.

Run `solid_lineview --help` for the command line; docs/visualization.md of solid_dd4hep describes
the workflow and the options.
"""

import argparse
import math
import os
import re
import sys
import time
import zlib
import struct
import warnings
from collections import Counter, defaultdict

import numpy as np

__version__ = "0.6.0"

TWO_PI = 2.0 * math.pi
VERBOSE = 1
_TIMINGS = []


def log(msg, level=1):
    if VERBOSE >= level:
        print(msg, file=sys.stderr, flush=True)


class timed:
    """Context manager recording wall-clock time of a pipeline stage (--profile)."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        _TIMINGS.append((self.label, dt))
        log("  [%7.2f s] %s" % (dt, self.label), 2)
        return False


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


_UNITS = {"nm": 1e-6, "um": 1e-3, "mm": 1.0, "cm": 10.0, "dm": 100.0, "m": 1000.0}


def parse_length(s):
    """'12', '12mm', '1.5m', '-30cm' -> mm (Geant4 .prim files are in mm)."""
    if isinstance(s, (int, float)):
        return float(s)
    m = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([a-z]*)\s*", s)
    if not m:
        raise argparse.ArgumentTypeError("bad length '%s'" % s)
    u = m.group(2) or "mm"
    if u not in _UNITS:
        raise argparse.ArgumentTypeError("unknown unit '%s' (use nm um mm cm m)" % u)
    return float(m.group(1)) * _UNITS[u]


def parse_color(s):
    s = s.strip()
    named = {"black": (0, 0, 0), "white": (1, 1, 1), "red": (1, 0, 0), "green": (0, 0.6, 0),
             "blue": (0, 0, 1), "gray": (0.5, 0.5, 0.5), "grey": (0.5, 0.5, 0.5),
             "none": None}
    if s.lower() in named:
        return named[s.lower()]
    if s.startswith("#") and len(s) == 7:
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    parts = [float(x) for x in re.split(r"[ ,]+", s) if x]
    if len(parts) == 3:
        if max(parts) > 1.0:
            parts = [p / 255.0 for p in parts]
        return tuple(parts)
    raise argparse.ArgumentTypeError("bad colour '%s' (name, #rrggbb or r,g,b)" % s)


# =============================================================================
#  1.  Primitive tessellation (DAWN / g4.prim CSG primitives -> polygon meshes)
# =============================================================================
#  All builders return (V, F): V (n,3) local coordinates, F (m,4) int vertex indices with
#  -1 padding for triangles.  Duplicated vertices (axis points, collapsed rings) are
#  welded and degenerate faces dropped by _clean_mesh.  Faces are oriented outwards
#  (counter-clockwise seen from outside, same as G4Polyhedron); _orient_closed() flips a
#  closed mesh with negative signed volume, as a safety net.

_HEXA_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                        [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)


def _hexahedron(bottom, top, dz):
    """bottom/top: 4 (x,y) points ordered (-x,-y), (+x,-y), (+x,+y), (-x,+y)."""
    V = np.array([[x, y, -dz] for x, y in bottom] + [[x, y, dz] for x, y in top], float)
    return V, _HEXA_FACES.copy()


def prim_box(dx, dy, dz):
    q = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return _hexahedron(q, q, dz)


def prim_trd(dx1, dx2, dy1, dy2, dz):
    return _hexahedron([(-dx1, -dy1), (dx1, -dy1), (dx1, dy1), (-dx1, dy1)],
                       [(-dx2, -dy2), (dx2, -dy2), (dx2, dy2), (-dx2, dy2)], dz)


def prim_trap(dz, th, ph, h1, bl1, tl1, a1, h2, bl2, tl2, a2):
    """DAWN /Trap (alpha has the opposite sign of G4Trap; see FRTrapForBrep.cc)."""
    cx = dz * math.tan(th) * math.cos(ph)
    cy = dz * math.tan(th) * math.sin(ph)
    d1 = h1 * math.tan(a1)
    d2 = h2 * math.tan(a2)
    bot = [(-bl1 + d1 - cx, -h1 - cy), (bl1 + d1 - cx, -h1 - cy),
           (tl1 - d1 - cx, h1 - cy), (-tl1 - d1 - cx, h1 - cy)]
    top = [(-bl2 + d2 + cx, -h2 + cy), (bl2 + d2 + cx, -h2 + cy),
           (tl2 - d2 + cx, h2 + cy), (-tl2 - d2 + cx, h2 + cy)]
    return _hexahedron(bot, top, dz)


def prim_para(dx, dy, dz, tan_alpha, tt_cphi, tt_sphi):
    cx, cy, d = dz * tt_cphi, dz * tt_sphi, dy * tan_alpha
    bot = [(-dx - d - cx, -dy - cy), (dx - d - cx, -dy - cy),
           (dx + d - cx, dy - cy), (-dx + d - cx, dy - cy)]
    top = [(-dx - d + cx, -dy + cy), (dx - d + cx, -dy + cy),
           (dx + d + cx, dy + cy), (-dx + d + cx, dy + cy)]
    return _hexahedron(bot, top, dz)


def _nphi(ndiv, dphi):
    """Number of azimuthal steps, same rule as HepPolyhedron::RotateAroundZ."""
    return max(1, int(ndiv * min(dphi, TWO_PI) / TWO_PI + 0.5))


def revolve_ladder(I, O, sphi, dphi, nseg, closed=False):
    """Surface of revolution around z of the region between two (r,z) polylines.

    I, O : (n,2) inner / outer profile points (k-th points are 'rungs' of a ladder).
    closed : the profile polylines are closed loops (torus); otherwise the first and last
             rungs (I0-O0, In-On) are part of the boundary (end discs / cones).
    Partial phi ranges get planar end caps built from the ladder quads.
    """
    I = np.asarray(I, float)
    O = np.asarray(O, float)
    n = len(I)
    full = dphi >= TWO_PI - 1e-9
    nph = nseg if full else nseg + 1
    phis = sphi + (dphi if not full else TWO_PI) * np.arange(nph) / nseg
    prof = np.vstack([I, O])                                    # point p: 0..n-1 inner, n.. outer
    c, s = np.cos(phis), np.sin(phis)
    V = np.empty((2 * n, nph, 3))
    V[:, :, 0] = prof[:, 0:1] * c[None, :]
    V[:, :, 1] = prof[:, 0:1] * s[None, :]
    V[:, :, 2] = prof[:, 1:2]
    V = V.reshape(-1, 3) + 0.0                                  # -0.0 -> 0.0 for welding
    j = np.arange(nseg)
    j1 = (j + 1) % nph if full else j + 1

    def vid(p, jj):
        return p * nph + jj

    # profile boundary traversed clockwise in (r,z) => outward normals for quads below
    edges = [(k, k + 1) for k in range(n - 1)]
    edges += [(n - 1, 0)] if closed else [(n - 1, 2 * n - 1)]
    edges += [(n + k, n + k - 1) for k in range(n - 1, 0, -1)]
    edges += [(n, 2 * n - 1)] if closed else [(n, 0)]
    faces = []
    for p, q in edges:
        faces.append(np.stack([vid(p, j), vid(q, j), vid(q, j1), vid(p, j1)], 1))
    if not full:
        ks = list(range(n - 1)) + ([n - 1] if closed else [])
        for k in ks:
            k1 = (k + 1) % n
            e = nph - 1
            faces.append(np.array([[vid(k, e), vid(k1, e), vid(n + k1, e), vid(n + k, e)],
                                   [vid(n + k, 0), vid(n + k1, 0), vid(k1, 0), vid(k, 0)]]))
    return V, np.vstack(faces).astype(np.int64)


def prim_tubs(rmin, rmax, dz, sphi, dphi, ndiv):
    return revolve_ladder([(rmin, -dz), (rmin, dz)], [(rmax, -dz), (rmax, dz)],
                          sphi, dphi, _nphi(ndiv, dphi))


def prim_cons(rmin1, rmax1, rmin2, rmax2, dz, sphi, dphi, ndiv):
    return revolve_ladder([(rmin1, -dz), (rmin2, dz)], [(rmax1, -dz), (rmax2, dz)],
                          sphi, dphi, _nphi(ndiv, dphi))


def prim_sphereseg(rmin, rmax, sth, dth, sphi, dphi, ndiv):
    nth = max(1, int(((ndiv + 1) // 2) * min(dth, math.pi) / math.pi + 0.5))
    th = (sth + dth) - dth * np.arange(nth + 1) / nth                 # z increasing
    I = np.stack([rmin * np.sin(th), rmin * np.cos(th)], 1)
    O = np.stack([rmax * np.sin(th), rmax * np.cos(th)], 1)
    return revolve_ladder(I, O, sphi, dphi, _nphi(ndiv, dphi))


def prim_torus(rmin, rmax, rtor, sphi, dphi, ndiv):
    na = max(3, ndiv)
    a = TWO_PI * np.arange(na) / na
    I = np.stack([rtor + rmin * np.cos(a), rmin * np.sin(a)], 1)
    O = np.stack([rtor + rmax * np.cos(a), rmax * np.sin(a)], 1)
    return revolve_ladder(I, O, sphi, dphi, _nphi(ndiv, dphi), closed=True)


def prim_polycone(sphi, dphi, z, rmin, rmax, ndiv, nsides=None):
    I = np.stack([rmin, z], 1)
    O = np.stack([rmax, z], 1)
    if nsides:      # DAWN /PolyGon: radii are corner radii, nsides steps over dphi
        return revolve_ladder(I, O, sphi, dphi, int(nsides))
    return revolve_ladder(I, O, sphi, dphi, _nphi(ndiv, dphi))


def _weld(V, F):
    """Merge bit-identical vertices; returns (V', F')."""
    if len(V) == 0:
        return V, F
    Vc = V + 0.0
    order = np.lexsort((Vc[:, 2], Vc[:, 1], Vc[:, 0]))
    Vs = Vc[order]
    new = np.ones(len(V), bool)
    new[1:] = np.any(Vs[1:] != Vs[:-1], axis=1)
    uid = np.cumsum(new) - 1
    remap = np.empty(len(V), np.int64)
    remap[order] = uid
    Vn = Vs[new]
    Fn = np.where(F >= 0, remap[np.maximum(F, 0)], -1)
    return Vn, Fn


def _clean_faces(F, Fvis=None):
    """Drop repeated consecutive vertices; quads with 3 distinct vertices become
    triangles, faces with < 3 distinct vertices are removed."""
    F = np.asarray(F, np.int64)
    if Fvis is None:
        Fvis = np.ones(F.shape, bool)
    m = len(F)
    if m == 0:
        return F.reshape(0, 4), Fvis.reshape(0, 4)
    nv = np.where(F[:, 3] >= 0, 4, 3)
    nxt = np.where(nv[:, None] == 4, F[:, [1, 2, 3, 0]], F[:, [1, 2, 0, 0]])
    keep = F != nxt
    keep[:, 3] &= nv == 4
    keep &= F >= 0
    cnt = keep.sum(1)
    ok = cnt >= 3
    if np.all(cnt[ok] == nv[ok]) and ok.all():
        return F, Fvis
    order = np.argsort(~keep, axis=1, kind="stable")
    Fk = np.take_along_axis(F, order, 1)
    Vk = np.take_along_axis(Fvis, order, 1)
    Fk[:, 3] = np.where(cnt == 4, Fk[:, 3], -1)
    return Fk[ok], Vk[ok]


def _signed_volume(V, F):
    tri = _triangulate(F)
    a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _triangulate(F):
    """(m,4) faces -> (k,3) triangles (fan); also returns nothing else."""
    tri = F[:, :3]
    q = F[:, 3] >= 0
    return np.vstack([tri, F[q][:, [0, 2, 3]]])


def finish_primitive(V, F):
    V, F = _weld(V, F)
    F, _ = _clean_faces(F)
    if len(F) and _signed_volume(V, F) < 0:
        F = _flip_faces(F)
    return V, F


def _flip_faces(F):
    G = F.copy()
    q = F[:, 3] >= 0
    G[q] = F[q][:, ::-1]
    G[~q, :3] = F[~q][:, 2::-1]
    return G


# =============================================================================
#  2.  .prim parser
# =============================================================================

class Solid:
    __slots__ = ("index", "name", "depth", "color", "wire", "kind", "v0", "v1", "f0", "f1",
                 "line", "parent", "path", "closed", "bmin", "bmax", "has_daughters")

    def __repr__(self):
        return "Solid(%d %s d=%d %s)" % (self.index, self.name, self.depth, self.kind)


class PrimScene:
    """Everything read from a .prim file, geometry concatenated into global arrays.

    V      (Nv,3)  world coordinates [mm]
    F      (Nf,4)  vertex indices (-1 padded triangles)
    Fvis   (Nf,4)  DAWN edge-visibility flag of edge k (vertex k -> k+1)
    fsolid (Nf,)   owning solid index
    """

    def __init__(self):
        self.solids = []
        self.V = np.zeros((0, 3))
        self.F = np.zeros((0, 4), np.int64)
        self.Fvis = np.zeros((0, 4), bool)
        self.fsolid = np.zeros(0, np.int32)
        self.polylines = []      # (rgb, (n,3))
        self.markers = []        # (kind 'circle'|'square', rgb, xyz, size, screen?)
        self.texts = []          # (rgb, xyz, size, dx, dy, text, screen?, font)
        self.texts2d = []        # (rgb, x_mm, y_mm, size_pt, text, font)
        self.bbox = None         # /BoundingBox
        self.clip_comments = []  # '#/ClippingPlane a b c d' written by DAWNCUT
        self.counts = Counter()
        self.warnings = Counter()


def _floats(s):
    return [float(x) for x in s.split()]


def _fromstring(s, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.fromstring(s, dtype=dtype, sep=" ")


def parse_prim(path):
    """Parse a g4.prim (format 2.4) file written by Geant4's DAWNFILE driver."""
    with open(path, "rb") as fh:
        raw = fh.read()
    # Geant4 11.x writes /MarkText2D(S) strings through a dangling pointer (G4Text::GetText
    # returns by value), so arbitrary bytes can appear: never fail on decoding.
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    sc = PrimScene()

    color = (1.0, 1.0, 1.0)
    wire = False
    visible = True
    ndiv = 24
    font = "Times-Roman"
    O = np.zeros(3)
    M = np.eye(3)                         # rows: e1, e2, e3 (local -> world)
    pv_name, pv_depth = None, -1
    blocks_V, blocks_F, blocks_Fvis = [], [], []
    nvtot = 0
    nftot = 0
    cache = {}

    def to_world(P):
        return P @ M + O

    def add_solid(kind, Vloc, F, Fvis, lineno):
        nonlocal nvtot, nftot, pv_name, pv_depth
        if not visible or len(F) == 0:
            pv_name = None
            return
        s = Solid()
        s.index = len(sc.solids)
        s.name = pv_name if pv_name is not None else "<%s@%d>" % (kind, lineno)
        s.depth = pv_depth if pv_name is not None else -1
        s.color = color
        s.wire = wire
        s.kind = kind
        s.line = lineno
        s.v0, s.v1 = nvtot, nvtot + len(Vloc)
        s.f0, s.f1 = nftot, nftot + len(F)
        Vw = to_world(Vloc)
        s.bmin, s.bmax = Vw.min(0), Vw.max(0)
        blocks_V.append(Vw)
        blocks_F.append(np.where(F >= 0, F + nvtot, -1))
        blocks_Fvis.append(Fvis)
        nvtot += len(Vloc)
        nftot += len(F)
        sc.solids.append(s)
        sc.counts[kind] += 1
        pv_name = None

    def primitive(cmd, args, lineno):
        key = (cmd, args, ndiv)
        hit = cache.get(key)
        if hit is None:
            v = _floats(args)
            try:
                if cmd == "/Box":
                    VF = prim_box(*v[:3])
                elif cmd == "/Column":
                    VF = prim_tubs(0.0, v[0], v[1], 0.0, TWO_PI, ndiv)
                elif cmd == "/Tubs":
                    VF = prim_tubs(*v[:5], ndiv)
                elif cmd == "/Cons":
                    VF = prim_cons(*v[:7], ndiv)
                elif cmd == "/Trd":
                    VF = prim_trd(*v[:5])
                elif cmd == "/Trap":
                    VF = prim_trap(*v[:11])
                elif cmd == "/Parallelepiped":
                    VF = prim_para(*v[:6])
                elif cmd == "/Sphere":
                    VF = prim_sphereseg(0.0, v[0], 0.0, math.pi, 0.0, TWO_PI, ndiv)
                elif cmd == "/SphereSeg":
                    VF = prim_sphereseg(*v[:6], ndiv)
                elif cmd == "/Torus":
                    VF = prim_torus(*v[:5], ndiv)
                elif cmd in ("/PolyCone", "/PolyGon"):
                    if cmd == "/PolyGon":
                        sphi, dphi, nsides, nz = v[0], v[1], int(v[2]), int(v[3])
                        rest = v[4:]
                    else:
                        sphi, dphi, nz = v[0], v[1], int(v[2])
                        nsides = None
                        rest = v[3:]
                    z = np.array(rest[:nz])
                    rmin = np.array(rest[nz:2 * nz])
                    rmax = np.array(rest[2 * nz:3 * nz])
                    VF = prim_polycone(sphi, dphi, z, rmin, rmax, ndiv, nsides)
                else:
                    return
            except (TypeError, ValueError, IndexError) as exc:
                sc.warnings["bad %s at line %d: %s" % (cmd, lineno, exc)] += 1
                return
            hit = finish_primitive(*VF)
            cache[key] = hit
        Vloc, F = hit
        add_solid(cmd[1:], Vloc, F, np.ones(F.shape, bool), lineno)

    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        i += 1
        if not s:
            continue
        c0 = s[0]
        if c0 == "#":
            if s.startswith("#/PVName"):
                rest = s[8:]
                nsp = len(rest) - len(rest.lstrip(" "))
                pv_depth = max(0, (nsp - 5) // 5)
                pv_name = rest.strip()
            elif s.startswith("#/ClippingPlane"):
                try:
                    sc.clip_comments.append(tuple(_floats(s[15:])[:4]))
                except ValueError:
                    pass
            continue
        if c0 == "!":
            if s.startswith("!ClearData"):
                blocks_V.clear(); blocks_F.clear(); blocks_Fvis.clear()
                sc.solids.clear(); sc.polylines.clear(); sc.markers.clear()
                sc.texts.clear(); sc.texts2d.clear()
                nvtot = nftot = 0
            continue
        parts = s.split(None, 1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        if cmd == "/Polyhedron":
            start = i
            vl, fl = [], []
            while i < n:
                b = lines[i].strip()
                i += 1
                if b.startswith("/Vertex"):
                    vl.append(b[7:])
                elif b.startswith("/Facet"):
                    fl.append(b[6:])
                elif b.startswith("/EndPolyhedron"):
                    break
                elif b and not b.startswith("#"):
                    sc.warnings["unexpected '%s' inside /Polyhedron" % b.split()[0]] += 1
            Vloc = _fromstring(" ".join(vl), float)
            if len(Vloc) != 3 * len(vl):
                Vloc = np.array([_floats(x)[:3] for x in vl], float)
            Vloc = Vloc.reshape(-1, 3)
            F, Fvis = _parse_facets(fl, len(Vloc), sc, start)
            # merge exact duplicates (robust adjacency), drop degenerate facets
            Vloc, F = _weld(Vloc, F)
            F, Fvis = _clean_faces(F, Fvis)
            add_solid("Polyhedron", Vloc, F, Fvis, start)
        elif cmd in ("/Box", "/Column", "/Tubs", "/Cons", "/Trd", "/Trap", "/Parallelepiped",
                     "/Sphere", "/SphereSeg", "/Torus", "/PolyCone", "/PolyGon"):
            primitive(cmd, args, i)
        elif cmd == "/ColorRGB" or cmd == "/DiffuseRGB":
            try:
                color = tuple(min(1.0, max(0.0, x)) for x in _floats(args)[:3])
            except ValueError:
                sc.warnings["bad /ColorRGB"] += 1
        elif cmd == "/ForceWireframe" or cmd == "/Transparency":
            wire = args.strip()[:1] not in ("0", "")
        elif cmd == "/Visibility":
            visible = args.strip()[:1] != "0"
        elif cmd == "/Origin":
            O = np.array(_floats(args)[:3])
        elif cmd == "/BaseVector":
            v = _floats(args)
            e1, e2 = np.array(v[:3]), np.array(v[3:6])
            M = np.vstack([e1, e2, np.cross(e1, e2)])
        elif cmd == "/Ndiv":
            try:
                ndiv = max(3, int(float(args)))
            except ValueError:
                pass
        elif cmd == "/BoundingBox":
            sc.bbox = np.array(_floats(args)[:6])
        elif cmd == "/Polyline":
            pts = []
            while i < n:
                b = lines[i].strip()
                i += 1
                if b.startswith("/PLVertex") or b.startswith("PLVertex"):
                    pts.append(b.split(None, 1)[1])
                elif b.startswith("/EndPolyline"):
                    break
            if pts and visible:
                P = _fromstring(" ".join(pts), float).reshape(-1, 3)
                sc.polylines.append((color, to_world(P)))
                sc.counts["Polyline"] += 1
        elif cmd in ("/MarkCircle2D", "/MarkCircle2DS", "/MarkSquare2D", "/MarkSquare2DS"):
            v = _floats(args)
            if len(v) >= 4 and visible:
                kind = "circle" if "Circle" in cmd else "square"
                sc.markers.append((kind, color, to_world(np.array([v[:3]]))[0], v[3],
                                   cmd.endswith("S")))
                sc.counts[cmd[1:]] += 1
        elif cmd in ("/MarkText2D", "/MarkText2DS"):
            t = args.split(None, 6)
            if len(t) >= 7 and visible:
                try:
                    xyz = to_world(np.array([[float(t[0]), float(t[1]), float(t[2])]]))[0]
                    sc.texts.append((color, xyz, float(t[3]), float(t[4]), float(t[5]),
                                     t[6], cmd.endswith("S"), font))
                    sc.counts[cmd[1:]] += 1
                except ValueError:
                    sc.warnings["bad %s" % cmd] += 1
        elif cmd == "/Text2DS":
            t = args.split(None, 3)
            if len(t) >= 4:
                try:
                    sc.texts2d.append((color, float(t[0]), float(t[1]), float(t[2]), t[3], font))
                except ValueError:
                    pass
        elif cmd == "/FontName":
            font = args.strip() or font
        elif cmd in ("/SpecularRGB", "/PhongPower", "/CameraPosition", "/TargetPoint",
                     "/ZoomFactor", "/Scale", "/FocalDistance", "/Wireframe", "/Surface",
                     "/Lines"):
            pass
        else:
            sc.warnings["unknown command %s" % cmd] += 1

    if blocks_V:
        sc.V = np.vstack(blocks_V)
        sc.F = np.vstack(blocks_F).astype(np.int64)
        sc.Fvis = np.vstack(blocks_Fvis)
        sc.fsolid = np.repeat(np.arange(len(sc.solids), dtype=np.int32),
                              [s.f1 - s.f0 for s in sc.solids])
    _build_hierarchy(sc)
    return sc


def load_scene(path, drawing_style=False):
    """Read a geometry scene into a PrimScene: a scene.npz (see solid_lineview.scene) or a DAWNFILE .prim.

    The kind of file is taken from its content (a .npz is a zip archive), so the name does not matter.
    drawing_style applies to scene.npz only, see scene.load_scene_npz().
    """
    from . import scene
    if scene.is_scene_file(path):
        return scene.load_scene_npz(path, drawing_style=drawing_style)
    return parse_prim(path)


def _parse_facets(fl, nv, sc, lineno):
    m = len(fl)
    if m == 0:
        return np.zeros((0, 4), np.int64), np.zeros((0, 4), bool)
    flat = _fromstring(" ".join(fl), np.int64)
    if len(flat) == 3 * m:
        A = np.full((m, 4), 0, np.int64)
        A[:, :3] = flat.reshape(m, 3)
        cnt = np.full(m, 3)
    elif len(flat) == 4 * m:
        A = flat.reshape(m, 4)
        cnt = np.full(m, 4)
    else:
        cnt = np.array([len(x.split()) for x in fl])
        if cnt.sum() != len(flat):          # comments or garbage: slow path
            rows = [[int(t) for t in x.split() if re.fullmatch(r"-?\d+", t)] for x in fl]
            cnt = np.array([len(r) for r in rows])
            flat = np.array([t for r in rows for t in r], np.int64)
        mx = max(4, int(cnt.max()))
        A = np.zeros((m, mx), np.int64)
        row = np.repeat(np.arange(m), cnt)
        col = np.arange(len(flat)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        A[row, col] = flat
    mx = A.shape[1]
    colidx = np.arange(mx)[None, :]
    valid = colidx < cnt[:, None]
    # DAWN: edge (v_{j-1} -> v_j) is invisible if label v_j is negative
    neg = (A < 0) & valid
    evis = ~np.roll(neg, -1, axis=1)
    # roll wraps around within the valid count only for full rows; fix rows with cnt < mx
    if np.any(cnt < mx):
        evis = np.ones_like(neg)
        rr, cc = np.nonzero(neg)
        evis[rr, (cc - 1) % cnt[rr]] = False
    idx = np.abs(A) - 1
    bad = valid & ((idx < 0) | (idx >= nv))
    if bad.any():
        sc.warnings["facet index out of range near line %d" % lineno] += 1
        keep = ~bad.any(1)
        idx, valid, evis, cnt = idx[keep], valid[keep], evis[keep], cnt[keep]
    F, Fv = [], []
    for k in (3, 4):
        sel = cnt == k
        if sel.any():
            f = np.full((sel.sum(), 4), -1, np.int64)
            f[:, :k] = idx[sel][:, :k]
            v = np.ones((sel.sum(), 4), bool)
            v[:, :k] = evis[sel][:, :k]
            F.append(f)
            Fv.append(v)
    big = cnt > 4                                   # n-gons: fan triangulation
    for r in np.nonzero(big)[0]:
        k = cnt[r]
        for t in range(1, k - 1):
            F.append(np.array([[idx[r, 0], idx[r, t], idx[r, t + 1], -1]]))
            Fv.append(np.array([[t == 1 and evis[r, 0], evis[r, t],
                                 t == k - 2 and evis[r, k - 1], True]]))
    return np.vstack(F), np.vstack(Fv)


def _build_hierarchy(sc):
    """Depth-first order + #/PVName indentation -> parent links.  Invisible envelopes are
    not written by Geant4, so a candidate parent must also contain the child's bbox."""
    stack = []
    for s in sc.solids:
        s.parent = -1
        s.has_daughters = False
        if s.depth < 0:
            s.path = s.name
            continue
        while stack and stack[-1].depth >= s.depth:
            stack.pop()
        if stack and stack[-1].depth == s.depth - 1:
            p = stack[-1]
            tol = 1e-6 * (1.0 + float(np.abs(p.bmax - p.bmin).max()))
            if np.all(s.bmin >= p.bmin - tol) and np.all(s.bmax <= p.bmax + tol):
                s.parent = p.index
                p.has_daughters = True
        s.path = (sc.solids[s.parent].path + "/" if s.parent >= 0 else "") + s.name
        stack.append(s)


# =============================================================================
#  3.  Mesh analysis: normals, triangles, edges
# =============================================================================

class Mesh:
    """Static (view independent) derived data of a PrimScene."""

    def __init__(self, sc, crease_deg):
        self.sc = sc
        V, F = sc.V, sc.F
        nf = len(F)
        quad = F[:, 3] >= 0
        a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        d = np.where(quad[:, None], V[np.maximum(F[:, 3], 0)], a)
        nrm = np.where(quad[:, None], np.cross(c - a, d - b), np.cross(b - a, c - a))
        ln = np.linalg.norm(nrm, axis=1)
        self.fnormal = nrm / np.maximum(ln, 1e-300)[:, None]
        self.farea = 0.5 * ln
        # triangles (quads split along 0-2)
        qi = np.nonzero(quad)[0]
        self.tri = np.vstack([F[:, :3], F[qi][:, [0, 2, 3]]])
        self.tri_face = np.concatenate([np.arange(nf), qi]).astype(np.int64)
        # edges of the original polygons
        nxt = np.where(quad[:, None], F[:, [1, 2, 3, 0]], F[:, [1, 2, 0, 0]])
        valid = np.ones((nf, 4), bool)
        valid[:, 3] = quad
        ea, eb = F[valid], nxt[valid]
        ef = np.broadcast_to(np.arange(nf)[:, None], (nf, 4))[valid]
        evis = sc.Fvis[valid]
        lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
        nv = max(1, len(V))
        key = lo * nv + hi
        order = np.argsort(key, kind="stable")
        key, lo, hi, ef, evis = key[order], lo[order], hi[order], ef[order], evis[order]
        fwd = (ea < eb)[order]
        first = np.ones(len(key), bool)
        first[1:] = key[1:] != key[:-1]
        start = np.nonzero(first)[0]
        cnt = np.diff(np.append(start, len(key)))
        self.e0, self.e1 = lo[start], hi[start]
        self.ef1 = ef[start]
        has2 = cnt >= 2
        self.ef2 = np.where(has2, ef[np.minimum(start + 1, len(key) - 1)], -1)
        self.ecount = cnt
        # DAWN edge flags: drawn as feature only if visible in at least one face
        evis_any = np.maximum.reduceat(evis.astype(np.int8), start).astype(bool) \
            if len(start) else np.zeros(0, bool)
        self.esolid = sc.fsolid[self.ef1] if len(self.ef1) else np.zeros(0, np.int32)
        # orientation consistency (both faces traverse the edge in opposite directions)
        same_dir = has2 & (fwd[start] == fwd[np.minimum(start + 1, len(key) - 1)])
        n1 = self.fnormal[self.ef1]
        n2 = self.fnormal[np.maximum(self.ef2, 0)]
        cosang = np.einsum("ij,ij->i", n1, n2)
        crease = has2 & (cosang < math.cos(math.radians(crease_deg)))
        self.ecrease = crease & evis_any
        self.eboundary = (cnt == 1) & evis_any
        self.enonmanifold = cnt > 2
        self.efeature = self.ecrease | self.eboundary | self.enonmanifold
        # per solid: closed = every edge has exactly two faces with opposite orientation
        nsol = len(sc.solids)
        openish = (cnt != 2) | same_dir
        bad = np.bincount(self.esolid[openish], minlength=nsol) if nsol else np.zeros(0)
        for s in sc.solids:
            s.closed = bool(bad[s.index] == 0)
        self.n_inconsistent = int(same_dir.sum())


# =============================================================================
#  4.  Camera
# =============================================================================

class Camera:
    """eye: unit vector from the target towards the camera (Geant4 'viewpoint direction').
    dist None -> orthographic, else perspective with the camera at target + dist*eye."""

    def __init__(self, eye, up=(0, 1, 0), target=(0, 0, 0), dist=None):
        e = _unit(eye)
        u0 = _unit(up)
        r = np.cross(u0, e)
        if np.linalg.norm(r) < 1e-9:
            # DAWN / Geant4 rule for a viewpoint along the up vector: screen-right = +z
            r = np.array([0.0, 0.0, 1.0]) if abs(e[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            r = r - np.dot(r, e) * e
        r = _unit(r)
        self.e, self.r, self.u = e, r, np.cross(e, r)
        self.T = np.asarray(target, float)
        self.dist = dist
        self.ortho = dist is None

    @property
    def position(self):
        return None if self.ortho else self.T + self.dist * self.e

    def view(self, P):
        """world (N,3) -> (x, y, depth) ; for perspective x,y are scaled to the target plane."""
        d = np.asarray(P, float) - self.T
        x = d @ self.r
        y = d @ self.u
        z = -(d @ self.e)
        if self.ortho:
            return x, y, z
        depth = self.dist + z
        f = self.dist / np.maximum(depth, 1e-9)
        return x * f, y * f, depth

    def zkey(self, depth):
        """A depth key that is affine in screen coordinates (1/w for perspective)."""
        return depth if self.ortho else -1.0 / np.maximum(depth, 1e-9)

    def facing(self, normals, points):
        """>0 for faces whose outward normal points to the camera."""
        if self.ortho:
            return normals @ self.e
        return np.einsum("ij,ij->i", normals, self.position[None, :] - points)

    def near_plane(self):
        if self.ortho:
            return None
        n = -self.e
        C = self.position
        near = 1e-3 * self.dist
        return (n, float(-(n @ C) - near))  # keep  n.P + d > 0  <=>  (C-P).e > near


def dawn_angles_to_eye(polar_deg, azim_deg):
    """DAWN camera angles (.DAWN_1.history lines 2,3) = Geant4 /vis/viewer/set/
    viewpointThetaPhi: polar angle from +z, azimuth from +x of the camera direction."""
    t, p = math.radians(polar_deg), math.radians(azim_deg)
    return np.array([math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)])


VIEW_PRESETS = {
    # name: (eye = target->camera direction, up)
    "side": ((-1.0, 0.0, 0.0), (0, 1, 0)),     # camera at -x: z to the right, y up
    "side-x": ((1.0, 0.0, 0.0), (0, 1, 0)),    # camera at +x: z to the left, y up
    "top": ((0.0, 1.0, 0.0), (0, 1, 0)),       # camera at +y: z to the right, x up (DAWN rule)
    "bottom": ((0.0, -1.0, 0.0), (0, 1, 0)),
    "beam": ((0.0, 0.0, 1.0), (0, 1, 0)),      # from downstream: x to the right, y up
    "front": ((0.0, 0.0, -1.0), (0, 1, 0)),    # from upstream (beam's eye): x to the left
    "iso": ((-0.75, 0.48, -0.45), (0, 1, 0)),  # oblique, beam left -> right
    "iso-down": ((-0.75, 0.48, 0.45), (0, 1, 0)),
}

SECTION_FOR_VIEW = {"side": (1, 0, 0), "side-x": (1, 0, 0), "top": (0, 1, 0),
                    "bottom": (0, 1, 0), "beam": (0, 0, 1), "front": (0, 0, 1)}


# =============================================================================
#  5.  Half-space clipping and plane sections
# =============================================================================
#  A plane is (n, d) with n a unit 3-vector; the KEPT half space is n.P + d >= 0
#  (Geant4 cut-away convention, /vis/viewer/addCutawayPlane).

def clip_triangles(P, fid, planes):
    """P (M,3,3) world triangles, fid (M,) payload -> clipped (M',3,3), fid'."""
    for n, d in planes:
        if len(P) == 0:
            break
        s = P @ n + d
        ins = s >= 0
        k = ins.sum(1)
        keep = P[k == 3], fid[k == 3]
        out = [keep]
        # one vertex inside -> one triangle
        m1 = k == 1
        if m1.any():
            Q, S, f = P[m1], s[m1], fid[m1]
            r = np.argmax(ins[m1], 1)
            idx = (r[:, None] + np.arange(3)[None, :]) % 3
            Q = np.take_along_axis(Q, idx[:, :, None], 1)
            S = np.take_along_axis(S, idx, 1)
            t1 = (S[:, 0] / (S[:, 0] - S[:, 1]))[:, None]
            t2 = (S[:, 0] / (S[:, 0] - S[:, 2]))[:, None]
            A = Q[:, 0]
            out.append((np.stack([A, A + t1 * (Q[:, 1] - A), A + t2 * (Q[:, 2] - A)], 1), f))
        m2 = k == 2
        if m2.any():
            Q, S, f = P[m2], s[m2], fid[m2]
            r = np.argmin(ins[m2], 1)                      # the outside vertex first
            idx = (r[:, None] + np.arange(3)[None, :]) % 3
            Q = np.take_along_axis(Q, idx[:, :, None], 1)
            S = np.take_along_axis(S, idx, 1)
            Ov, A, B = Q[:, 0], Q[:, 1], Q[:, 2]
            tb = (S[:, 2] / (S[:, 2] - S[:, 0]))[:, None]  # on B -> O
            ta = (S[:, 0] / (S[:, 0] - S[:, 1]))[:, None]  # on O -> A
            Qb = B + tb * (Ov - B)
            Qa = Ov + ta * (A - Ov)
            out.append((np.stack([A, B, Qb], 1), f))
            out.append((np.stack([A, Qb, Qa], 1), f))
        P = np.concatenate([o[0] for o in out])
        fid = np.concatenate([o[1] for o in out])
    return P, fid


def clip_segments(A, B, planes, *payload):
    """Clip segments A->B (N,3) by half spaces; payload arrays are filtered alongside."""
    keep = np.ones(len(A), bool)
    A = A.copy()
    B = B.copy()
    for n, d in planes:
        sa = A @ n + d
        sb = B @ n + d
        keep &= (sa >= 0) | (sb >= 0)
        ca = (sa < 0) & (sb >= 0)
        cb = (sb < 0) & (sa >= 0)
        if ca.any():
            t = (sa[ca] / (sa[ca] - sb[ca]))[:, None]
            A[ca] = A[ca] + t * (B[ca] - A[ca])
        if cb.any():
            t = (sb[cb] / (sb[cb] - sa[cb]))[:, None]
            B[cb] = B[cb] + t * (A[cb] - B[cb])
    return (A[keep], B[keep]) + tuple(p[keep] for p in payload)


def clip_segments_masked(A, B, sol, trims, *payload):
    """clip_segments with per-volume planes (n, d, solid_mask): a plane only cuts the segments of
    its volumes; sol (volume of each segment) is returned filtered after A, B."""
    for n, d, mask in trims:
        m = mask[sol]
        if not m.any():
            continue
        r = clip_segments(A[m], B[m], [(n, d)], sol[m], *[p[m] for p in payload])
        A = np.concatenate([A[~m], r[0]])
        B = np.concatenate([B[~m], r[1]])
        sol = np.concatenate([sol[~m], r[2]])
        payload = tuple(np.concatenate([p[~m], q]) for p, q in zip(payload, r[3:]))
    return (A, B, sol) + tuple(payload)


def clip_triangles_masked(P, fid, fsolid, trims):
    """clip_triangles with per-volume planes (n, d, solid_mask); fid are face indices."""
    for n, d, mask in trims:
        m = mask[fsolid[fid]]
        if m.any():
            Q, f = clip_triangles(P[m], fid[m], [(n, d)])
            P = np.concatenate([P[~m], Q])
            fid = np.concatenate([fid[~m], f])
    return P, fid


def snap_vertices(sc, V, tri, planes, snap, section):
    """Vertices V of the scene (sc.V or a copy), with those closer than `snap` to one of the planes moved off it by `snap`.

    A face lying in a cut or section plane (the side of a detector sector at x = 0) is otherwise cut,
    kept or dropped depending on the rounding noise of its coordinates.  Each solid's vertices near a
    plane all go to the same side:
      cut plane (section=False)  the kept side if the solid has other vertices there; a solid that only
                                 touches the plane from the removed side goes away completely
      section plane              the side that makes a solid touching the plane with a whole face be cut
                                 along that face, and leaves a solid touching with an edge or corner uncut
    The shift (1 um by default) is far below anything visible.  Returns V itself if nothing is near."""
    if snap <= 0 or not len(V) or not sc.solids:
        return V
    vsolid = None
    out = V
    for n, d in planes:
        s = out @ n + d
        on = np.abs(s) < snap
        if not on.any():
            continue
        if vsolid is None:
            vsolid = np.repeat(np.arange(len(sc.solids)), [x.v1 - x.v0 for x in sc.solids])
        nsol = len(sc.solids)
        above = np.zeros(nsol, bool)
        below = np.zeros(nsol, bool)
        above[vsolid[(s > 0) & ~on]] = True
        below[vsolid[(s < 0) & ~on]] = True
        if section:
            face = np.zeros(nsol, bool)                 # solids with a triangle in the plane
            face[vsolid[tri[on[tri].all(1), 0]]] = True
            plus = (above & below) | (above & ~face) | (below & ~above & face)
        else:
            plus = above
        if out is V:
            out = V.copy()
        target = np.where(plus[vsolid[on]], snap, -snap)
        out[on] += (target - s[on])[:, None] * np.asarray(n, float)[None, :]
    return out


def section_segments(V, tri, plane):
    """Intersect triangles (vertex index triples) with a plane.
    Returns Q0, Q1 (k,3), the triangle index, and integer node keys of the two crossed
    mesh edges (for exact topological chaining)."""
    n, d = plane
    s = V @ n + d
    st = s[tri]
    pos = st >= 0
    k = pos.sum(1)
    cross = (k == 1) | (k == 2)
    ti = np.nonzero(cross)[0]
    if len(ti) == 0:
        z = np.zeros((0, 3))
        e = np.zeros(0, np.int64)
        return z, z, e, e, e
    T = tri[ti]
    S = st[ti]
    lone = np.where(k[ti] == 1, np.argmax(pos[ti], 1), np.argmin(pos[ti], 1))
    idx = (lone[:, None] + np.arange(3)[None, :]) % 3
    T = np.take_along_axis(T, idx, 1)
    S = np.take_along_axis(S, idx, 1)
    Va, Vb, Vc = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    t1 = (S[:, 0] / (S[:, 0] - S[:, 1]))[:, None]
    t2 = (S[:, 0] / (S[:, 0] - S[:, 2]))[:, None]
    Q0 = Va + t1 * (Vb - Va)
    Q1 = Va + t2 * (Vc - Va)
    nv = len(V)
    k0 = np.minimum(T[:, 0], T[:, 1]) * nv + np.maximum(T[:, 0], T[:, 1])
    k1 = np.minimum(T[:, 0], T[:, 2]) * nv + np.maximum(T[:, 0], T[:, 2])
    return Q0, Q1, ti, k0, k1


def chain_segments(k0, k1, Q0, Q1):
    """Chain segments sharing node keys into polylines.  Returns a list of
    (points (m,3), closed) tuples.  Works on a per-call basis (call per solid)."""
    nseg = len(k0)
    if nseg == 0:
        return []
    adj = defaultdict(list)
    for i in range(nseg):
        adj[k0[i]].append((i, 0))
        adj[k1[i]].append((i, 1))
    used = np.zeros(nseg, bool)
    out = []
    # start from odd-degree nodes first (open chains), then loops
    starts = [kk for kk, v in adj.items() if len(v) % 2 == 1] + list(adj.keys())
    for node in starts:
        while True:
            nxt = None
            for (i, end) in adj[node]:
                if not used[i]:
                    nxt = (i, end)
                    break
            if nxt is None:
                break
            pts = []
            first_node = node
            cur = node
            while nxt is not None:
                i, end = nxt
                used[i] = True
                a, b = (Q0[i], Q1[i]) if end == 0 else (Q1[i], Q0[i])
                if not pts:
                    pts.append(a)
                pts.append(b)
                cur = k1[i] if end == 0 else k0[i]
                nxt = None
                for (j, e2) in adj[cur]:
                    if not used[j]:
                        nxt = (j, e2)
                        break
            closed = cur == first_node and len(pts) > 2
            if closed:
                pts = pts[:-1]
            out.append((np.array(pts), closed))
    return out


def clip_polygon(P, planes):
    """Sutherland-Hodgman clip of one closed 3-D polygon by half spaces."""
    for n, d in planes:
        if len(P) == 0:
            return P
        s = P @ n + d
        res = []
        m = len(P)
        for i in range(m):
            a, b = P[i], P[(i + 1) % m]
            sa, sb = s[i], s[(i + 1) % m]
            if sa >= 0:
                res.append(a)
            if (sa >= 0) != (sb >= 0):
                res.append(a + (sa / (sa - sb)) * (b - a))
        P = np.array(res).reshape(-1, 3)
    return P


# =============================================================================
#  6.  Depth-buffer rasterisation (vectorised scan conversion)
# =============================================================================

ID_BITS = 24
ID_MASK = (1 << ID_BITS) - 1
ZQ_MAX = (1 << (62 - ID_BITS)) - 1
EMPTY = np.iinfo(np.int64).max


class DepthBuffer:
    """Packed (quantised zkey << 24 | id) int64 buffer, scatter-min with np.minimum.at.
    ids 0..M-1 are triangles, M.. are cut-plane caps.  zkey is affine in screen space,
    so the exact surface depth at any sub-pixel point is recovered from the per-id plane
    coefficients (A x + B y + C) -- quantisation never enters the visibility test."""

    def __init__(self, W, H, zlo, zhi):
        self.W, self.H = int(W), int(H)
        self.buf = np.full(self.W * self.H, EMPTY, np.int64)
        self.zlo = float(zlo)
        span = max(float(zhi) - float(zlo), 1e-30)
        self.zscale = ZQ_MAX / span
        self.fragments = 0

    def quant(self, z):
        q = (z - self.zlo) * self.zscale
        return np.clip(q, 0, ZQ_MAX).astype(np.int64)

    def add_triangles(self, X, Y, A, B, C, ids, max_frag=6_000_000):
        """X,Y (M,3) pixel coords (pixel (i,j) centre at (i+.5, j+.5)); zkey = A x+B y+C."""
        W, H = self.W, self.H
        eps = 1e-7
        ymin = Y.min(1)
        ymax = Y.max(1)
        j0 = np.maximum(np.ceil(ymin - 0.5 - eps), 0).astype(np.int64)
        j1 = np.minimum(np.floor(ymax - 0.5 + eps), H - 1).astype(np.int64)
        xmin = X.min(1)
        xmax = X.max(1)
        ok = (j1 >= j0) & (xmax >= 0) & (xmin <= W)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            return
        area = 0.5 * np.abs((X[idx, 1] - X[idx, 0]) * (Y[idx, 2] - Y[idx, 0])
                            - (X[idx, 2] - X[idx, 0]) * (Y[idx, 1] - Y[idx, 0]))
        est = area + (j1[idx] - j0[idx] + 1) * 2.0
        cum = np.cumsum(est)
        nb = int(cum[-1] // max_frag) + 1
        bounds = np.searchsorted(cum, np.linspace(0, cum[-1], nb + 1)[1:-1])
        for part in np.split(idx, bounds):
            if len(part):
                self._raster(part, X, Y, A, B, C, ids, j0, j1)

    def _raster(self, part, X, Y, A, B, C, ids, j0, j1):
        W, H = self.W, self.H
        eps = 1e-7
        nr = j1[part] - j0[part] + 1
        tp = np.repeat(part, nr)
        row = np.repeat(j0[part], nr) + (np.arange(nr.sum()) - np.repeat(np.cumsum(nr) - nr, nr))
        yc = row + 0.5
        Xt, Yt = X[tp], Y[tp]
        xl = np.full(len(tp), np.inf)
        xr = np.full(len(tp), -np.inf)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ya, yb, xa, xb = Yt[:, a], Yt[:, b], Xt[:, a], Xt[:, b]
            lo = np.minimum(ya, yb) - eps
            hi = np.maximum(ya, yb) + eps
            v = (yc >= lo) & (yc <= hi)
            dy = yb - ya
            flat = np.abs(dy) < 1e-12
            t = np.clip((yc - ya) / np.where(flat, 1.0, dy), 0.0, 1.0)
            x = xa + t * (xb - xa)
            l_ = np.where(flat, np.minimum(xa, xb), x)
            r_ = np.where(flat, np.maximum(xa, xb), x)
            xl = np.where(v, np.minimum(xl, l_), xl)
            xr = np.where(v, np.maximum(xr, r_), xr)
        c0 = np.maximum(np.ceil(xl - 0.5 - eps), 0)
        c1 = np.minimum(np.floor(xr - 0.5 + eps), W - 1)
        good = np.isfinite(xl) & (c1 >= c0)
        tp, row, c0, c1 = tp[good], row[good], c0[good].astype(np.int64), c1[good].astype(np.int64)
        nc = c1 - c0 + 1
        tot = int(nc.sum())
        if tot == 0:
            return
        self.fragments += tot
        pr = np.repeat(np.arange(len(tp)), nc)
        col = np.repeat(c0, nc) + (np.arange(tot) - np.repeat(np.cumsum(nc) - nc, nc))
        tf = tp[pr]
        z = A[tf] * (col + 0.5) + B[tf] * (row[pr] + 0.5) + C[tf]
        key = (self.quant(z) << ID_BITS) | ids[tf]
        np.minimum.at(self.buf, row[pr] * W + col, key)

    def add_cap(self, mask, A, B, C, cid):
        """mask (H,W) pixels covered by a cap with zkey = A x + B y + C."""
        jj, ii = np.nonzero(mask)
        if len(jj) == 0:
            return
        z = A * (ii + 0.5) + B * (jj + 0.5) + C
        key = (self.quant(z) << ID_BITS) | np.int64(cid)
        p = jj * self.W + ii
        self.buf[p] = np.minimum(self.buf[p], key)

    def ids(self):
        b = self.buf
        return np.where(b == EMPTY, -1, b & ID_MASK).reshape(self.H, self.W)


def polygon_spans(X0, Y0, X1, Y1, sid, W, H):
    """Even-odd scan conversion of a soup of closed-polygon edges, separately for each
    solid id (sid).  Returns (row, c0, c1, sid) spans of pixel columns [c0, c1]."""
    ylo = np.minimum(Y0, Y1)
    yhi = np.maximum(Y0, Y1)
    j0 = np.maximum(np.ceil(ylo - 0.5), 0).astype(np.int64)        # centre yc in [ylo, yhi)
    j1 = np.minimum(np.ceil(yhi - 0.5) - 1, H - 1).astype(np.int64)
    nr = np.maximum(j1 - j0 + 1, 0)
    if nr.sum() == 0:
        e = np.zeros(0, np.int64)
        return e, e, e, e
    seg = np.repeat(np.arange(len(X0)), nr)
    row = np.repeat(j0, nr) + (np.arange(nr.sum()) - np.repeat(np.cumsum(nr) - nr, nr))
    yc = row + 0.5
    t = (yc - Y0[seg]) / (Y1[seg] - Y0[seg])
    x = X0[seg] + t * (X1[seg] - X0[seg])
    s = sid[seg]
    o = np.lexsort((x, row, s))
    x, row, s = x[o], row[o], s[o]
    grp = np.ones(len(x), bool)
    grp[1:] = (row[1:] != row[:-1]) | (s[1:] != s[:-1])
    gstart = np.nonzero(grp)[0]
    gcount = np.diff(np.append(gstart, len(x)))
    rank = np.arange(len(x)) - np.repeat(gstart, gcount)
    # within a group use pairs (0,1),(2,3)...; drop a trailing odd crossing
    even = (rank % 2 == 0) & (rank + 1 < np.repeat(gcount, gcount))
    a = np.nonzero(even)[0]
    xa, xb = x[a], x[a + 1]
    c0 = np.maximum(np.ceil(xa - 0.5), 0).astype(np.int64)
    c1 = np.minimum(np.ceil(xb - 0.5) - 1, W - 1).astype(np.int64)
    ok = c1 >= c0
    return row[a][ok], c0[ok], c1[ok], s[a][ok]


def spans_to_pixels(row, c0, c1, W):
    n = c1 - c0 + 1
    tot = int(n.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    k = np.repeat(np.arange(len(row)), n)
    col = np.repeat(c0, n) + (np.arange(tot) - np.repeat(np.cumsum(n) - n, n))
    return row[k] * W + col, k


# =============================================================================
#  7.  Rendering
# =============================================================================

class Edges:
    """A batch of 3-D segments to draw, with attributes."""

    def __init__(self):
        self.A, self.B = [], []
        self.f1, self.f2, self.solid, self.kind = [], [], [], []

    def add(self, A, B, f1, f2, solid, kind):
        if len(A) == 0:
            return
        self.A.append(A); self.B.append(B)
        self.f1.append(np.asarray(f1, np.int64)); self.f2.append(np.asarray(f2, np.int64))
        self.solid.append(np.asarray(solid, np.int64))
        self.kind.append(np.full(len(A), kind, np.int8))

    def arrays(self):
        if not self.A:
            z = np.zeros((0, 3))
            e = np.zeros(0, np.int64)
            return z, z, e, e, e, np.zeros(0, np.int8)
        return (np.concatenate(self.A), np.concatenate(self.B), np.concatenate(self.f1),
                np.concatenate(self.f2), np.concatenate(self.solid), np.concatenate(self.kind))


K_CREASE, K_SIL, K_CUT, K_ALL = 0, 1, 2, 3


class DrawList:
    """Output primitives in canvas pixel coordinates (x right, y down)."""

    def __init__(self, W, H, dpi, bg):
        self.W, self.H, self.dpi, self.bg = W, H, dpi, bg
        self.items = []
        self.origin = np.zeros(2)       # accumulated translate(): renderer px -> canvas px

    def lines(self, polylines, rgb, lw_pt, dash=None, cap="round"):
        if polylines:
            self.items.append(("lines", polylines, rgb, lw_pt, dash, cap))

    def fill(self, rings, rgb, stroke=None, lw_pt=0.0, alpha=1.0):
        if rings:
            self.items.append(("fill", rings, rgb, stroke, lw_pt, alpha))

    def marker(self, kind, x, y, r_px, rgb):
        self.items.append(("marker", kind, x, y, r_px, rgb))

    def text(self, x, y, s, size_pt, rgb, anchor="start", font="Helvetica"):
        self.items.append(("text", x, y, s, size_pt, rgb, anchor, font))

    def image(self, rgb_u8, x=0.0, y=0.0, w=None, h=None):
        self.items.append(("image", rgb_u8, x, y, self.W if w is None else w, self.H if h is None else h))

    def translate(self, dx, dy):
        """Shift every item by (dx, dy) px (makes room for labels around the drawing)."""
        d = np.array([dx, dy], float)
        self.origin = self.origin + d
        out = []
        for it in self.items:
            k = it[0]
            if k in ("lines", "fill"):
                out.append((k, [np.asarray(p, float) + d for p in it[1]]) + it[2:])
            elif k == "marker":
                out.append(it[:2] + (it[2] + dx, it[3] + dy) + it[4:])
            elif k == "text":
                out.append((k, it[1] + dx, it[2] + dy) + it[3:])
            elif k == "image":
                out.append(it[:2] + (it[2] + dx, it[3] + dy) + it[4:])
        self.items = out

    def bbox(self):
        """(x0, y0, x1, y1) of everything drawn, in px (text from the Helvetica metrics)."""
        pt = self.dpi / 72.0
        xs, ys = [], []
        for it in self.items:
            k = it[0]
            if k in ("lines", "fill"):
                lw = 0.5 * (it[3] if k == "lines" else it[4]) * pt
                for p in it[1]:
                    p = np.asarray(p, float)
                    if len(p):
                        xs += [p[:, 0].min() - lw, p[:, 0].max() + lw]
                        ys += [p[:, 1].min() - lw, p[:, 1].max() + lw]
            elif k == "marker":
                xs += [it[2] - it[4], it[2] + it[4]]
                ys += [it[3] - it[4], it[3] + it[4]]
            elif k == "text":
                x0, y0, x1, y1 = text_box(it[1], it[2], it[3], it[4] * pt, it[6], bold=_is_bold(it[7]))
                xs += [x0, x1]
                ys += [y0, y1]
            elif k == "image":
                xs += [it[2], it[2] + it[4]]
                ys += [it[3], it[3] + it[5]]
        if not xs:
            return 0.0, 0.0, float(self.W), float(self.H)
        return min(xs), min(ys), max(xs), max(ys)


def _rgbhex(c):
    return "#%02x%02x%02x" % tuple(int(round(255 * min(1, max(0, v)))) for v in c)


def chain_2d(X0, Y0, X1, Y1, tol=1e-3):
    """Join 2-D segments with coincident end points into polylines."""
    n = len(X0)
    if n == 0:
        return []
    P = np.concatenate([np.stack([X0, Y0], 1), np.stack([X1, Y1], 1)])
    q = np.round(P / tol).astype(np.int64)
    _, node = np.unique(q, axis=0, return_inverse=True)
    node = node.ravel()
    na, nb = node[:n], node[n:]
    deg = np.bincount(np.concatenate([na, nb]))
    adj = defaultdict(list)
    for i in range(n):
        adj[na[i]].append(i)
        adj[nb[i]].append(i)
    used = np.zeros(n, bool)
    out = []
    order = list(np.nonzero(deg != 2)[0]) + list(range(len(deg)))
    for start in order:
        for i0 in adj[start]:
            if used[i0]:
                continue
            pts = []
            cur = start
            i = i0
            while i is not None:
                used[i] = True
                if na[i] == cur:
                    a, b, cur = (X0[i], Y0[i]), (X1[i], Y1[i]), nb[i]
                else:
                    a, b, cur = (X1[i], Y1[i]), (X0[i], Y0[i]), na[i]
                if not pts:
                    pts.append(a)
                pts.append(b)
                i = None
                if deg[cur] == 2:
                    for j in adj[cur]:
                        if not used[j]:
                            i = j
                            break
            out.append(np.array(pts))
    return out


def clip_segments_rect(x0, y0, x1, y1, rect):
    """Parts of 2-D segments inside rect = (xa, ya, xb, yb) (Liang-Barsky).
    Returns (x0, y0, x1, y1, src), src = index of the input segment of each piece."""
    xa, ya, xb, yb = rect
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = np.zeros(len(x0)), np.ones(len(x0))
    out = np.zeros(len(x0), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p_, q_ in ((-dx, x0 - xa), (dx, xb - x0), (-dy, y0 - ya), (dy, yb - y0)):
            out |= (p_ == 0) & (q_ < 0)
            r = q_ / p_
            t0 = np.where(p_ < 0, np.maximum(t0, r), t0)
            t1 = np.where(p_ > 0, np.minimum(t1, r), t1)
    src = np.nonzero(~out & (t0 < t1))[0]
    a, b = t0[src], t1[src]
    return (x0[src] + a * dx[src], y0[src] + a * dy[src], x0[src] + b * dx[src], y0[src] + b * dy[src], src)


def clip_polyline_rect(xy, rect):
    """Open polyline (n, 2) clipped to rect = (xa, ya, xb, yb): the list of its pieces inside."""
    if len(xy) < 2:
        return []
    xa, ya, xb, yb = rect
    if xy[:, 0].min() >= xa and xy[:, 0].max() <= xb and xy[:, 1].min() >= ya and xy[:, 1].max() <= yb:
        return [xy]
    x0, y0, x1, y1, src = clip_segments_rect(xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1], rect)
    if len(src) == 0:
        return []
    # a piece continues the previous one if it is the next segment and starts where that one ended
    brk = np.nonzero((np.diff(src) != 1) | (np.abs(x0[1:] - x1[:-1]) > 1e-9) | (np.abs(y0[1:] - y1[:-1]) > 1e-9))[0] + 1
    return [np.stack([np.append(x0[g], x1[g[-1]]), np.append(y0[g], y1[g[-1]])], 1)
            for g in np.split(np.arange(len(src)), brk)]


def clip_polygon_rect(P, rect):
    """Closed 2-D polygon (n, 2) clipped to rect = (xa, ya, xb, yb) (Sutherland-Hodgman); may be empty."""
    xa, ya, xb, yb = rect
    if len(P) == 0 or (P[:, 0].min() >= xa and P[:, 0].max() <= xb and P[:, 1].min() >= ya and P[:, 1].max() <= yb):
        return P
    if P[:, 0].max() < xa or P[:, 0].min() > xb or P[:, 1].max() < ya or P[:, 1].min() > yb:
        return P[:0]
    for ax, sign, lim in ((0, 1.0, xa), (0, -1.0, xb), (1, 1.0, ya), (1, -1.0, yb)):
        if len(P) == 0:
            break
        d = sign * (P[:, ax] - lim)
        res = []
        m = len(P)
        for i in range(m):
            j = (i + 1) % m
            if d[i] >= 0:
                res.append(P[i])
            if (d[i] >= 0) != (d[j] >= 0):
                res.append(P[i] + (d[i] / (d[i] - d[j])) * (P[j] - P[i]))
        P = np.array(res).reshape(-1, 2)
    return P


def merge_collinear(x0, y0, x1, y1, prio, style, tol=0.02):
    """Remove overdraw: where segments lie on the same straight line and overlap, keep one.

    Edges of touching volumes, a cut outline and the mesh edge under it, or edges that only coincide in
    the projection are otherwise drawn on top of each other: anti-aliased lines come out darker there,
    and whether the second copy is drawn can depend on rounding.  Segments are grouped by supporting
    line (foot point of the perpendicular from the origin and direction, both within `tol` px over the
    size of the drawing).  On each line the covered intervals are united; where several segments
    overlap, the one with the highest `prio` wins (then the first), and neighbouring pieces with the
    same `style` are joined.  End points keep the coordinates of the input points they come from, so
    polylines still chain at corners.

    Returns (x0, y0, x1, y1, src), src = index of the input segment each piece takes its style from."""
    n = len(x0)
    src = np.arange(n)
    if n < 2:
        return x0, y0, x1, y1, src
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    ok = L > 1e-9
    if not ok.all():
        x0, y0, x1, y1, dx, dy, L, src = x0[ok], y0[ok], x1[ok], y1[ok], dx[ok], dy[ok], L[ok], src[ok]
        n = len(x0)
        if n < 2:
            return x0, y0, x1, y1, src
    dx, dy = dx / L, dy / L
    flip = np.where(np.abs(dx) >= np.abs(dy), dx < 0, dy < 0)        # one orientation per line
    dx, dy = np.where(flip, -dx, dx), np.where(flip, -dy, dy)
    rho = y0 * dx - x0 * dy                                           # offset along the normal (-dy, dx)
    R = max(float(np.abs([x0, y0, x1, y1]).max()), 1.0)
    K = np.stack([-rho * dy, rho * dx, R * (dx * dx - dy * dy), R * 2.0 * dx * dy], 1)
    lab = np.zeros(n, np.int64)
    for c in range(4):                                                # single linkage, one coordinate at a time
        order = np.lexsort((K[:, c], lab))
        ls, ks = lab[order], K[order, c]
        new = np.ones(n, bool)
        new[1:] = (ls[1:] != ls[:-1]) | (ks[1:] - ks[:-1] > tol)
        lab[order] = np.cumsum(new) - 1
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    starts = np.nonzero(np.r_[True, ls[1:] != ls[:-1]])[0]
    sizes = np.diff(np.r_[starts, n])
    single = sizes == 1
    keep = order[starts[single]]
    ox0, oy0, ox1, oy1, osrc = [x0[keep]], [y0[keep]], [x1[keep]], [y1[keep]], [src[keep]]
    px0, py0, px1, py1, psrc = [], [], [], [], []
    for st, m in zip(starts[~single], sizes[~single]):
        idx = np.sort(order[st:st + m])                               # input order: the first wins a tie
        rx, ry, ex, ey = x0[idx[0]], y0[idx[0]], dx[idx[0]], dy[idx[0]]
        ta = (x0[idx] - rx) * ex + (y0[idx] - ry) * ey
        tb = (x1[idx] - rx) * ex + (y1[idx] - ry) * ey
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        tt = np.concatenate([ta, tb])
        qx = np.concatenate([x0[idx], x1[idx]])
        qy = np.concatenate([y0[idx], y1[idx]])
        o2 = np.argsort(tt, kind="stable")
        tt, qx, qy = tt[o2], qx[o2], qy[o2]
        first = np.r_[True, np.diff(tt) > 1e-6]                       # break points, each with its input point
        bt, bx, by = tt[first], qx[first], qy[first]
        if len(bt) < 2:
            continue
        mid = 0.5 * (bt[:-1] + bt[1:])
        cover = (lo[:, None] < mid[None, :]) & (hi[:, None] > mid[None, :])
        score = np.where(cover, prio[src[idx]][:, None], -np.inf)
        win = np.argmax(score, 0)                                     # first of the best
        has = cover.any(0)
        sty = np.where(has, style[src[idx]][win], -1)
        # runs of covered elementary intervals with one style
        brk = np.r_[True, (sty[1:] != sty[:-1]), True]
        for a, b in zip(np.nonzero(brk[:-1])[0], np.nonzero(brk[1:])[0]):
            if sty[a] < 0:
                continue
            px0.append(bx[a]); py0.append(by[a]); px1.append(bx[b + 1]); py1.append(by[b + 1])
            psrc.append(src[idx[win[a]]])
    if px0:
        ox0.append(np.array(px0)); oy0.append(np.array(py0)); ox1.append(np.array(px1)); oy1.append(np.array(py1))
        osrc.append(np.array(psrc, np.int64))
    return (np.concatenate(ox0), np.concatenate(oy0), np.concatenate(ox1), np.concatenate(oy1),
            np.concatenate(osrc))


class Renderer:
    window = None                       # (x0, y0, x1, y1) px when the view is fitted to a chosen window

    def __init__(self, sc, opts):
        self.sc = sc
        self.o = opts
        with timed("mesh analysis (normals, edges, crease %.0f deg)" % opts.crease):
            self.mesh = Mesh(sc, opts.crease)

    # ------------------------------------------------------------------ selection
    def select_solids(self):
        o, sc = self.o, self.sc
        n = len(sc.solids)
        sel = np.ones(n, bool)
        transparent = np.zeros(n, bool)
        inc = [re.compile(x) for x in (o.include or [])]
        exc = [re.compile(x) for x in (o.exclude or [])]
        trn = [re.compile(x) for x in (o.transparent or [])]
        for s in sc.solids:
            key = s.path if o.match_path else s.name
            if inc and not any(r.search(key) for r in inc):
                sel[s.index] = False
            if exc and any(r.search(key) for r in exc):
                sel[s.index] = False
            if o.min_depth is not None and 0 <= s.depth < o.min_depth:
                sel[s.index] = False
            if o.max_depth is not None and s.depth > o.max_depth:
                sel[s.index] = False
            if o.exclude_color and any(np.allclose(s.color, c, atol=0.02) for c in o.exclude_color):
                sel[s.index] = False
            if o.only_color and not any(np.allclose(s.color, c, atol=0.02) for c in o.only_color):
                sel[s.index] = False
            if s.wire:
                if o.wireframe_volumes == "hide":
                    sel[s.index] = False
                else:
                    transparent[s.index] = True
            if trn and any(r.search(key) for r in trn):
                transparent[s.index] = True
        if o.mothers != "opaque":
            for s in sc.solids:
                if s.has_daughters and any(sel[d.index] for d in sc.solids if d.parent == s.index):
                    if o.mothers == "hide":
                        sel[s.index] = False
                    else:
                        transparent[s.index] = True
        return sel, transparent

    def resolve_trims(self, sel, quiet=False):
        """--trim REGEX=SPEC -> [(n, d, solid_mask)]: half spaces that cut only the matching volumes.
        'auto' keeps the z range of all other selected volumes (+- --trim-margin of its length)."""
        o, sc = self.o, self.sc
        trims = []
        if not getattr(o, "trim", None):
            return trims
        keys = [s.path if o.match_path else s.name for s in sc.solids]
        masks = [(np.array([bool(sel[i] and rx.search(k)) for i, k in enumerate(keys)], bool), terms)
                 for rx, terms in o.trim]
        anym = np.zeros(len(sc.solids), bool)
        for m, _ in masks:
            anym |= m
        rest = [s for s in sc.solids if sel[s.index] and not anym[s.index]]
        for m, terms in masks:
            if not m.any():
                continue
            for t in terms:
                if t == "auto":
                    if not rest:
                        continue
                    z0 = min(float(s.bmin[2]) for s in rest)
                    z1 = max(float(s.bmax[2]) for s in rest)
                    pad = o.trim_margin * (z1 - z0)
                    trims += [(np.array([0.0, 0.0, 1.0]), -(z0 - pad), m), (np.array([0.0, 0.0, -1.0]), z1 + pad, m)]
                else:
                    n, d = t
                    trims.append((n, d, m))
        for n, d, m in ([] if quiet else trims):
            log("  trim %d volumes: keep (%.3g, %.3g, %.3g).P + %.6g >= 0" % (int(m.sum()), n[0], n[1], n[2], d))
        return trims

    # ------------------------------------------------------------------ main entry
    def render(self, cam, planes, cap_planes, mode):
        o, sc, mesh = self.o, self.sc, self.mesh
        sel, transp = self.select_solids()
        self.sel, self.transp = sel, transp
        nsel = int(sel.sum())
        log("  %d of %d solids selected (%d non-occluding)" % (nsel, len(sc.solids),
                                                              int((sel & transp).sum())))
        trims = self.trims = self.resolve_trims(sel)
        V = snap_vertices(sc, sc.V, mesh.tri, planes, o.plane_snap, False)
        fsel = sel[sc.fsolid] if len(sc.fsolid) else np.zeros(0, bool)
        tri_sel = fsel[mesh.tri_face] if len(mesh.tri_face) else np.zeros(0, bool)
        planes = list(planes)
        near = cam.near_plane()
        all_planes = planes + ([near] if near else [])

        # ---- edges to draw (3-D)
        edges = Edges()
        with timed("feature + silhouette edges"):
            esel = sel[mesh.esolid] if len(mesh.esolid) else np.zeros(0, bool)
            facing = cam.facing(mesh.fnormal, V[sc.F[:, 0]]) if len(sc.F) else np.zeros(0)
            front = facing > 0
            f1, f2 = mesh.ef1, mesh.ef2
            has2 = f2 >= 0
            sil = has2 & (front[f1] != front[np.maximum(f2, 0)])
            if mode == "wireframe" and o.all_edges:
                draw = esel.copy()
            else:
                draw = esel & (mesh.efeature | (sil if not o.no_silhouettes else False))
            # cull edges whose faces all face away (hidden by their own closed solid)
            if mode in ("hlr", "surface"):
                closed = np.array([s.closed for s in sc.solids], bool)[mesh.esolid]
                opaque = ~transp[mesh.esolid]
                anyfront = front[f1] | (has2 & front[np.maximum(f2, 0)])
                draw &= ~(closed & opaque & ~anyfront)
            di = np.nonzero(draw)[0]
            kind = np.where(sil[di] & ~mesh.efeature[di], K_SIL, K_CREASE)
            A, B = V[mesh.e0[di]], V[mesh.e1[di]]
            A, B, ef1, ef2, es, ek = clip_segments(A, B, all_planes, f1[di], f2[di],
                                                   mesh.esolid[di], kind)
            if trims:
                A, B, es, ef1, ef2, ek = clip_segments_masked(A, B, es, trims, ef1, ef2, ek)
            for k in (K_CREASE, K_SIL):
                m = ek == k
                edges.add(A[m], B[m], ef1[m], ef2[m], es[m], k)

        # ---- triangles (clipped)
        with timed("clip triangles (%d cut planes)" % len(all_planes)):
            ti = np.nonzero(tri_sel)[0]
            P = V[mesh.tri[ti]]
            P, tface = clip_triangles(P, mesh.tri_face[ti], all_planes)
            if trims:
                P, tface = clip_triangles_masked(P, tface, sc.fsolid, trims)

        # ---- cut-plane sections (caps); trim planes cap only their own volumes
        cap_items = [(pl, None) for pl in cap_planes]
        if not o.no_caps:
            cap_items += [((n, d), m) for n, d, m in trims]
        self.cap_items = cap_items
        caps = []          # (plane_index, Q0, Q1, solid, face, k0, k1)
        with timed("cut-plane sections"):
            tsel_idx = np.nonzero(tri_sel)[0]
            tsel_sol = sc.fsolid[mesh.tri_face[tsel_idx]]
            noncap = [p for p in all_planes if not any(p is q for q in cap_planes)]
            for pi, (pl, cmask) in enumerate(cap_items):
                tset = tsel_idx if cmask is None else tsel_idx[cmask[tsel_sol]]
                Q0, Q1, ti_, k0, k1 = section_segments(V, mesh.tri[tset], pl)
                tri_i = tset[ti_]
                face = mesh.tri_face[tri_i]
                sol = sc.fsolid[face]
                caps.append((pi, Q0, Q1, sol, face, k0, k1))
                others = [p for j, (p, mm) in enumerate(cap_items) if j != pi and mm is None] + noncap
                C0, C1, cf, cs = clip_segments(Q0, Q1, others, face, sol)
                masked = [(p[0], p[1], mm) for j, (p, mm) in enumerate(cap_items) if j != pi and mm is not None]
                if masked:
                    C0, C1, cs, cf = clip_segments_masked(C0, C1, cs, masked, cf)
                edges.add(C0, C1, cf, np.full(len(cf), -1), cs, K_CUT)

        self.edges3d = edges
        self.tris3d = (P, tface)
        self.caps = caps
        return self._finish(cam, planes, cap_planes, mode)

    # ------------------------------------------------------------------ screen mapping
    def _fit(self, cam, pts_list):
        o = self.o
        W, H = o.width, o.height
        if o.mag is not None:
            # DAWN-compatible: magnification relative to the /BoundingBox fit, centred on target
            bb = self.sc.bbox if self.sc.bbox is not None else np.concatenate(
                [self.sc.V.min(0), self.sc.V.max(0)])
            corners = np.array([[bb[i], bb[j], bb[k]] for i in (0, 3) for j in (1, 4) for k in (2, 5)])
            x, y, _ = cam.view(corners)
            s = min(W / max(np.ptp(x), 1e-9), H / max(np.ptp(y), 1e-9)) * o.mag
            cx, cy = 0.0, 0.0
            return s, cx, cy, W, H
        xs, ys = [], []
        for P in pts_list:
            if len(P):
                x, y, _ = cam.view(P)
                xs.append(x); ys.append(y)
        if not xs:
            return 1.0, 0.0, 0.0, W, H
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        if o.fit_volumes:
            # the box around the named volumes instead of the whole content
            rx = re.compile(o.fit_volumes)
            boxes = [(v.bmin, v.bmax) for v in self.sc.solids
                     if v.bmin is not None and rx.search(v.path if o.match_path else v.name)]
            if boxes:
                lo, hi = np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)
                corners = np.array([[(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]]
                                    for i in (0, 1) for j in (0, 1) for k in (0, 1)])
                bx, by, _ = cam.view(corners)
                x0, x1, y0, y1 = bx.min(), bx.max(), by.min(), by.max()
            else:
                log("warning: --fit-volumes: no volume matches '%s'; fitting everything" % o.fit_volumes, 0)
        if o.fit_pad:
            x0, x1, y0, y1 = x0 - o.fit_pad, x1 + o.fit_pad, y0 - o.fit_pad, y1 + o.fit_pad
        if o.fit_box:
            x0, x1, y0, y1 = o.fit_box
        windowed = bool(o.fit_volumes or o.fit_box)
        m = o.margin * min(W, H)
        dx, dy = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
        s = min((W - 2 * m) / dx, (H - 2 * m) / dy)
        cy = 0.5 * (y0 + y1)
        if not o.no_crop:
            # the limiting side is the full size up to rounding: do not let 3000.0000000001 become 3001
            W = int(math.ceil(dx * s + 2 * m - 1e-6))
            H = int(math.ceil(dy * s + 2 * m - 1e-6))
            band = title_band_px(o)
            if band:
                # reserve a band above the drawing so the title never overlaps the geometry
                band = int(math.ceil(band))
                H += band
                cy += 0.5 * band / s   # move the drawing down by the band
        # a chosen window: lines and fills are clipped to it (sections and line modes), below the title band
        self.window = (0.0, float(title_band_px(o)) if not o.no_crop else 0.0, float(W), float(H)) if windowed else None
        return s, 0.5 * (x0 + x1), cy, W, H

    # ------------------------------------------------------------------ beam arrow
    def beam_arrow_points(self, zlo, zhi):
        """--beam-arrow: on the z axis, pointing downstream, ending just upstream of the drawing
        (tail, tip, upstream end of the drawing); returned for the auto-fit."""
        L = max(zhi - zlo, 1.0)
        zb = zlo - 0.04 * L
        za = zb - self.o.beam_arrow_length * L
        self.beam_arrow = np.array([[0.0, 0.0, za], [0.0, 0.0, zb], [0.0, 0.0, zlo]])
        return self.beam_arrow[:2]

    def project_beam_axis(self, to_px):
        """Beam axis points on the canvas, for place_beam_arrow() (drawn after the geometry)."""
        pts = getattr(self, "beam_arrow", None)
        if pts is not None:
            x, y, _ = to_px(pts)
            self.beam_axis_px = [np.array([x[i], y[i]]) for i in range(3)]

    def _finish(self, cam, planes, cap_planes, mode):
        o, sc, mesh = self.o, self.sc, self.mesh
        P, tface = self.tris3d
        A3, B3, ef1, ef2, es, ek = self.edges3d.arrays()
        fitpts = [P.reshape(-1, 3), A3, B3]
        if o.fit_traj:
            fitpts += [pl for _, pl in sc.polylines]
        if o.beam_arrow is not None and len(P):
            fitpts.append(self.beam_arrow_points(float(P[..., 2].min()), float(P[..., 2].max())))
        s, cx, cy, W, H = self._fit(cam, fitpts)
        self.scale_px_per_mm = s if cam.ortho else None
        log("  canvas %d x %d px, scale %.4g px/mm" % (W, H, s))

        def to_px(Pw):
            x, y, depth = cam.view(Pw)
            return (x - cx) * s + 0.5 * W, 0.5 * H - (y - cy) * s, cam.zkey(depth)

        self.to_px = to_px
        dl = DrawList(W, H, o.dpi, o.background)
        hs = o.hlr_scale
        Wd, Hd = int(round(W * hs)), int(round(H * hs))

        # ---- project triangles
        if len(P):
            X, Y, Z = to_px(P.reshape(-1, 3))
            X, Y, Z = X.reshape(-1, 3) * hs, Y.reshape(-1, 3) * hs, Z.reshape(-1, 3)
        else:
            X = Y = Z = np.zeros((0, 3))
        tsol = sc.fsolid[tface] if len(tface) else np.zeros(0, np.int64)
        opaque_t = ~self.transp[tsol] if len(tsol) else np.zeros(0, bool)
        # plane coefficients of zkey in depth-buffer pixel space
        det = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
        good = np.abs(det) > 1e-12
        dsafe = np.where(good, det, 1.0)
        zA = ((Z[:, 1] - Z[:, 0]) * (Y[:, 2] - Y[:, 0]) - (Z[:, 2] - Z[:, 0]) * (Y[:, 1] - Y[:, 0])) / dsafe
        zB = ((X[:, 1] - X[:, 0]) * (Z[:, 2] - Z[:, 0]) - (X[:, 2] - X[:, 0]) * (Z[:, 1] - Z[:, 0])) / dsafe
        zC = Z[:, 0] - zA * X[:, 0] - zB * Y[:, 0]
        # screen-space orientation: front facing if counter-clockwise on screen (y down -> det<0)
        closed = np.array([sl.closed for sl in sc.solids], bool)
        frontfacing = det < 0
        rast = good & opaque_t
        if not o.no_cull:
            rast &= frontfacing | ~closed[tsol]
        self.n_tri = len(P)
        need_depth = mode in ("hlr", "surface")
        db = None
        if need_depth:
            zall = Z[rast]
            zlo = float(zall.min()) if zall.size else 0.0
            zhi = float(zall.max()) if zall.size else 1.0
            pad = 1e-3 * (zhi - zlo + 1e-9)
            zlo, zhi = zlo - pad, zhi + pad
            db = DepthBuffer(Wd, Hd, zlo, zhi)
            M = len(P)
            if M + len(self.cap_items) > ID_MASK:
                raise RuntimeError("too many triangles for the depth buffer id field")
            ids_all = np.arange(M, dtype=np.int64)
            with timed("rasterise %d triangles into %dx%d depth buffer" % (int(rast.sum()), Wd, Hd)):
                ri = np.nonzero(rast)[0]
                db.add_triangles(X[ri], Y[ri], zA[ri], zB[ri], zC[ri], ids_all[ri])
            log("  depth buffer: %d triangles, %.3g fragments" % (int(rast.sum()), db.fragments), 2)
            # caps
            capA, capB, capC = [], [], []
            self.cap_solid_maps = []
            with timed("rasterise %d cut-plane caps" % len(self.cap_items)):
                for (pi, Q0, Q1, sol, face, k0, k1) in self.caps:
                    cA, cB, cC, smap = self._cap_raster(cam, to_px, hs, Wd, Hd, self.cap_items, pi,
                                                        planes, Q0, Q1, sol)
                    capA.append(cA); capB.append(cB); capC.append(cC)
                    self.cap_solid_maps.append(smap)
                    if smap is not None:
                        db.add_cap(smap >= 0, cA, cB, cC, M + pi)
            self.db = db
            self.zA = np.concatenate([zA, np.array(capA, float)])
            self.zB = np.concatenate([zB, np.array(capB, float)])
            self.zC = np.concatenate([zC, np.array(capC, float)])
            self.zface = np.concatenate([tface, np.full(len(capA), -2, np.int64)])
            self.zgrad = np.hypot(self.zA, self.zB)
            self.eps_z = 1e-7 * (zhi - zlo)

        # ---- surface image / polygons
        if mode == "surface":
            self._surface(dl, cam, X, Y, Z, tface, tsol, rast, hs)

        # ---- cap fills (hlr) / section look
        if mode in ("hlr", "surface", "wireframe") and o.cap_fill != "none" and self.caps:
            self._cap_fills(dl, cam, self.cap_items, planes, to_px)

        # ---- project edges, hidden-line test
        X0, Y0, Z0 = to_px(A3)
        X1, Y1, Z1 = to_px(B3)
        if need_depth and len(A3):
            with timed("hidden-line test of %d edges" % len(A3)):
                segs = self._visible_runs(X0 * hs, Y0 * hs, Z0, X1 * hs, Y1 * hs, Z1, ef1, ef2)
            seg_e, t0, t1 = segs
        else:
            seg_e = np.arange(len(A3))
            t0 = np.zeros(len(A3))
            t1 = np.ones(len(A3))
        sx0 = X0[seg_e] + t0 * (X1[seg_e] - X0[seg_e])
        sy0 = Y0[seg_e] + t0 * (Y1[seg_e] - Y0[seg_e])
        sx1 = X0[seg_e] + t1 * (X1[seg_e] - X0[seg_e])
        sy1 = Y0[seg_e] + t1 * (Y1[seg_e] - Y0[seg_e])
        if self.window is not None and len(seg_e):
            sx0, sy0, sx1, sy1, src = clip_segments_rect(sx0, sy0, sx1, sy1, self.window)
            seg_e = seg_e[src]
        self.n_vis_segments = len(seg_e)
        self.vis_segs = (sx0, sy0, sx1, sy1, es[seg_e])      # what is drawn: label anchors
        with timed("style + chain %d visible segments" % len(seg_e)):
            self._emit_edges(dl, sx0, sy0, sx1, sy1, es[seg_e], ek[seg_e])

        # ---- overlays
        self._overlays(dl, cam, to_px, s, planes)
        self.project_beam_axis(to_px)
        self._decorations(dl, cam, s)
        return dl

    # ------------------------------------------------------------------ caps
    def _cap_raster(self, cam, to_px, hs, Wd, Hd, cap_items, pi, planes, Q0, Q1, sol):
        n, d = cap_items[pi][0]
        # zkey plane of the cut plane in depth-buffer pixel coords, from 3 points on it
        p0 = -d * n
        t1 = _unit(np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0]))
        t2 = np.cross(n, t1)
        L = 1000.0
        pts = np.array([p0, p0 + L * t1, p0 + L * t2])
        X, Y, Z = to_px(pts)
        X, Y = X * hs, Y * hs
        det = (X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0])
        if abs(det) < 1e-12:            # plane seen edge-on: no visible cap area
            return 0.0, 0.0, 0.0, None
        A = ((Z[1] - Z[0]) * (Y[2] - Y[0]) - (Z[2] - Z[0]) * (Y[1] - Y[0])) / det
        B = ((X[1] - X[0]) * (Z[2] - Z[0]) - (X[2] - X[0]) * (Z[1] - Z[0])) / det
        C = Z[0] - A * X[0] - B * Y[0]
        opaque = ~self.transp[sol]
        if not opaque.any():
            return A, B, C, None
        x0, y0, _ = to_px(Q0[opaque])
        x1, y1, _ = to_px(Q1[opaque])
        row, c0, c1, ss = polygon_spans(x0 * hs, y0 * hs, x1 * hs, y1 * hs, sol[opaque], Wd, Hd)
        smap = np.full(Wd * Hd, -1, np.int64)
        pix, k = spans_to_pixels(row, c0, c1, Wd)
        if len(pix):
            np.maximum.at(smap, pix, ss[k])
        smap = smap.reshape(Hd, Wd)
        # other cut planes: drop cap pixels outside the kept region (trim planes: their volumes only)
        others = [it for j, it in enumerate(cap_items) if j != pi]
        if others and (smap >= 0).any():
            jj, ii = np.nonzero(smap >= 0)
            Pw = self._unproject_to_plane(cam, (ii + 0.5) / hs, (jj + 0.5) / hs, n, d)
            sid = smap[jj, ii]
            ok = np.ones(len(jj), bool)
            for (m_, dd), mm in others:
                ok &= (Pw @ m_ + dd >= 0) if mm is None else ((Pw @ m_ + dd >= 0) | ~mm[sid])
            smap[jj[~ok], ii[~ok]] = -1
        return A, B, C, smap

    def _unproject_to_plane(self, cam, xs, ys, n, d):
        s, cx, cy, W, H = self._fitinfo
        xv = (xs - 0.5 * W) / s + cx
        yv = (0.5 * H - ys) / s + cy
        if cam.ortho:
            O = cam.T[None, :] + xv[:, None] * cam.r + yv[:, None] * cam.u
            D = -cam.e
        else:
            C = cam.position
            tgt = cam.T[None, :] + xv[:, None] * cam.r + yv[:, None] * cam.u
            O = np.broadcast_to(C, tgt.shape)
            D = tgt - C
        if D.ndim == 1:
            D = np.broadcast_to(D, O.shape)
        den = D @ n
        t = -(O @ n + d) / np.where(np.abs(den) < 1e-12, 1e-12, den)
        return O + t[:, None] * D

    def _cap_fills(self, dl, cam, cap_items, planes, to_px):
        o, sc = self.o, self.sc
        by_solid = defaultdict(list)
        for (pi, Q0, Q1, sol, face, k0, k1) in self.caps:
            for sid in np.unique(sol):
                if self.transp[sid]:
                    continue
                others = [p for j, (p, mm) in enumerate(cap_items) if j != pi and (mm is None or mm[sid])]
                m = sol == sid
                for pts, closed in chain_segments(k0[m], k1[m], Q0[m], Q1[m]):
                    if not closed:
                        continue
                    pts = clip_polygon(pts, others)
                    if len(pts) >= 3:
                        x, y, _ = to_px(pts)
                        by_solid[sid].append(np.stack([x, y], 1))
        for sid in sorted(by_solid):
            col = self._fill_color(sc.solids[sid].color)
            dl.fill(by_solid[sid], col)

    def _fill_color(self, c):
        o = self.o
        if o.cap_fill == "white":
            return o.background or (1, 1, 1)
        if o.cap_fill not in ("color", "colour"):
            fc = parse_color(o.cap_fill)
            return fc
        w = o.fill_lighten
        return tuple(w + (1 - w) * v for v in c)

    # ------------------------------------------------------------------ HLR core
    def _visible_runs(self, x0, y0, z0, x1, y1, z1, f1, f2, step=None, max_samples=8_000_000):
        o = self.o
        step = step or o.sample_step
        L = np.hypot(x1 - x0, y1 - y0)
        ns = np.maximum(2, np.ceil(L / step).astype(np.int64) + 1)
        out_e, out_t0, out_t1 = [], [], []
        cum = np.cumsum(ns)
        nb = int(cum[-1] // max_samples) + 1 if len(cum) else 0
        bounds = np.searchsorted(cum, np.linspace(0, cum[-1], nb + 1)[1:-1]) if nb > 1 else []
        for part in np.split(np.arange(len(ns)), bounds):
            if len(part) == 0:
                continue
            e, a, b = self._runs_part(part, ns[part], x0, y0, z0, x1, y1, z1, f1, f2)
            out_e.append(e); out_t0.append(a); out_t1.append(b)
        if not out_e:
            z = np.zeros(0)
            return np.zeros(0, np.int64), z, z
        return np.concatenate(out_e), np.concatenate(out_t0), np.concatenate(out_t1)

    def _runs_part(self, part, ns, x0, y0, z0, x1, y1, z1, f1, f2):
        db = self.db
        W, H = db.W, db.H
        tot = int(ns.sum())
        e = np.repeat(part, ns)
        first = np.repeat(np.cumsum(ns) - ns, ns)
        k = np.arange(tot) - first
        nn = np.repeat(ns, ns)
        t = k / (nn - 1.0)
        x = x0[e] + t * (x1[e] - x0[e])
        y = y0[e] + t * (y1[e] - y0[e])
        z = z0[e] + t * (z1[e] - z0[e])
        ix = np.floor(x - 0.5).astype(np.int64)
        iy = np.floor(y - 0.5).astype(np.int64)
        vis = np.zeros(tot, bool)
        inside = (ix >= -1) & (ix < W) & (iy >= -1) & (iy < H)
        vis |= ~inside
        a1, a2 = f1[e], f2[e]
        tol_k = self.o.slope_tol
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            px = np.clip(ix + dx, 0, W - 1)
            py = np.clip(iy + dy, 0, H - 1)
            key = db.buf[py * W + px]
            empty = key == EMPTY
            idv = np.where(empty, 0, key & ID_MASK)
            fz = self.zface[idv]
            own = (fz == a1) | (fz == a2)
            zs = self.zA[idv] * x + self.zB[idv] * y + self.zC[idv]
            tol = self.eps_z + tol_k * self.zgrad[idv]
            vis |= empty | own | (z <= zs + tol)
        # morphological clean-up along each edge: close 1-sample gaps, drop lone samples
        same_prev = np.zeros(tot, bool)
        same_prev[1:] = e[1:] == e[:-1]
        same_next = np.zeros(tot, bool)
        same_next[:-1] = e[:-1] == e[1:]
        prev = np.zeros(tot, bool)
        prev[1:] = vis[:-1]
        nxt = np.zeros(tot, bool)
        nxt[:-1] = vis[1:]
        prev &= same_prev
        nxt &= same_next
        vis = vis | (prev & nxt)
        prev = np.zeros(tot, bool)
        prev[1:] = vis[:-1]
        nxt = np.zeros(tot, bool)
        nxt[:-1] = vis[1:]
        lone = vis & ~(prev & same_prev) & ~(nxt & same_next) & (nn > 4)
        vis &= ~lone
        # runs
        pv = np.zeros(tot, bool)
        pv[1:] = vis[:-1]
        pv &= same_prev
        nv_ = np.zeros(tot, bool)
        nv_[:-1] = vis[1:]
        nv_ &= same_next
        st = np.nonzero(vis & ~pv)[0]
        en = np.nonzero(vis & ~nv_)[0]
        # extend run ends half a sample towards the hidden neighbour
        ts = np.where(k[st] > 0, (k[st] - 0.5) / (nn[st] - 1.0), 0.0)
        te = np.where(k[en] < nn[en] - 1, (k[en] + 0.5) / (nn[en] - 1.0), 1.0)
        keep = te > ts + 1e-12
        return e[st][keep], ts[keep], te[keep]

    # ------------------------------------------------------------------ styling
    def _emit_edges(self, dl, x0, y0, x1, y1, solid, kind):
        o, sc = self.o, self.sc
        depth = np.array([max(s.depth, 0) for s in sc.solids], float)
        dmin = depth[self.sel].min() if self.sel.any() else 0.0
        groups = defaultdict(list)
        lw_base = np.where(kind == K_CUT, o.lw_cut, np.where(kind == K_SIL, o.lw_outline, o.lw))
        if o.depth_lw_factor != 1.0:
            lw_base = lw_base * o.depth_lw_factor ** (depth[solid] - dmin)
        lwq = np.round(lw_base, 3)
        if o.color_lines:
            cols = np.array([sc.solids[i].color for i in range(len(sc.solids))])
            cols = cols * o.line_color_scale
            ci = solid
        else:
            cols = np.array([o.line_color])
            ci = np.zeros(len(solid), np.int64)
        key = np.stack([ci, np.round(lwq * 1000).astype(np.int64)], 1) if len(ci) else np.zeros((0, 2), np.int64)
        if len(key) == 0:
            return
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        if not o.keep_overdraw:
            with timed("merge overlapping lines"):
                x0, y0, x1, y1, src = merge_collinear(x0, y0, x1, y1, lwq, inv)
            self.n_drawn_segments = len(src)
            inv = inv[src]
        for g in range(len(uk)):
            m = inv == g
            pls = chain_2d(x0[m], y0[m], x1[m], y1[m])
            dl.lines(pls, tuple(cols[uk[g][0]]), uk[g][1] / 1000.0)

    # ------------------------------------------------------------------ surface mode
    def _surface(self, dl, cam, X, Y, Z, tface, tsol, rast, hs):
        o, sc, mesh = self.o, self.sc, self.mesh
        nf = len(sc.F)
        Lw = _unit(o.light[0] * cam.r + o.light[1] * cam.u + o.light[2] * cam.e)
        nrm = mesh.fnormal
        towards = cam.facing(nrm, sc.V[sc.F[:, 0]]) if nf else np.zeros(0)
        nn = np.where((towards < 0)[:, None], -nrm, nrm)
        lam = np.clip(nn @ Lw, 0, 1)
        base = np.array([s.color for s in sc.solids]) if sc.solids else np.zeros((0, 3))
        fcol = base[sc.fsolid] * (o.ambient + (1 - o.ambient) * lam)[:, None]
        fcol = np.clip(fcol, 0, 1)
        M = len(tface)
        ncap = len(self.caps)
        if o.surface_vector:
            ids = self.db.ids()
            vis_ids = np.unique(ids[(ids >= 0) & (ids < M)])
            zc = Z[vis_ids].mean(1)
            order = vis_ids[np.argsort(-zc)]
            polys = [np.stack([X[i] / hs, Y[i] / hs], 1) for i in order]
            cols = fcol[tface[order]]
            # group consecutive equal colours to keep the files small
            for i, pl in enumerate(polys):
                dl.items.append(("fill", [pl], tuple(cols[i]), tuple(cols[i]), 0.25, 1.0))
        else:
            ids = self.db.ids()
            img = np.empty(ids.shape + (3,), float)
            img[:] = o.background or (1, 1, 1)
            m = (ids >= 0) & (ids < M)
            img[m] = fcol[tface[ids[m]]]
            for c in range(ncap):
                mc = ids == M + c
                smap = self.cap_solid_maps[c]
                if mc.any() and smap is not None:
                    sid = smap[mc]
                    img[mc] = base[sid] * (o.ambient + (1 - o.ambient) * abs(float(
                        self._cap_normal(c) @ Lw)))
            if hs != 1:
                # box-filter down to the canvas resolution
                f = int(round(hs))
                if f > 1 and img.shape[0] % f == 0 and img.shape[1] % f == 0:
                    img = img.reshape(img.shape[0] // f, f, img.shape[1] // f, f, 3).mean((1, 3))
            dl.image((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))

    def _cap_normal(self, c):
        return self.cap_items[c][0][0]

    # ------------------------------------------------------------------ overlays
    def _overlays(self, dl, cam, to_px, s, planes):
        o, sc = self.o, self.sc
        dpi = o.dpi
        pt2px = dpi / 72.0
        near = cam.near_plane()
        pl_planes = ([near] if near else []) + (list(planes) if o.clip_traj else [])
        if not o.no_traj and sc.polylines:
            groups = defaultdict(list)
            for rgb, pts in sc.polylines:
                if len(pts) < 2:
                    continue
                A, B = pts[:-1], pts[1:]
                if pl_planes:
                    A, B = clip_segments(A, B, pl_planes)
                if len(A) == 0:
                    continue
                xa, ya, za = to_px(A)
                xb, yb, zb = to_px(B)
                if o.traj_hlr and hasattr(self, "db") and self.db is not None:
                    hs = o.hlr_scale
                    m1 = np.full(len(A), -9, np.int64)
                    e, t0, t1 = self._visible_runs(xa * hs, ya * hs, za, xb * hs, yb * hs, zb, m1, m1)
                    xa2 = xa[e] + t0 * (xb[e] - xa[e]); ya2 = ya[e] + t0 * (yb[e] - ya[e])
                    xb2 = xa[e] + t1 * (xb[e] - xa[e]); yb2 = ya[e] + t1 * (yb[e] - ya[e])
                    xa, ya, xb, yb = xa2, ya2, xb2, yb2
                col = o.traj_color if o.traj_color is not None else rgb
                groups[tuple(col)].append((xa, ya, xb, yb))
            for col, parts in groups.items():
                xa = np.concatenate([p[0] for p in parts]); ya = np.concatenate([p[1] for p in parts])
                xb = np.concatenate([p[2] for p in parts]); yb = np.concatenate([p[3] for p in parts])
                dl.lines(chain_2d(xa, ya, xb, yb), col, o.traj_lw)
        if not o.no_hits:
            for kind, rgb, xyz, size, screen in sc.markers:
                x, y, z = to_px(xyz[None, :])
                if any(pl[0] @ xyz + pl[1] < 0 for pl in pl_planes):
                    continue
                r = size * pt2px * o.marker_scale if screen else size * s * o.marker_scale
                dl.marker(kind, float(x[0]), float(y[0]), max(r, 0.5), rgb)
        if not o.no_text:
            for rgb, xyz, size, dx, dy, txt, screen, font in sc.texts:
                x, y, z = to_px(xyz[None, :])
                k = pt2px if screen else s
                if not txt.isprintable() or "�" in txt:
                    continue      # Geant4 11 G4Text dangling-pointer garbage
                dl.text(float(x[0]) + dx * k, float(y[0]) - dy * k, txt, size * (1 if screen else s / pt2px),
                        rgb if not o.mono_text else (0, 0, 0))
            for rgb, xmm, ymm, size, txt, font in sc.texts2d:
                if "�" in txt:
                    continue
                dl.text(xmm / 25.4 * dpi, dl.H - ymm / 25.4 * dpi, txt, size, rgb)

    def _decorations(self, dl, cam, s):
        """Scale bar and axis triad (in the drawing frame; the title is added last, draw_title)."""
        o = self.o
        pt2px = o.dpi / 72.0
        W, H = dl.W, dl.H
        fs = o.font_size
        if o.scale_bar is not None and cam.ortho:
            L = o.scale_bar
            if L <= 0:      # auto: 1-2-5 sequence, ~15 % of the width
                target = 0.15 * W / s
                p10 = 10 ** math.floor(math.log10(target))
                L = max(m * p10 for m in (1, 2, 5) if m * p10 <= target * 1.0001) if target >= p10 else p10
            Lpx = L * s
            x0 = o.margin * min(W, H) * 0.6 + 4 * pt2px
            y0 = H - 4 * pt2px - 0.6 * o.margin * min(W, H)
            tick = 3 * pt2px
            dl.lines([np.array([[x0, y0 - tick], [x0, y0], [x0 + Lpx, y0], [x0 + Lpx, y0 - tick]])],
                     (0, 0, 0), 1.0, cap="butt")
            lab = ("%g m" % (L / 1000.0)) if L >= 1000 else ("%g cm" % (L / 10.0)) if L >= 10 else "%g mm" % L
            dl.text(x0 + Lpx / 2, y0 - tick - 2 * pt2px, lab, fs, (0, 0, 0), "middle")
        if o.axes:
            ax = np.eye(3)
            L = 18 * pt2px
            ox = W - 30 * pt2px - 0.6 * o.margin * min(W, H)
            oy = H - 30 * pt2px - 0.6 * o.margin * min(W, H)
            for i, name in enumerate("xyz"):
                vx, vy = ax[i] @ cam.r, -(ax[i] @ cam.u)
                if math.hypot(vx, vy) < 0.15:
                    dl.marker("circle", ox, oy, 2.5 * pt2px, (0, 0, 0))
                    dl.text(ox + 4 * pt2px, oy - 4 * pt2px, name + ("⊙" if ax[i] @ cam.e > 0 else "⊗"),
                            fs, (0, 0, 0))
                    continue
                ex, ey = ox + L * vx, oy + L * vy
                dl.lines([np.array([[ox, oy], [ex, ey]])], (0, 0, 0), 0.8)
                ang = math.atan2(vy, vx)
                hl = 4 * pt2px
                for da in (2.6, -2.6):
                    dl.lines([np.array([[ex, ey], [ex + hl * math.cos(ang + da), ey + hl * math.sin(ang + da)]])],
                             (0, 0, 0), 0.8)
                dl.text(ex + 6 * pt2px * vx, ey + 6 * pt2px * vy + 0.35 * fs * pt2px, name, fs, (0, 0, 0), "middle")


# =============================================================================
#  7b.  Magnetic field under a section (--field)
# =============================================================================
#  Two sources:
#    * a GEMC 2-D map (rows R Z Br Bz behind an <mfield> header, axisymmetric about the z axis of the
#      hall frame), the format of the SoLID solenoid maps;
#    * field.npz written by solid_field_dump: the total field of a configuration (all its DD4hep field
#      overlays, e.g. solenoid plus target magnet) sampled on the section plane.
#  Under the outlines of a section the field is drawn as
#    * |B| in colour bands (one hue, light to dark, edges 1-3-10-...), for any section plane;
#    * field lines.  Where the in-plane field is axisymmetric about the z axis (a 2-D map, or a sampled
#      field in a plane through the axis whose two halves mirror each other) they are contours of the
#      flux function psi(r, z) = integral_0^r Bz r' dr' in equal steps, so the same flux runs between
#      neighbours.  A 1 cm grid cannot be integrated through iron (B jumps at the faces), so psi is
#      carried from the axis through the air only; the iron is where the solids matching --field-iron
#      are cut.  Otherwise the lines follow the in-plane field direction at an even spacing: they show
#      the direction only, the bands show the strength;
#    * unit arrows of the in-plane field direction on a coarse grid, in air where |B| >= 1 G; where the
#      field points mostly out of the plane, a dot in a circle (towards the viewer) or a cross in a
#      circle (away from the viewer) instead.
#  The iron itself is filled grey: the field is not meaningful inside it.

FIELD_BANDS_G = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
FIELD_RAMP = ["#eef5fe", "#dcebfc", "#c3dcfa", "#a5caf6", "#86b6f1", "#689fea", "#4b88e0", "#3272d0", "#1f5cb5",
              "#143f85"]
FIELD_MIN_BANDS = 9                 # the key always reaches 3 T; the 3-10 T band appears when it is used
FIELD_LINE_RGB = (0.05, 0.16, 0.33)
FIELD_ARROW_RGB = (0.32, 0.32, 0.31)
FIELD_IRON_RGB = (0.80, 0.80, 0.78)


def _hexrgb(h):
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))


def field_text(g):
    return ("%g T" % (g / 1e4)) if g >= 1000 else "%g G" % g


class AxialField:
    """In-plane field that is axisymmetric about the z axis, on a regular grid: r, z (cm), Br, Bz (gauss)
    shaped (nR, nZ)."""

    def __init__(self, r, z, Br, Bz):
        self.r, self.z, self.Br, self.Bz = r, z, Br, Bz

    def at(self, r_cm, z_cm):
        """(Br, Bz, inside) by bilinear interpolation"""
        r, z = self.r, self.z
        inside = (r_cm >= r[0]) & (r_cm <= r[-1]) & (z_cm >= z[0]) & (z_cm <= z[-1])
        fi = np.clip((r_cm - r[0]) / (r[1] - r[0]), 0, len(r) - 1.000001)
        fj = np.clip((z_cm - z[0]) / (z[1] - z[0]), 0, len(z) - 1.000001)
        i, j = fi.astype(np.int64), fj.astype(np.int64)
        a, b = fi - i, fj - j
        out = []
        for F in (self.Br, self.Bz):
            out.append(F[i, j] * (1 - a) * (1 - b) + F[i + 1, j] * a * (1 - b) + F[i, j + 1] * (1 - a) * b
                       + F[i + 1, j + 1] * a * b)
        return out[0], out[1], inside

    def iron_mask(self, polys, grow=1):
        """map nodes inside the (r, z) polygons (cm), grown by `grow` nodes"""
        mask = np.zeros(self.Br.shape, bool)
        for P in polys:
            i0, i1 = np.searchsorted(self.r, [P[:, 0].min(), P[:, 0].max()])
            j0, j1 = np.searchsorted(self.z, [P[:, 1].min(), P[:, 1].max()])
            if i1 <= i0 or j1 <= j0:
                continue
            X, Y = np.meshgrid(self.r[i0:i1], self.z[j0:j1], indexing="ij")
            ins = np.zeros(X.shape, bool)
            Q = np.roll(P, -1, axis=0)
            for (x1, y1), (x2, y2) in zip(P, Q):
                if y1 == y2:
                    continue
                cond = (y1 > Y) != (y2 > Y)
                ins ^= cond & (X < x1 + (Y - y1) * (x2 - x1) / (y2 - y1))
            mask[i0:i1, j0:j1] |= ins
        for _ in range(grow):
            g = mask.copy()
            g[1:] |= mask[:-1]
            g[:-1] |= mask[1:]
            g[:, 1:] |= mask[:, :-1]
            g[:, :-1] |= mask[:, 1:]
            mask = g
        return mask

    def flux(self, iron):
        """psi (G cm2) carried from the axis through the air; NaN in iron and in unreachable pockets"""
        r = self.r[:, None]
        fr, fz = self.Bz * r, -self.Br * r
        dr = 0.5 * (fr[1:] + fr[:-1]) * np.diff(self.r)[:, None]
        dz = 0.5 * (fz[:, 1:] + fz[:, :-1]) * np.diff(self.z)[None, :]
        air = ~iron
        psi = np.full(fr.shape, np.nan)
        psi[0] = np.where(air[0], 0.0, np.nan)
        nr, nz = psi.shape
        for _ in range(12):
            before = int(np.isfinite(psi).sum())
            for i in range(1, nr):
                new = air[i] & np.isnan(psi[i]) & np.isfinite(psi[i - 1])
                psi[i, new] = psi[i - 1, new] + dr[i - 1, new]
            for j in range(1, nz):
                new = air[:, j] & np.isnan(psi[:, j]) & np.isfinite(psi[:, j - 1])
                psi[new, j] = psi[new, j - 1] + dz[new, j - 1]
            for j in range(nz - 2, -1, -1):
                new = air[:, j] & np.isnan(psi[:, j]) & np.isfinite(psi[:, j + 1])
                psi[new, j] = psi[new, j + 1] - dz[new, j]
            for i in range(nr - 2, -1, -1):
                new = air[i] & np.isnan(psi[i]) & np.isfinite(psi[i + 1])
                psi[i, new] = psi[i + 1, new] - dr[i, new]
            if int(np.isfinite(psi).sum()) == before:
                break
        return psi


class FieldMap(AxialField):
    """GEMC 2-D map: r, z (cm), Br, Bz (gauss) shaped (nR, nZ)."""

    def __init__(self, path, scale=1.0):
        head, nhead = [], 0
        with open(path) as fh:
            for line in fh:
                if not head and not line.strip():
                    nhead += 1
                    continue
                if not head and "<mfield>" not in line:
                    break
                head.append(line)
                nhead += 1
                if "</mfield>" in line:
                    break
        text = "".join(head)
        if "<mfield>" not in text:
            raise SystemExit("%s: no <mfield> header (not a GEMC field map)" % path)
        if not re.search(r'<symmetry\s+type="cylindrical-z"', text):
            raise SystemExit("%s: only cylindrical-z (2-D) maps can be drawn" % path)

        def axis(which):
            m = re.search(r"<%s\b([^>]*)>" % which, text)
            at = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
            k = {"mm": 0.1, "cm": 1.0, "m": 100.0}[at.get("units", "cm")]
            return int(at["npoints"]), float(at["min"]) * k, float(at["max"]) * k

        nr, r0, r1 = axis("first")
        nz, z0, z1 = axis("second")
        m = re.search(r'<field\s+unit="([^"]*)"', text)
        fac = {"gauss": 1.0, "g": 1.0, "kilogauss": 1e3, "kg": 1e3, "tesla": 1e4, "t": 1e4}[
            m.group(1).lower() if m else "gauss"]
        m = re.search(r'<description\s+name="([^"]*)"', text)
        self.name = m.group(1) if m else os.path.basename(path)
        data = np.loadtxt(path, skiprows=nhead, usecols=(2, 3))
        if len(data) != nr * nz:
            raise SystemExit("%s: %d rows, the header says %d x %d" % (path, len(data), nr, nz))
        AxialField.__init__(self, np.linspace(r0, r1, nr), np.linspace(z0, z1, nz),
                            data[:, 0].reshape(nr, nz) * fac * scale, data[:, 1].reshape(nr, nz) * fac * scale)

    def check_plane(self, n, d):
        pass                                        # axisymmetric: defined on every plane

    def vector(self, P):
        """world points (N, 3) mm -> field (N, 3) gauss in the hall frame, inside the map (N,)"""
        rr = np.hypot(P[:, 0], P[:, 1])
        br, bz, inside = self.at(rr / 10.0, P[:, 2] / 10.0)
        k = np.where(rr > 1e-9, br / np.maximum(rr, 1e-9), 0.0)
        return np.stack([k * P[:, 0], k * P[:, 1], bz], 1), inside

    def axial(self, n, d):
        """the (r, z) field for flux lines, if the plane holds the z axis"""
        return self if abs(n[2]) < 1e-6 and abs(d) < 1e-6 else None


class SampledField:
    """field.npz of solid_field_dump: the field (hall-frame components) on a regular grid in one plane."""

    def __init__(self, path, scale=1.0):
        with np.load(path, allow_pickle=False) as f:
            need = ("u", "v", "B", "origin", "u_dir", "v_dir", "normal")
            if any(k not in f for k in need):
                raise SystemExit("%s: not a field file of solid_field_dump (arrays %s)" % (path, ", ".join(need)))
            self.u, self.v = np.asarray(f["u"], float), np.asarray(f["v"], float)             # mm
            self.B = np.asarray(f["B"], float) * 1e4 * scale                                   # tesla -> gauss
            self.origin, self.u_dir, self.v_dir, self.normal = (np.asarray(f[k], float) for k in need[3:])
        if self.B.shape != (len(self.u), len(self.v), 3) or len(self.u) < 2 or len(self.v) < 2:
            raise SystemExit("%s: B must have the shape (len(u), len(v), 3)" % path)
        self.name = os.path.basename(path)

    def check_plane(self, n, d):
        if abs(abs(float(n @ self.normal)) - 1.0) > 1e-9 or abs(float(n @ self.origin) + d) > 1e-3:
            raise SystemExit("--field %s: the field was sampled on the plane through (%g, %g, %g) mm with normal "
                             "(%g, %g, %g), not on this section plane; run solid_field_dump for this plane"
                             % ((self.name,) + tuple(self.origin) + tuple(self.normal)))

    def vector(self, P):
        """world points (N, 3) mm on the plane -> field (N, 3) gauss, inside the sampled range (N,)"""
        u, v = self.u, self.v
        q = P - self.origin[None, :]
        pu, pv = q @ self.u_dir, q @ self.v_dir
        inside = (pu >= u[0]) & (pu <= u[-1]) & (pv >= v[0]) & (pv <= v[-1])
        fi = np.clip((pu - u[0]) / (u[1] - u[0]), 0, len(u) - 1.000001)
        fj = np.clip((pv - v[0]) / (v[1] - v[0]), 0, len(v) - 1.000001)
        i, j = fi.astype(np.int64), fj.astype(np.int64)
        a, b = (fi - i)[:, None], (fj - j)[:, None]
        B = self.B
        return (B[i, j] * (1 - a) * (1 - b) + B[i + 1, j] * a * (1 - b) + B[i, j + 1] * (1 - a) * b
                + B[i + 1, j + 1] * a * b), inside

    def axial(self, n, d):
        """The in-plane field as an AxialField, if the plane holds the z axis and the two halves of the
        in-plane field mirror each other (Br odd, Bz even in the distance from the axis); else None.
        A component normal to the plane (a transverse target magnet seen in its mid plane) does not
        matter: the flux lines are those of the in-plane field."""
        ez = np.array([0.0, 0.0, 1.0])
        if abs(self.normal[2]) > 1e-9 or abs(float(self.normal @ self.origin)) > 1e-6:
            return None
        if abs(abs(self.u_dir[2]) - 1.0) < 1e-9:
            zc, w, wdir, B = self.u * self.u_dir[2], self.v, self.v_dir, np.swapaxes(self.B, 0, 1)
        elif abs(abs(self.v_dir[2]) - 1.0) < 1e-9:
            zc, w, wdir, B = self.v * self.v_dir[2], self.u, self.u_dir, self.B
        else:
            return None
        zc = zc + float(self.origin[2])
        if zc[0] > zc[-1]:
            zc, B = zc[::-1], B[:, ::-1]
        i0 = int(np.argmin(np.abs(w)))
        npos, nneg = len(w) - 1 - i0, i0
        if abs(w[i0]) > 1e-6 or npos < 2 or nneg < 2:
            return None                              # the axis is not a grid line, or one half is missing
        Br, Bz = B[i0:, :, :] @ wdir, B[i0:, :, :] @ ez            # (r >= 0, z)
        m = min(npos, nneg) + 1
        Br2, Bz2 = -(B[i0::-1, :, :] @ wdir)[:m], (B[i0::-1, :, :] @ ez)[:m]
        scale = max(float(np.abs(Br[:m]).max()), float(np.abs(Bz[:m]).max()), 1e-9)
        if max(float(np.abs(Br[:m] - Br2).max()), float(np.abs(Bz[:m] - Bz2).max())) > 2e-3 * scale:
            return None
        return AxialField(w[i0:] / 10.0, zc / 10.0, Br, Bz)


def load_field(path, scale=1.0):
    """FieldMap (GEMC text map) or SampledField (field.npz), told apart by content"""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    return SampledField(path, scale) if magic == b"PK\x03\x04" else FieldMap(path, scale)


def _in_rings(px, py, rings_by_solid):
    """points inside the section outlines: even-odd within one solid (holes stay open), any solid"""
    px, py = np.asarray(px, float), np.asarray(py, float)
    out = np.zeros(px.shape, bool)
    for rings in rings_by_solid:
        x0 = min(r[:, 0].min() for r in rings)
        x1 = max(r[:, 0].max() for r in rings)
        y0 = min(r[:, 1].min() for r in rings)
        y1 = max(r[:, 1].max() for r in rings)
        box = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        if not box.any():
            continue
        X, Y = px[box], py[box]
        ins = np.zeros(X.shape, bool)
        for ring in rings:
            Q = np.roll(ring, -1, axis=0)
            for (xa, ya), (xb, yb) in zip(ring, Q):
                if ya == yb:
                    continue
                cond = (ya > Y) != (yb > Y)
                ins ^= cond & (X < xa + (Y - ya) * (xb - xa) / (yb - ya))
        out[box] |= ins
    return out


def _streamlines(ux, uy, valid, sep, h=1.0):
    """Evenly spaced lines along a direction field.

    ux, uy (ny, nx): unit direction on a regular grid (grid units, 0 where not valid); `sep`: wanted
    distance between lines, in grid units.  All seeds of a regular grid are integrated in both
    directions (midpoint rule, step h), until they leave the grid or the valid region.  The
    candidates are then taken longest first: of each one the longest part that stays at least a
    quarter to a half of `sep` away from the lines already taken is kept, and a line that closes on
    itself is cut after one turn.
    Returns a list of (n, 2) arrays (x, y in grid units)."""
    ny, nx = ux.shape
    if nx < 3 or ny < 3 or not valid.any():
        return []

    def sample(x, y):
        inside = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
        i = np.clip(np.floor(y).astype(np.int64), 0, ny - 2)
        j = np.clip(np.floor(x).astype(np.int64), 0, nx - 2)
        a, b = y - i, x - j
        dx = (ux[i, j] * (1 - a) * (1 - b) + ux[i, j + 1] * (1 - a) * b + ux[i + 1, j] * a * (1 - b)
              + ux[i + 1, j + 1] * a * b)
        dy = (uy[i, j] * (1 - a) * (1 - b) + uy[i, j + 1] * (1 - a) * b + uy[i + 1, j] * a * (1 - b)
              + uy[i + 1, j + 1] * a * b)
        nrm = np.hypot(dx, dy)
        near = valid[np.clip(np.rint(y).astype(np.int64), 0, ny - 1), np.clip(np.rint(x).astype(np.int64), 0, nx - 1)]
        ok = inside & near & (nrm > 0.5)
        nrm = np.maximum(nrm, 1e-9)
        return dx / nrm, dy / nrm, ok

    sx, sy = np.meshgrid(np.arange(0.3 * sep, nx - 1, 0.6 * sep), np.arange(0.3 * sep, ny - 1, 0.6 * sep))
    sx, sy = sx.ravel(), sy.ravel()
    _, _, ok = sample(sx, sy)
    sx, sy = sx[ok], sy[ok]
    if len(sx) == 0:
        return []
    def valid_at(x, y):
        return ((x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
                & valid[np.clip(np.rint(y).astype(np.int64), 0, ny - 1), np.clip(np.rint(x).astype(np.int64), 0, nx - 1)])

    nsteps = int(1.2 * (nx + ny) / h)
    half = []
    for sign in (1.0, -1.0):
        T = np.full((len(sx), nsteps + 1, 2), np.nan)
        T[:, 0, 0], T[:, 0, 1] = sx, sy
        idx = np.arange(len(sx))                       # the lines still running
        x, y = sx.copy(), sy.copy()
        for k in range(nsteps):
            d1x, d1y, ok1 = sample(x, y)
            d2x, d2y, ok2 = sample(x + 0.5 * h * sign * d1x, y + 0.5 * h * sign * d1y)
            x, y = x + h * sign * d2x, y + h * sign * d2y
            ok = ok1 & ok2 & valid_at(x, y)
            if not ok.all():
                idx, x, y = idx[ok], x[ok], y[ok]
                if len(idx) == 0:
                    break
            T[idx, k + 1, 0], T[idx, k + 1, 1] = x, y
        half.append(T)
    cand = []
    for k in range(len(sx)):
        f, bk = half[0][k], half[1][k]
        f, bk = f[np.isfinite(f[:, 0])], bk[np.isfinite(bk[:, 0])]
        pts = np.concatenate([bk[:0:-1], f])
        if len(pts) >= 3:
            cand.append(pts)
    cand.sort(key=lambda q: -len(q))
    cell = 0.25 * sep                                 # a taken line keeps others 1 to 2 cells away
    occ = np.zeros((int(ny / cell) + 3, int(nx / cell) + 3), bool)
    min_pts = int(2.0 * sep / h)
    out = []
    for pts in cand:
        ci, cj = (pts[:, 1] / cell).astype(np.int64) + 1, (pts[:, 0] / cell).astype(np.int64) + 1
        # a line that closes on itself: stop when it comes back to a cell it left more than 3 cells ago
        ent = np.nonzero(np.r_[True, (ci[1:] != ci[:-1]) | (cj[1:] != cj[:-1])])[0]
        seen, cut = {}, len(pts)
        for q, k in enumerate(ent):
            key = (int(ci[k]), int(cj[k]))
            if key in seen and q - seen[key] > 3:
                cut = int(k)
                break
            seen[key] = q
        free = ~occ[ci[:cut], cj[:cut]]
        if not free.any():
            continue
        edges = np.diff(np.r_[False, free, False].astype(np.int8))
        st, en = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        k = int(np.argmax(en - st))
        a, b = int(st[k]), int(en[k])
        if b - a < min_pts:
            continue
        out.append(pts[a:b])
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                occ[ci[a:b] + di, cj[a:b] + dj] = True
    return out


def _clip_to_box(xy, W, H, y0=0.0):
    """pieces of a polyline inside the canvas (points outside are dropped)"""
    ok = (xy[:, 0] >= 0) & (xy[:, 0] <= W) & (xy[:, 1] >= y0) & (xy[:, 1] <= H)
    out = []
    idx = np.nonzero(ok)[0]
    if len(idx) < 2:
        return out
    for grp in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
        if len(grp) >= 2:
            out.append(xy[grp])
    return out


def _simplify(xy, tol):
    """Douglas-Peucker: drop vertices closer than tol px to the simplified polyline"""
    n = len(xy)
    if n < 3:
        return xy
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        p, q = xy[a], xy[b]
        v = q - p
        L = math.hypot(v[0], v[1])
        w = xy[a + 1:b] - p
        dist = np.abs(w[:, 0] * v[1] - w[:, 1] * v[0]) / L if L > 1e-12 else np.hypot(w[:, 0], w[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return xy[keep]


def draw_field(dl, sc, o, cam, plane, fit, loops, to_px, window=None):
    """Field layer of a section: bands, field lines, arrows.  Returns what was drawn, for the key
    (None if nothing was)."""
    if not cam.ortho:
        log("warning: --field needs an orthographic view, skipped", 0)
        return None
    s, cx, cy, W, H = fit
    n, d = plane
    ne = float(n @ cam.e)
    if abs(ne) < 1e-6:
        log("warning: --field: the section plane is seen edge-on, skipped", 0)
        return None
    with timed("read field"):
        fld = load_field(o.field, o.field_scale)
    fld.check_plane(n, d)
    pt = o.dpi / 72.0
    ytop = float(title_band_px(o))              # the title band stays clear
    bgc = o.background if o.background is not None else (1.0, 1.0, 1.0)
    info = {"bands": FIELD_MIN_BANDS, "lines": None, "arrows": False, "out": False, "into": False, "iron": False}

    def world(px, py):
        x, y = (px - 0.5 * W) / s + cx, (0.5 * H - py) / s + cy
        P = cam.T[None, :] + x[:, None] * cam.r[None, :] + y[:, None] * cam.u[None, :]
        t = -(P @ n + d) / ne
        return P + t[:, None] * cam.e[None, :]

    # ---- |B| bands, one value per canvas pixel
    with timed("field bands"):
        X, Y = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        B, inside = fld.vector(world(X.ravel(), Y.ravel()))
        mag = np.linalg.norm(B, axis=1)
        band = np.searchsorted(FIELD_BANDS_G, mag, side="right") - 1          # -1: below the first edge
        pal = np.array([bgc] + [_hexrgb(h) for h in FIELD_RAMP])
        band = np.where(inside & (Y.ravel() >= ytop), np.clip(band, -1, len(FIELD_RAMP) - 1), -1)
        info["bands"] = max(FIELD_MIN_BANDS, int(band.max()) + 1)
        img = (pal[band + 1].reshape(H, W, 3) * 255 + 0.5).astype(np.uint8)
        dl.image(img, 0.0, 0.0, W, H)

    # ---- iron in the plane: filled grey at the end (the field is not meaningful inside it)
    rx = re.compile(o.field_iron) if o.field_iron else None
    iron_loops, iron_rings = [], defaultdict(list)
    for sid, pts, closed in loops:
        if rx is not None and closed and len(pts) >= 3 and rx.search(sc.solids[sid].name):
            iron_loops.append(pts)
            x, y, _ = to_px(pts)
            iron_rings[sid].append(np.stack([x, y], 1))
    rings = [iron_rings[sid] for sid in sorted(iron_rings)]
    info["iron"] = bool(rings)

    # ---- field lines
    ax = fld.axial(n, d) if o.field_lines > 0 else None
    if ax is not None:
        # the in-plane field is axisymmetric about z: equal-flux lines
        try:
            from contourpy import contour_generator
        except ImportError:
            contour_generator = None
            log("warning: --field: no field lines (python module contourpy not found)", 0)
        if contour_generator is not None:
            with timed("field lines (equal flux)"):
                ur = _unit(np.cross(n, [0.0, 0.0, 1.0]))
                polys = [np.stack([np.abs(pts @ ur), pts[:, 2]], 1) / 10.0 for pts in iron_loops]
                if not polys:
                    log("  --field: no solid matches --field-iron; the flux function is integrated along r", 1)
                psi = ax.flux(ax.iron_mask(polys, grow=1))
                # levels from the part of the field that is on the canvas
                c = world(np.array([0.0, W, 0.0, W]), np.array([0.0, 0.0, H, H]))
                rmax, z0, z1 = np.abs(c @ ur).max() / 10.0, c[:, 2].min() / 10.0, c[:, 2].max() / 10.0
                sel = np.isfinite(psi) & (ax.r[:, None] <= rmax) & (ax.z[None, :] >= z0) & (ax.z[None, :] <= z1)
                pls = []
                if sel.any():
                    lo, hi = psi[sel].min(), psi[sel].max()
                    cg = contour_generator(x=ax.z, y=ax.r, z=np.ma.masked_invalid(psi))
                    for k in range(o.field_lines):
                        for ln in cg.lines(lo + (hi - lo) * (k + 0.5) / o.field_lines):
                            for sign in (1.0, -1.0):
                                P = (sign * ln[:, 1:2] * ur[None, :] + ln[:, 0:1] * np.array([[0.0, 0.0, 1.0]])) * 10.0
                                x, y, _ = to_px(P)
                                pls += [_simplify(q, 0.25) for q in _clip_to_box(np.stack([x, y], 1), W, H, ytop)]
                dl.lines(pls, bgc, 2.6 * o.field_lw)
                dl.lines(pls, FIELD_LINE_RGB, o.field_lw)
                if pls:
                    info["lines"] = "flux"
    elif o.field_lines > 0:
        # no symmetry to use: lines along the in-plane direction, evenly spaced
        with timed("field lines (evenly spaced)"):
            g = 4.0                                                    # px between direction samples
            gx, gy = np.arange(0.0, W + g, g), np.arange(ytop, H + g, g)
            GX, GY = np.meshgrid(gx, gy)
            Bg, ins = fld.vector(world(GX.ravel(), GY.ravel()))
            mg = np.linalg.norm(Bg, axis=1)
            vx, vy = Bg @ cam.r, -(Bg @ cam.u)
            vn = np.hypot(vx, vy)
            ok = ins & (mg >= FIELD_BANDS_G[0]) & (vn > 0.3 * mg) & (GX.ravel() <= W) & (GY.ravel() <= H)
            if rings:
                ok &= ~_in_rings(GX.ravel(), GY.ravel(), rings)
            vn = np.maximum(vn, 1e-30)
            ux = np.where(ok, vx / vn, 0.0).reshape(GX.shape)
            uy = np.where(ok, vy / vn, 0.0).reshape(GX.shape)
            sep = max(H - ytop, 0.5 * W) / float(o.field_lines)
            pls = [_simplify(np.stack([q[:, 0] * g, ytop + q[:, 1] * g], 1), 0.3)
                   for q in _streamlines(ux, uy, ok.reshape(GX.shape), sep / g)]
            dl.lines(pls, bgc, 2.6 * o.field_lw)
            dl.lines(pls, FIELD_LINE_RGB, o.field_lw)
            if pls:
                info["lines"] = "even"

    # ---- direction: arrows in the plane, dot / cross in a circle where the field leaves the plane
    if o.field_arrows > 0:
        step = o.field_arrows * s
        xs = np.arange(0.5 * (W % step if W % step else step), W, step)
        ys = np.arange(0.5 * (H % step if H % step else step), H, step)
        X, Y = np.meshgrid(xs, ys)
        X, Y = X.ravel(), Y.ravel()
        B, inside = fld.vector(world(X, Y))
        mag = np.linalg.norm(B, axis=1)
        vx, vy = B @ cam.r, -(B @ cam.u)
        vn = np.hypot(vx, vy)
        bn = B @ cam.e                                                 # > 0: towards the viewer
        L, hl = 0.42 * step, min(0.16 * step, 3.2 * pt)
        ok = inside & (mag >= FIELD_BANDS_G[0]) & (Y - 0.5 * step >= ytop)
        if rings:
            for ox, oy in ((0.0, 0.0), (0.5 * L, 0.0), (-0.5 * L, 0.0), (0.0, 0.5 * L), (0.0, -0.5 * L)):
                ok &= ~_in_rings(X + ox, Y + oy, rings)
        arr = ok & (vn > 0.3 * mag)
        pls = []
        for x, y, ax_, ay_ in zip(X[arr], Y[arr], (vx / np.maximum(vn, 1e-30))[arr],
                                  (vy / np.maximum(vn, 1e-30))[arr]):
            tip = np.array([x + 0.5 * L * ax_, y + 0.5 * L * ay_])
            tail = np.array([x - 0.5 * L * ax_, y - 0.5 * L * ay_])
            ang = math.atan2(ay_, ax_)
            pls.append(np.array([tail, tip]))
            pls.append(np.array([tip + hl * np.array([math.cos(ang + 2.7), math.sin(ang + 2.7)]), tip,
                                 tip + hl * np.array([math.cos(ang - 2.7), math.sin(ang - 2.7)])]))
        info["arrows"] = bool(arr.any())
        rad = min(0.15 * step, 4.2 * pt)
        for x, y, toward in zip(X[ok & ~arr], Y[ok & ~arr], (bn > 0)[ok & ~arr]):
            pls += _normal_mark(x, y, rad, toward, dl)
            info["out" if toward else "into"] = True
        dl.lines(pls, FIELD_ARROW_RGB, 0.45)
    # iron on top of the field layer; one fill per solid, so that its holes stay open (even-odd) and
    # touching pieces do not cancel
    for r_ in rings:
        if window is not None:
            r_ = [q for q in (clip_polygon_rect(q, window) for q in r_) if len(q) >= 3]
        dl.fill(r_, FIELD_IRON_RGB)
    return info


def _normal_mark(x, y, rad, toward, dl):
    """Circle with a dot (field towards the viewer) or a cross (away).  Returns its polylines; the dot
    is added to the draw list."""
    t = np.linspace(0.0, TWO_PI, 25)
    pls = [np.stack([x + rad * np.cos(t), y + rad * np.sin(t)], 1)]
    if toward:
        dl.marker("circle", x, y, 0.32 * rad, FIELD_ARROW_RGB)
    else:
        k = rad * math.sqrt(0.5)
        pls += [np.array([[x - k, y - k], [x + k, y + k]]), np.array([[x - k, y + k], [x + k, y - k]])]
    return pls


def draw_field_legend(dl, o, W0, H0, info):
    """Key right of the drawing (the canvas grows): the bands with their edges, then what the lines,
    arrows and marks mean."""
    pt = o.dpi / 72.0
    fs = max(6.0, o.font_size - 1.5)
    fpx = fs * pt
    sw, sh = 1.1 * fpx, 1.55 * fpx
    nb = info["bands"]
    keys = []                                        # (kind, text, small second line)
    if info["iron"]:
        keys.append(("iron", "iron", None))
    if info["lines"]:
        keys.append(("line", "field lines", "equal flux between lines" if info["lines"] == "flux"
                     else "direction only, evenly spaced"))
    if info["arrows"]:
        keys.append(("arrow", "direction", None))
    if info["out"]:
        keys.append(("out", "out of the page", None))
    if info["into"]:
        keys.append(("into", "into the page", None))
    row = 1.75 * fpx
    hkeys = sum(row + (0.95 * fpx if k[2] else 0.0) for k in keys)
    x0 = W0 + 1.2 * fpx
    tx = x0 + sw + 0.45 * fpx
    y1 = 0.5 * H0 + 0.5 * (nb * sh - hkeys - 1.0 * fpx)             # lower edge of the first band
    ink = (0.32, 0.32, 0.31)
    for k in range(nb):
        ya, yb = y1 - (k + 1) * sh, y1 - k * sh
        dl.fill([np.array([[x0, ya], [x0 + sw, ya], [x0 + sw, yb], [x0, yb]])], _hexrgb(FIELD_RAMP[k]))
    for k in range(nb + 1):
        dl.text(tx, y1 - k * sh + 0.34 * fpx, field_text(FIELD_BANDS_G[k]), fs, ink)
    dl.lines([np.array([[x0, y1 - nb * sh], [x0 + sw, y1 - nb * sh], [x0 + sw, y1], [x0, y1], [x0, y1 - nb * sh]])],
             (0.75, 0.75, 0.73), 0.4)
    dl.text(x0, y1 - nb * sh - 0.9 * fpx, "|B|", fs + 1, (0.05, 0.05, 0.05))
    dl.text(x0, y1 + 1.5 * fpx, "below 1 G: blank", fs - 0.5, ink)
    wmax = text_width("below 1 G: blank", (fs - 0.5) * pt) - sw - 0.45 * fpx
    yk = y1 + 3.4 * fpx
    xm = x0 + 0.5 * sw
    for kind, text, small in keys:
        ym = yk - 0.32 * fpx                         # mid height of the text line
        if kind == "iron":
            dl.fill([np.array([[x0, yk - 0.8 * fpx], [x0 + sw, yk - 0.8 * fpx], [x0 + sw, yk + 0.2 * fpx],
                               [x0, yk + 0.2 * fpx]])], FIELD_IRON_RGB)
        elif kind == "line":
            dl.lines([np.array([[x0, ym], [x0 + sw, ym]])], FIELD_LINE_RGB, max(o.field_lw, 0.6))
        elif kind == "arrow":
            hl = 0.3 * sw
            dl.lines([np.array([[x0, ym], [x0 + sw, ym]]),
                      np.array([[x0 + sw - hl, ym - 0.6 * hl], [x0 + sw, ym], [x0 + sw - hl, ym + 0.6 * hl]])],
                     FIELD_ARROW_RGB, 0.45)
        else:
            dl.lines(_normal_mark(xm, ym, 0.42 * sw, kind == "out", dl), FIELD_ARROW_RGB, 0.45)
        dl.text(tx, yk, text, fs, ink)
        wmax = max(wmax, text_width(text, fs * pt))
        yk += row
        if small:
            dl.text(tx, yk - 0.8 * fpx, small, fs - 1.5, ink)
            wmax = max(wmax, text_width(small, (fs - 1.5) * pt))
            yk += 0.95 * fpx
    dl.W = max(dl.W, int(math.ceil(tx + wmax + 0.8 * fpx)))


# =============================================================================
#  8.  2-D section mode
# =============================================================================

def render_section(sc, o, cam, plane, extra_planes):
    """Exact cut of every selected solid by `plane`; outlines (+ optional fills)."""
    rd = Renderer(sc, o)
    sel, transp = rd.select_solids()
    rd.sel, rd.transp = sel, transp
    trims = rd.trims = rd.resolve_trims(sel)
    mesh = rd.mesh
    V = snap_vertices(sc, sc.V, mesh.tri, extra_planes, o.plane_snap, False)
    V = snap_vertices(sc, V, mesh.tri, [plane], o.plane_snap, True)
    fsel = sel[sc.fsolid]
    tsel = np.nonzero(fsel[mesh.tri_face])[0]
    with timed("plane section of %d triangles" % len(tsel)):
        Q0, Q1, ti, k0, k1 = section_segments(V, mesh.tri[tsel], plane)
        sol = sc.fsolid[mesh.tri_face[tsel[ti]]]
    loops = []          # (solid, pts, closed)
    with timed("chain outlines"):
        order = np.argsort(sol, kind="stable")
        Q0, Q1, k0, k1, sol = Q0[order], Q1[order], k0[order], k1[order], sol[order]
        bounds = np.nonzero(np.diff(sol))[0] + 1
        for grp in np.split(np.arange(len(sol)), bounds):
            if len(grp) == 0:
                continue
            sid = int(sol[grp[0]])
            cut = list(extra_planes) + [(n, d) for n, d, m in trims if m[sid]]
            for pts, closed in chain_segments(k0[grp], k1[grp], Q0[grp], Q1[grp]):
                if cut:
                    if closed:
                        pts = clip_polygon(pts, cut)
                    else:
                        a, b = clip_segments(pts[:-1], pts[1:], cut)
                        if len(a) == 0:
                            continue
                        pts = np.vstack([a[:1], b])
                if len(pts) >= 2:
                    loops.append((sid, pts, closed))
    rd.n_loops = len(loops)
    allpts = [p for _, p, _ in loops]
    fit = [np.vstack(allpts)] if allpts else [V[:1]]
    if o.fit_traj:
        fit += [pl for _, pl in sc.polylines]
    if o.beam_arrow is not None and allpts:
        fit.append(rd.beam_arrow_points(float(fit[0][:, 2].min()), float(fit[0][:, 2].max())))
    s, cx, cy, W, H = rd._fit(cam, fit)
    rd._fitinfo = (s, cx, cy, W, H)
    log("  canvas %d x %d px, scale %.4g px/mm, %d outlines" % (W, H, s, len(loops)))

    def to_px(Pw):
        x, y, depth = cam.view(Pw)
        return (x - cx) * s + 0.5 * W, 0.5 * H - (y - cy) * s, cam.zkey(depth)

    rd.to_px = to_px
    dl = DrawList(W, H, o.dpi, o.background)
    win = rd.window
    field_info = draw_field(dl, sc, o, cam, plane, (s, cx, cy, W, H), loops, to_px, win) if o.field else None
    if o.fill:
        by = defaultdict(list)
        for sid, pts, closed in loops:
            if closed and len(pts) >= 3:
                x, y, _ = to_px(pts)
                ring = np.stack([x, y], 1)
                if win is not None:
                    ring = clip_polygon_rect(ring, win)
                if len(ring) >= 3:
                    by[sid].append(ring)
        w = o.fill_lighten
        for sid in sorted(by):
            c = sc.solids[sid].color
            dl.fill(by[sid], tuple(w + (1 - w) * v for v in c), alpha=o.fill_alpha)
    groups = defaultdict(list)
    vis = []
    cols = []                                       # line colours, in order of first use
    for sid, pts, closed in loops:
        x, y, _ = to_px(pts)
        xy = np.stack([x, y], 1)
        if closed:
            xy = np.vstack([xy, xy[:1]])
        col = tuple(np.array(sc.solids[sid].color) * o.line_color_scale) if o.color_lines else o.line_color
        groups[col].append(xy)
        if col not in cols:
            cols.append(col)
        k = len(xy) - 1
        vis.append((xy[:-1], xy[1:], np.full(k, sid, np.int64), np.full(k, cols.index(col), np.int64)))
    lw_sec = o.lw_cut if o.lw_cut else o.lw
    rd.vis_segs = None
    if vis:
        a = np.concatenate([v[0] for v in vis])
        b = np.concatenate([v[1] for v in vis])
        sid_, cid = np.concatenate([v[2] for v in vis]), np.concatenate([v[3] for v in vis])
        x0, y0, x1, y1 = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
        if win is not None:
            x0, y0, x1, y1, src = clip_segments_rect(x0, y0, x1, y1, win)
            sid_, cid = sid_[src], cid[src]
        rd.vis_segs = (x0, y0, x1, y1, sid_)
        if not o.keep_overdraw:
            # outlines of touching volumes share parts of their boundary: draw those once
            x0, y0, x1, y1, src = merge_collinear(x0, y0, x1, y1, np.zeros(len(x0)), cid)
            cid = cid[src]
        if win is not None or not o.keep_overdraw:
            for k, col in enumerate(cols):
                m = cid == k
                if m.any():
                    dl.lines(chain_2d(x0[m], y0[m], x1[m], y1[m]), col, lw_sec)
        else:
            for col, pls in groups.items():
                dl.lines(pls, col, lw_sec)
    rd._overlays(dl, cam, to_px, s, extra_planes)
    rd.project_beam_axis(to_px)
    rd._decorations(dl, cam, s)
    if field_info:
        draw_field_legend(dl, o, W, H, field_info)
    return dl, rd


# =============================================================================
#  9.  Callout labels, title
# =============================================================================
#  A label names the volumes whose PV name matches its regex (the first matching label wins).
#  Its anchor is a point on a line of those volumes that is visible in the finished drawing
#  (after hidden-line removal and cuts; in section mode on the section outline).  The texts go in
#  one row above and one row below the drawing; in each row they keep the order of their anchors
#  and sit as close as possible above/below them without overlapping (isotonic least squares).
#  Anchor and row of every label minimise: lines crossed by the leader, the share of other volumes'
#  lines around the anchor (ambiguity), leader length and horizontal offset, plus penalties for
#  leaders that cross each other or pass next to another anchor.  Deterministic.

LABEL_PRESETS = {
    # SoLID subsystems by PV-name prefix (assemblies appear as AV_n!Name_assembly_0#0!child), each with
    # a preferred label row (top, bottom or None), so that a subsystem is labelled in the same place in
    # every configuration.  The 3He target's holding-field coils (Target_Coil*) are no anchor for
    # 'Target': the dot belongs on the cell.
    "solid": [
        (r"(^|!)Target_(?!Coil)[^!]*$", "Target", None),
        (r"(^|!)MagnetYoke_(Endcap|Endplate|ConePlate)", "Yoke endcap", "top"),
        (r"(^|!)MagnetYoke", "CLEO-II yoke", "top"),
        (r"(^|!)MagnetCoil", "Coil + cryostat", None),
        (r"(^|!)PVDISBaffle_(plate|slab)", "Baffles", None),
        (r"(^|!)GEMTracker", "GEM", "top"),
        (r"(^|!)LightGasCherenkov", "LGC", "top"),
        (r"(^|!)HeavyGasCherenkov", "HGC", "top"),
        (r"(^|!)LiteCherenkov", "CO₂ Cherenkov", "top"),
        (r"(^|!)LAEC", "LAEC", "bottom"),
        (r"(^|!)LASPD", "LASPD", "bottom"),
        (r"(^|!)FAEC", "FAEC", "bottom"),
        (r"(^|!)FASPD", "FASPD", "bottom"),
        (r"(^|!)FAMRPC", "MRPC", "bottom"),
        (r"(^|!)Muon", "Muon system", "top"),
    ],
}


def label_list(o):
    """--labels preset merged with --label REGEX=TEXT (same REGEX replaces, empty TEXT removes):
    [(regex, text, preferred row or None)]."""
    out = list(LABEL_PRESETS[o.labels]) if o.labels else []
    for rx, txt in o.label or []:
        out = [lab for lab in out if lab[0] != rx]
        if txt:
            out.append((rx, txt, None))
    return out


def title_band_px(o):
    """Height of the band reserved above the drawing for --title / --subtitle (0 if none)."""
    if not o.title and not o.subtitle:
        return 0.0
    pt = o.dpi / 72.0
    T = o.title_size or 1.3 * o.font_size
    if not o.subtitle:
        return 3.0 / 1.3 * T * pt                 # = 3 x font size for the default title size
    return (1.6 / 1.3 * T + 2.35 * o.font_size) * pt


def draw_title(dl, o):
    pt = o.dpi / 72.0
    T = o.title_size or 1.3 * o.font_size
    font = "Helvetica-Bold" if o.title_bold else "Helvetica"
    if o.title:
        dl.text(dl.W / 2, 1.6 / 1.3 * T * pt, o.title, T, (0, 0, 0), "middle", font)
    if o.subtitle:
        y = (1.6 / 1.3 * T + 1.45 * o.font_size) * pt if o.title else 1.6 * o.font_size * pt
        dl.text(dl.W / 2, y, o.subtitle, o.font_size, (0.3, 0.3, 0.3), "middle")


def _sample_segments(A, B, step):
    """Points every ~step along 2-D segments A->B; returns (points, segment index)."""
    if len(A) == 0:
        return np.zeros((0, 2)), np.zeros(0, np.int64)
    L = np.hypot(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1])
    n = np.maximum(1, np.ceil(L / step).astype(np.int64))
    seg = np.repeat(np.arange(len(A)), n)
    k = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t = ((k + 0.5) / np.repeat(n, n))[:, None]
    return A[seg] + t * (B[seg] - A[seg]), seg


def _drawn_segments(dl):
    """Every stroked 2-D segment of a DrawList (lines and stroked fills)."""
    A, B = [], []
    for it in dl.items:
        if it[0] == "lines" or (it[0] == "fill" and it[3] is not None and it[4] > 0):
            for p in it[1]:
                p = np.asarray(p, float)
                if len(p) >= 2:
                    q = p if it[0] == "lines" else np.vstack([p, p[:1]])
                    A.append(q[:-1])
                    B.append(q[1:])
    if not A:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(A), np.concatenate(B)


def _segments_cross(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (orient(q1, q2, p1) * orient(q1, q2, p2) < 0) and (orient(p1, p2, q1) * orient(p1, p2, q2) < 0)


def _point_segment_distance(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _layout_row(ax, w, lo, hi, gap):
    """Centres of boxes (widths w, kept in the given order) as close as possible to ax in the
    least-squares sense, gap apart and inside [lo, hi]; None if they do not fit."""
    n = len(ax)
    if n == 0:
        return np.zeros(0)
    w = np.asarray(w, float)
    c = np.zeros(n)
    for k in range(1, n):
        c[k] = c[k - 1] + 0.5 * (w[k - 1] + w[k]) + gap
    if c[-1] + 0.5 * (w[0] + w[-1]) > hi - lo + 1e-9:
        return None
    vals, cnts = [], []                     # pool adjacent violators on t = ax - c
    for v in np.asarray(ax, float) - c:
        vals.append(v)
        cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, n2 = vals.pop(), cnts.pop()
            vals[-1] = (vals[-1] * cnts[-1] + v2 * n2) / (cnts[-1] + n2)
            cnts[-1] += n2
    y = np.repeat(vals, cnts)
    lower = np.maximum.accumulate(lo + 0.5 * w - c)
    upper = np.minimum.accumulate((hi - 0.5 * w - c)[::-1])[::-1]
    return np.minimum(np.maximum(y, lower), upper) + c


class _Grid:
    """Coarse occupancy raster of the canvas with integral-image box sums."""

    def __init__(self, W, H, cell):
        self.cell = cell
        self.nx, self.ny = int(W / cell) + 2, int(H / cell) + 2

    def index(self, x, y):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        return (np.clip(np.floor(y / self.cell).astype(np.int64), 0, self.ny - 1),
                np.clip(np.floor(x / self.cell).astype(np.int64), 0, self.nx - 1))

    def count(self, P):
        g = np.zeros((self.ny, self.nx))
        if len(P):
            iy, ix = self.index(P[:, 0], P[:, 1])
            np.add.at(g, (iy, ix), 1.0)
        return g

    @staticmethod
    def integral(g):
        S = np.zeros((g.shape[0] + 1, g.shape[1] + 1))
        S[1:, 1:] = g.cumsum(0).cumsum(1)
        return S

    @staticmethod
    def box(S, iy, ix, r):
        ny, nx = S.shape[0] - 1, S.shape[1] - 1
        y0, y1 = np.clip(iy - r, 0, ny), np.clip(iy + r + 1, 0, ny)
        x0, x1 = np.clip(ix - r, 0, nx), np.clip(ix + r + 1, 0, nx)
        return S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]


def _occupancy(dl, grid):
    """Ink counts of every stroked line on the grid, and an occupancy map in which drawn lines
    count 1 per cell and texts and dots are hard obstacles."""
    pt = dl.dpi / 72.0
    DA, DB = _drawn_segments(dl)
    ink = grid.count(_sample_segments(DA, DB, 0.5 * grid.cell)[0])
    occ = (ink > 0).astype(float)
    for it in dl.items:
        if it[0] == "text":
            b = text_box(it[1], it[2], it[3], it[4] * pt, it[6], _is_bold(it[7]))
        elif it[0] == "marker":
            b = (it[2] - it[4], it[3] - it[4], it[2] + it[4], it[3] + it[4])
        else:
            continue
        (iy0,), (ix0,) = grid.index(b[0], b[1])
        (iy1,), (ix1,) = grid.index(b[2], b[3])
        occ[iy0:iy1 + 1, ix0:ix1 + 1] += 20.0
    return ink, occ


def _grow_canvas(dl, pad, top):
    """Grow the canvas (never shrink it) so that everything is inside, at least `top` px below the
    upper edge and `pad` px from the other edges."""
    x0, y0, x1, y1 = dl.bbox()
    dx, dy = max(0.0, pad - x0), max(0.0, top - y0)
    if dx > 0 or dy > 0:
        dl.translate(dx, dy)
        dl.W += int(math.ceil(dx))
        dl.H += int(math.ceil(dy))
    dl.W = max(dl.W, int(math.ceil(x1 + dx + pad)))
    dl.H = max(dl.H, int(math.ceil(y1 + dy + pad)))


def finish_canvas(dl, o):
    """After the beam arrow and labels: equal left and right margins (1.2 em) and 0.9 em below the
    lowest text; the band reserved above for the title is kept."""
    fpx = (o.label_size or o.font_size) * o.dpi / 72.0
    x0, _, x1, y1 = dl.bbox()
    dl.translate(1.2 * fpx - x0, 0.0)
    dl.W = int(math.ceil(x1 - x0 + 2.4 * fpx))
    dl.H = int(math.ceil(y1 + 0.9 * fpx))


def place_beam_arrow(dl, rd, o):
    """--beam-arrow: arrow on the projected beam axis pointing downstream, labelled below.  Its tip
    starts at the upstream end of the drawing and moves upstream until shaft, tip clearance and
    text touch no drawn line."""
    axis = getattr(rd, "beam_axis_px", None)
    if axis is None:
        return
    pt = o.dpi / 72.0
    fs = o.label_size or o.font_size
    fpx = fs * pt
    tail0, tip0, start = [p + dl.origin for p in axis]
    L = float(np.linalg.norm(tip0 - tail0))
    if L < 1e-6:
        return
    u = (tip0 - tail0) / L
    nv = np.array([-u[1], u[0]])
    L = max(L, 4.0 * fpx)
    text = o.beam_arrow
    tw = text_width(text, fpx) if text else 0.0
    below = abs(u[0]) > 0.3

    def layout(tip):
        tail = tip - L * u
        mid = 0.5 * (tip + tail)
        if not text:
            return tail, None
        if below:      # under the middle of the shaft, clear of its slope
            return tail, (float(mid[0]), float(mid[1] + abs(u[1] / u[0]) * 0.5 * tw + 1.34 * fpx), "middle")
        return tail, (float(tail[0] - 0.5 * fpx), float(tail[1] + 0.35 * fpx), "end")

    grid = _Grid(dl.W, dl.H, max(2.0, 0.25 * fpx))
    _, occ = _occupancy(dl, grid)
    step = 0.5 * grid.cell
    tip = start
    for _ in range(600):
        tail, tpos = layout(tip)
        s = np.arange(0.0, L + 0.8 * fpx, step)
        P = [tail[None, :] + s[:, None] * u[None, :] + off * nv[None, :] for off in (-0.6 * fpx, 0.0, 0.6 * fpx)]
        if tpos:
            bx0, by0, bx1, by1 = text_box(tpos[0], tpos[1], text, fpx, tpos[2])
            gx, gy = np.meshgrid(np.arange(bx0 - 0.3 * fpx, bx1 + 0.3 * fpx, step),
                                 np.arange(by0 - 0.3 * fpx, by1 + 0.3 * fpx, step))
            P.append(np.stack([gx.ravel(), gy.ravel()], 1))
        P = np.concatenate(P)
        inside = (P[:, 0] >= 0) & (P[:, 0] < dl.W) & (P[:, 1] >= 0) & (P[:, 1] < dl.H)
        iy, ix = grid.index(P[inside, 0], P[inside, 1])
        if not occ[iy, ix].any():
            break
        tip = tip - step * u
    tail, tpos = layout(tip)
    lw, hl, hw = 1.2, min(9.0 * pt, 0.45 * L), 3.2 * pt
    head = np.array([tip, tip - hl * u + hw * nv, tip - hl * u - hw * nv])
    dl.lines([np.array([tail, tip - 0.9 * hl * u])], (0, 0, 0), lw, cap="butt")
    dl.fill([head], (0, 0, 0))
    if tpos:
        dl.text(tpos[0], tpos[1], text, fs, (0, 0, 0), tpos[2])
    _grow_canvas(dl, 0.5 * fpx, title_band_px(o) + 0.4 * fpx)


def place_labels(dl, rd, o, labels):
    """Add callout labels to dl (the canvas grows above/below as needed).  Returns the placements
    [(text, row, anchor, text x)]."""
    sc = rd.sc
    vis = getattr(rd, "vis_segs", None)
    if not labels or vis is None or len(vis[0]) == 0:
        return []
    pt = o.dpi / 72.0
    fs = o.label_size or o.font_size
    fpx = fs * pt
    regs = [re.compile(lab[0]) for lab in labels]
    pref = [lab[2] if len(lab) > 2 else None for lab in labels]          # preferred row
    gid = np.full(len(sc.solids), -1, np.int64)
    for s in sc.solids:
        if rd.sel[s.index]:
            key = s.path if o.match_path else s.name
            for gi, rx in enumerate(regs):
                if rx.search(key):
                    gid[s.index] = gi
                    break
    x0, y0, x1, y1, sol = vis
    g_all = gid[sol]
    m = g_all >= 0
    A = np.stack([x0[m], y0[m]], 1) + dl.origin          # renderer px -> current canvas px
    B = np.stack([x1[m], y1[m]], 1) + dl.origin
    g = g_all[m]
    if len(A) == 0:
        return []

    # ---- occupancy of everything drawn so far (lines; texts and dots are obstacles)
    grid = _Grid(dl.W, dl.H, max(2.0, 0.25 * fpx))
    cell = grid.cell
    ink, occ = _occupancy(dl, grid)
    col = np.cumsum(occ, 0)
    bx0, by0, bx1, by1 = dl.bbox()
    rows = [s for s in ("top", "bottom") if o.label_rows in (s, "both")]
    y_end = {"top": by0 - 0.9 * fpx, "bottom": by1 + 0.9 * fpx}            # leader ends
    baseline = {"top": y_end["top"] - 0.4 * fpx, "bottom": y_end["bottom"] + 0.95 * fpx}
    lo, hi, hgap = 0.5 * fpx, dl.W - 0.5 * fpx, 0.9 * fpx

    def crossings(a, b):
        """Occupied cells along a leader, ignoring the first 0.5 em at the anchor."""
        L = float(np.hypot(*(b - a)))
        t = np.arange(0.5 * fpx, L, 0.5 * cell) / max(L, 1e-9)
        if len(t) == 0:
            return 0.0
        iy, ix = grid.index(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        return float(occ.ravel()[np.unique(iy * grid.nx + ix)].sum())

    # ---- candidate anchors per label: a few good, spread-out points per row
    cands, texts, widths = {}, {}, {}
    pts, seg = _sample_segments(A, B, 0.35 * fpx)
    pg = g[seg]
    # grid cells holding lines of each label, and cells holding anything else (other volumes,
    # beam arrow); the ambiguity of an anchor is the share of the occupied cells within ~0.6 em
    # that hold other lines.  Counting cells rather than ink length keeps a dot on a short ragged
    # fragment of many coincident edges, pressed against another subsystem, from looking safe.
    vis_pts, vis_seg = _sample_segments(np.stack([x0, y0], 1) + dl.origin,
                                        np.stack([x1, y1], 1) + dl.origin, 0.5 * cell)
    vis_g = g_all[vis_seg]
    other_ink = (ink > 0) & (grid.count(vis_pts) == 0)
    for gi in sorted(set(pg.tolist())):
        P = pts[pg == gi]
        if len(P) > 800:
            P = P[np.linspace(0, len(P) - 1, 800).astype(np.int64)]
        S_own = grid.integral(grid.count(vis_pts[vis_g == gi]) > 0)
        S_other = grid.integral((grid.count(vis_pts[vis_g != gi]) > 0) | other_ink)
        iy, ix = grid.index(P[:, 0], P[:, 1])
        n_other = grid.box(S_other, iy, ix, 2)
        amb = n_other / np.maximum(grid.box(S_own, iy, ix, 2) + n_other, 1e-9)
        up = np.where(iy - 3 >= 0, col[np.maximum(iy - 3, 0), ix], 0.0)       # vertical-leader estimate
        down = col[-1, ix] - col[np.minimum(iy + 2, grid.ny - 1), ix]
        cands[gi] = []
        for side in rows:
            est = (up if side == "top" else down) + 8.0 * amb + 0.15 * np.abs(P[:, 1] - y_end[side]) / fpx
            picked = []
            for k in np.argsort(est, kind="stable"):
                if all(np.hypot(*(P[k] - P[j])) > 1.5 * fpx for j in picked):
                    picked.append(k)
                if len(picked) == 6:
                    break
            cands[gi] += [(side, P[k].copy(), float(amb[k])) for k in picked]
        texts[gi] = labels[gi][1]
        widths[gi] = text_width(texts[gi], fpx)

    def end_of(r):
        return np.array([r[2], y_end[r[0]]])

    def evaluate(choice):
        """choice {label: candidate} -> (cost, {label: (row, anchor, text x)}); in each row the
        texts follow the anchors, leaders that would cross swap places."""
        res = {}
        for side in rows:
            order = sorted((gi for gi in choice if cands[gi][choice[gi]][0] == side),
                           key=lambda gi: (cands[gi][choice[gi]][1][0], gi))
            for _ in range(2 * len(order) + 1):
                xs = _layout_row([cands[gi][choice[gi]][1][0] for gi in order],
                                 [widths[gi] for gi in order], lo, hi, hgap)
                if xs is None:
                    return 1e9, None
                anc = [cands[gi][choice[gi]][1] for gi in order]
                swap = next(((i, j) for i in range(len(order)) for j in range(i + 1, len(order))
                             if _segments_cross(anc[i], np.array([xs[i], y_end[side]]),
                                                anc[j], np.array([xs[j], y_end[side]]))), None)
                if swap is None:
                    break
                i, j = swap
                order[i], order[j] = order[j], order[i]
            for gi, x in zip(order, xs):
                res[gi] = (side, cands[gi][choice[gi]][1], float(x))
        keys = sorted(res)
        cost = 0.0
        for gi in keys:
            a, e = res[gi][1], end_of(res[gi])
            cost += crossings(a, e) + 8.0 * cands[gi][choice[gi]][2] + \
                0.15 * float(np.hypot(*(e - a))) / fpx + 0.1 * abs(e[0] - a[0]) / fpx
        for i, gi in enumerate(keys):
            ai, ei = res[gi][1], end_of(res[gi])
            for gj in keys[i + 1:]:
                aj, ej = res[gj][1], end_of(res[gj])
                if _segments_cross(ai, ei, aj, ej):
                    cost += 6.0
                if np.hypot(*(ai - aj)) < 0.8 * fpx:
                    cost += 6.0
                if min(_point_segment_distance(aj, ai, ei), _point_segment_distance(ai, aj, ej)) < 0.4 * fpx:
                    cost += 4.0
        if len(rows) == 2:
            ntop = sum(1 for gi in keys if res[gi][0] == "top")
            cost += 0.6 * abs(2 * ntop - len(keys))                             # balance the rows
            cost += 2.5 * sum(1 for gi in keys if pref[gi] not in (None, res[gi][0]))
        return cost, res

    # ---- start from the best anchor of each label, then coordinate descent over (row, anchor);
    #      if the rows cannot hold all texts, the last labels of the list are dropped
    choice = {}
    for gi in sorted(cands):
        if cands[gi]:
            choice[gi] = min(range(len(cands[gi])), key=lambda k: (crossings(cands[gi][k][1], np.array(
                [cands[gi][k][1][0], y_end[cands[gi][k][0]]])) + 8.0 * cands[gi][k][2], k))
    while choice:
        cost, res = evaluate(choice)
        for _ in range(6):
            improved = False
            for gi in sorted(choice):
                for k in range(len(cands[gi])):
                    if k != choice[gi]:
                        trial = dict(choice)
                        trial[gi] = k
                        c, r = evaluate(trial)
                        if c < cost - 1e-9:
                            cost, res, choice, improved = c, r, trial, True
            if not improved:
                break
        if res is not None:
            break
        dropped = max(choice)
        log("  labels: no room for '%s' in the %s row(s), dropped" % (texts[dropped], "/".join(rows)))
        del choice[dropped]
    if not choice or res is None:
        return []

    # ---- draw: leader halos, leaders, anchor dots, texts
    halo = o.background or (1, 1, 1)
    out = [(texts[gi],) + res[gi] for gi in sorted(res)]
    leaders = [np.array([a, end_of((side, a, x))]) for _, side, a, x in out]
    dl.lines(leaders, halo, o.label_lw + 1.8, cap="round")
    for _, _, a, _ in out:
        dl.marker("circle", float(a[0]), float(a[1]), (o.label_dot + 0.8) * pt, halo)
    dl.lines(leaders, o.label_color, o.label_lw, cap="round")
    for txt, side, a, x in out:
        dl.marker("circle", float(a[0]), float(a[1]), o.label_dot * pt, o.label_color)
        dl.text(float(x), float(baseline[side]), txt, fs, (0, 0, 0), "middle")
        log("  label %-16s %-6s anchor (%.0f, %.0f) px" % (txt, side, a[0], a[1]))
    _grow_canvas(dl, 0.4 * fpx, title_band_px(o) + 0.4 * fpx)
    return out


# =============================================================================
#  10.  Text metrics and rich text
# =============================================================================
#  All text is set in Helvetica: SVG asks for "Helvetica, Arial", PDF/EPS use the Type1
#  Helvetica and PNG the fontconfig match of "Helvetica" (Nimbus Sans, Liberation Sans or Arial,
#  all with the Helvetica advance widths), so label layouts computed from the Adobe metrics below
#  hold in every format.  Unicode sub/superscript digits (NH₃, ³He, e⁻) are drawn as smaller,
#  shifted runs of ordinary glyphs, which every font has.  Greek letters (J/ψ) are drawn as glyph
#  outlines in SVG, PDF and EPS (and in PNG when the PNG font lacks them), taken from the PNG font
#  if it is a scalable font with that glyph, else from the DejaVu Sans that matplotlib ships;
#  without either, SVG keeps the character and PDF/EPS use the Symbol font.  Rich SVG text is
#  written run by run with absolute positions: simple renderers (GraphicsMagick/ImageMagick)
#  misplace <tspan> runs shifted with dy and drop blanks at run borders.

_HELV_W = (278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
           556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
           1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
           667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
           333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
           556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584)
_HELV_BOLD_W = (278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
                556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
                975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
                667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
                333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
                611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584)
# Greek and the minus sign -> Symbol font code and width
_SYMBOL = {"α": ("a", 631), "β": ("b", 549), "γ": ("g", 411), "δ": ("d", 494), "ε": ("e", 439),
           "ζ": ("z", 494), "η": ("h", 603), "θ": ("q", 521), "ι": ("i", 329), "κ": ("k", 549),
           "λ": ("l", 549), "μ": ("m", 576), "ν": ("n", 521), "ξ": ("x", 493), "π": ("p", 549),
           "ρ": ("r", 549), "σ": ("s", 603), "τ": ("t", 439), "υ": ("u", 576), "φ": ("f", 521),
           "χ": ("c", 549), "ψ": ("y", 686), "ω": ("w", 686), "Γ": ("G", 603), "Δ": ("D", 612),
           "Θ": ("Q", 741), "Λ": ("L", 686), "Ξ": ("X", 645), "Π": ("P", 768), "Σ": ("S", 592),
           "Φ": ("F", 763), "Ψ": ("Y", 795), "Ω": ("W", 768), "−": ("-", 549)}
_GREEK = frozenset(ch for ch in _SYMBOL if ch != "−")
_SUBSCRIPTS = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+−"))
_SUPERSCRIPTS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+−"))
_RUN_STYLE = {"": (1.0, 0.0), "sub": (0.7, 0.18), "sup": (0.7, -0.38)}   # size factor, shift (em, y down)


def _is_bold(font):
    return "bold" in (font or "").lower()


def text_runs(s):
    """'NH₃ e⁻' -> [['NH', ''], ['3', 'sub'], [' e', ''], ['−', 'sup']]."""
    runs = []
    for ch in s:
        if ch in _SUBSCRIPTS:
            role, ch = "sub", _SUBSCRIPTS[ch]
        elif ch in _SUPERSCRIPTS:
            role, ch = "sup", _SUPERSCRIPTS[ch]
        else:
            role = ""
        if runs and runs[-1][1] == role:
            runs[-1][0] += ch
        else:
            runs.append([ch, role])
    return runs


_OUTLINES = {}


def _cairo_font_options(cairo):
    """The one set of font options used with cairo here: cairo resolves a toy font face once,
    with the options of its first use, so outlines and PNG text must agree on them."""
    fo = cairo.FontOptions()
    fo.set_antialias(cairo.ANTIALIAS_GRAY)
    fo.set_hint_style(cairo.HINT_STYLE_NONE)
    fo.set_hint_metrics(cairo.HINT_METRICS_OFF)
    return fo


def _cairo_glyph(ch, bold):
    """(advance, path) of ch in the fontconfig 'Helvetica' that the PNG uses, or None if pycairo
    is missing, the font is not scalable (X11 bitmap Helvetica) or lacks the glyph."""
    try:
        import cairo
    except ImportError:
        return None
    cr = cairo.Context(cairo.RecordingSurface(cairo.CONTENT_ALPHA, None))
    cr.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL,
                        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_options(_cairo_font_options(cairo))
    cr.set_font_size(1000.0)
    ops = {cairo.PATH_MOVE_TO: "M", cairo.PATH_LINE_TO: "L", cairo.PATH_CURVE_TO: "C", cairo.PATH_CLOSE_PATH: "Z"}

    def path_of(c):
        cr.new_path()
        cr.move_to(0.0, 0.0)
        cr.text_path(c)
        return [(ops[k],) + tuple(v) for k, v in cr.copy_path()]

    if not any(op[0] == "C" for op in path_of("o")):        # bitmap glyphs are traced as pixel squares
        return None
    glyphs = cr.get_scaled_font().text_to_glyphs(0.0, 0.0, ch, False)
    if len(glyphs) != 1 or glyphs[0].index == 0:
        return None
    return cr.text_extents(ch).x_advance, path_of(ch)


def _mpl_glyph(ch, bold):
    """(advance, path) of ch in the DejaVu Sans shipped with matplotlib, or None."""
    from matplotlib import ft2font
    from matplotlib.font_manager import FontProperties, findfont, get_font
    from matplotlib.textpath import TextToPath
    fp = FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    font = get_font(findfont(fp, fallback_to_default=False))
    if font.get_char_index(ord(ch)) == 0:
        return None
    t2p = TextToPath()
    verts, codes = t2p.get_text_path(fp, ch)                # units of 1/FONT_SCALE em, y up
    font.set_size(t2p.FONT_SCALE, t2p.DPI)
    flags = ft2font.LoadFlags.NO_HINTING if hasattr(ft2font, "LoadFlags") else ft2font.LOAD_NO_HINTING
    k = 1000.0 / t2p.FONT_SCALE
    adv = font.load_char(ord(ch), flags=flags).linearHoriAdvance / 65536.0 * k
    V = [(float(x) * k, -float(y) * k) for x, y in verts]
    path, cur, i = [], (0.0, 0.0), 0
    while i < len(codes):
        c = int(codes[i])
        if c in (1, 2):                                     # MOVETO, LINETO
            path.append(("M" if c == 1 else "L",) + V[i])
            cur, i = V[i], i + 1
        elif c == 3:                                        # quadratic -> cubic Bezier
            (qx, qy), (ex, ey) = V[i], V[i + 1]
            path.append(("C", cur[0] + 2.0 / 3.0 * (qx - cur[0]), cur[1] + 2.0 / 3.0 * (qy - cur[1]),
                         ex + 2.0 / 3.0 * (qx - ex), ey + 2.0 / 3.0 * (qy - ey), ex, ey))
            cur, i = (ex, ey), i + 2
        elif c == 4:
            path.append(("C",) + V[i] + V[i + 1] + V[i + 2])
            cur, i = V[i + 2], i + 3
        else:                                               # CLOSEPOLY (and STOP)
            if c == 79:
                path.append(("Z",))
            i += 1
    return adv, path


def _glyph_outline(ch, bold=False):
    """Outline of ch as (advance, [(op, coordinates...)]) in 1/1000 em, origin on the baseline,
    y down; op M/L/C/Z.  From the PNG font if it is scalable and has the glyph, else from the
    DejaVu Sans of matplotlib; None without either (cached)."""
    key = (ch, bool(bold))
    if key not in _OUTLINES:
        g = None
        for source in (_cairo_glyph, _mpl_glyph):
            try:
                g = source(ch, bold)
            except Exception:           # missing module or font, old API: try the next source
                g = None
            if g is not None:
                break
        _OUTLINES[key] = g
    return _OUTLINES[key]


def _outlined(ch, bold=False):
    """Characters Helvetica does not have are drawn as outlines in SVG/PDF/EPS when an outline is
    available: Greek letters, and anything beyond Latin-1 such as arrows.  As text they would come
    from a fallback font of the viewer, with another width than the layout assumes (a browser then
    squeezes the neighbouring spaces to keep the given text length)."""
    return (ch in _GREEK or (ord(ch) > 0xFF and ch not in _SYMBOL)) and _glyph_outline(ch, bold) is not None


def _glyph_width(ch, bold=False):
    o = ord(ch)
    if 32 <= o <= 126:
        return (_HELV_BOLD_W if bold else _HELV_W)[o - 32]
    if _outlined(ch, bold):
        return _glyph_outline(ch, bold)[0]
    if ch in _SYMBOL:
        return _SYMBOL[ch][1]
    return 556


def text_width(s, size, bold=False):
    """Advance width of the rich text s, in the units of size (Helvetica metrics)."""
    return sum(_RUN_STYLE[r][0] * size * sum(_glyph_width(c, bold) for c in t) / 1000.0
               for t, r in text_runs(s))


def text_box(x, y, s, size, anchor="start", bold=False):
    """Ink box (x0, y0, x1, y1) of text with baseline at y (y down), in the units of size."""
    w = text_width(s, size, bold)
    x0 = x - {"start": 0.0, "middle": 0.5 * w, "end": w}[anchor]
    roles = {r for _, r in text_runs(s)}
    top = 0.88 if "sup" in roles else 0.74
    bottom = 0.33 if "sub" in roles else 0.22
    return x0, y - top * size, x0 + w, y + bottom * size


_PS_FONTS = {"F1": "Helvetica-L1", "F2": "Symbol", "F3": "Helvetica-Bold-L1"}
_PS_PATH_OPS = {"M": "m", "L": "l", "C": "c", "Z": "h"}       # PDF operators, defined in the EPS prolog


def _font_segments(text, bold, encoding="cp1252"):
    """Split a run into [font, string] for PDF/EPS: F1/F3 Helvetica(-Bold) bytes in `encoding`
    (cp1252 = PDF WinAnsi, latin-1 = PostScript ISOLatin1), F2 Symbol, G one outlined glyph."""
    segs = []
    for ch in text:
        if _outlined(ch, bold):
            segs.append(["G", ch])
            continue
        if ch in _SYMBOL:
            f, c = "F2", _SYMBOL[ch][0]
        else:
            f, c = ("F3" if bold else "F1"), ch.encode(encoding, "replace").decode("latin-1")
        if segs and segs[-1][0] == f:
            segs[-1][1] += c
        else:
            segs.append([f, c])
    return segs


_SYMBOL_CODE_W = {code: w for code, w in _SYMBOL.values()}


def _segment_width(font, s, bold):
    """Width (1/1000 em) of an already encoded PDF/EPS segment."""
    if font == "F2":
        return sum(_SYMBOL_CODE_W.get(c, 600) for c in s)
    return sum(_glyph_width(c, bold) if 32 <= ord(c) <= 126 else 556 for c in s)


# =============================================================================
#  11.  Writers: SVG, PDF, EPS (hand written), PNG (pycairo or matplotlib)
# =============================================================================

def _fmt(a):
    return " ".join("%.2f" % v for v in a)


def _xml_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _outline_ops(path, ox, oy, k, flip=False):
    """Glyph outline (1/1000 em, y down) placed at (ox, oy) with scale k: [(op, x, y, ...)], y up
    when flip (PDF/EPS points)."""
    out = []
    for op in path:
        xy = [ox + k * v if i % 2 == 0 else (oy - k * v if flip else oy + k * v) for i, v in enumerate(op[1:])]
        out.append((op[0],) + tuple(xy))
    return out


def _svg_text(x, y, s, size_px, rgb, anchor, font):
    """SVG of one text item.  Plain text is a single <text>.  Rich text (sub/superscripts, Greek)
    is set run by run: each run is its own <text> at an absolute position from the Helvetica
    metrics, without blanks at its ends, and Greek letters are glyph outlines (<path>)."""
    bold = _is_bold(font)
    weight = ' font-weight="bold"' if bold else ""
    runs = text_runs(s)
    if all(r == "" for _, r in runs) and not any(_outlined(c, bold) for c in s):
        return ('<text x="%.2f" y="%.2f" font-family="Helvetica, Arial, sans-serif" font-size="%.2f" '
                'fill="%s" text-anchor="%s"%s>%s</text>' % (x, y, size_px, _rgbhex(rgb), anchor, weight,
                                                           _xml_escape(s)))
    cx = x - {"start": 0.0, "middle": 0.5, "end": 1.0}[anchor] * text_width(s, size_px, bold)
    parts = []
    for t, role in runs:
        f, shift = _RUN_STYLE[role]
        fs, by, k = f * size_px, y + shift * size_px, f * size_px / 1000.0
        pieces = []                                 # plain strings and single outlined glyphs
        for c in t:
            if _outlined(c, bold):
                pieces.append((True, c))
            elif pieces and not pieces[-1][0]:
                pieces[-1] = (False, pieces[-1][1] + c)
            else:
                pieces.append((False, c))
        for glyph, piece in pieces:
            if glyph:
                adv, path = _glyph_outline(piece, bold)
                d = "".join(op[0] + " ".join("%.2f" % v for v in op[1:]) for op in _outline_ops(path, cx, by, k))
                parts.append('<path fill="%s" d="%s"/>' % (_rgbhex(rgb), d))
                cx += adv * k
                continue
            core = piece.strip(" ")
            if core:
                x0 = cx + k * sum(_glyph_width(c, bold) for c in piece[:len(piece) - len(piece.lstrip(" "))])
                w = k * sum(_glyph_width(c, bold) for c in core)
                parts.append('<text x="%.2f" y="%.2f" font-family="Helvetica, Arial, sans-serif" font-size="%.2f" '
                             'fill="%s"%s%s>%s</text>' % (x0, by, fs, _rgbhex(rgb), weight,
                                                         ' textLength="%.2f"' % w if len(core) > 1 else "",
                                                         _xml_escape(core)))
            cx += k * sum(_glyph_width(c, bold) for c in piece)
    return "\n".join(parts)


def write_svg(dl, path):
    k = 72.0 / dl.dpi
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           'version="1.1" width="%.2fpt" height="%.2fpt" viewBox="0 0 %d %d">' % (dl.W * k, dl.H * k, dl.W, dl.H),
           '<!-- generated by solid_lineview %s -->' % __version__]
    if dl.bg:
        out.append('<rect x="0" y="0" width="%d" height="%d" fill="%s"/>' % (dl.W, dl.H, _rgbhex(dl.bg)))
    out += svg_items(dl.items, dl.dpi)
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def svg_items(items, dpi):
    """The SVG elements of DrawList items, one string per item (what write_svg puts between <svg> and
    </svg>; solid_evdisplay builds animated SVGs from them)."""
    out = []
    pt = dpi / 72.0
    for it in items:
        if it[0] == "lines":
            _, pls, rgb, lw, dash, cap = it
            d = []
            for p in pls:
                d.append("M" + _fmt(p[0]) + "L" + " ".join(_fmt(q) for q in p[1:]))
            extra = ' stroke-dasharray="%s"' % ",".join("%g" % (v * pt) for v in dash) if dash else ""
            out.append('<path fill="none" stroke="%s" stroke-width="%.3f" stroke-linecap="%s" '
                       'stroke-linejoin="round"%s d="%s"/>' % (_rgbhex(rgb), lw * pt, cap, extra, "".join(d)))
        elif it[0] == "fill":
            _, rings, rgb, stroke, lw, alpha = it
            d = "".join("M" + _fmt(r[0]) + "L" + " ".join(_fmt(q) for q in r[1:]) + "Z" for r in rings)
            st = ' stroke="%s" stroke-width="%.3f" stroke-linejoin="round"' % (_rgbhex(stroke), lw * pt) \
                if stroke is not None and lw > 0 else ' stroke="none"'
            op = ' fill-opacity="%.3f"' % alpha if alpha < 1 else ""
            out.append('<path fill="%s" fill-rule="evenodd"%s%s d="%s"/>' % (_rgbhex(rgb), op, st, d))
        elif it[0] == "marker":
            _, kind, x, y, r, rgb = it
            if kind == "circle":
                out.append('<circle cx="%.2f" cy="%.2f" r="%.2f" fill="%s"/>' % (x, y, r, _rgbhex(rgb)))
            else:
                out.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>'
                           % (x - r, y - r, 2 * r, 2 * r, _rgbhex(rgb)))
        elif it[0] == "text":
            _, x, y, s, size, rgb, anchor, font = it
            out.append(_svg_text(x, y, s, size * pt, rgb, anchor, font))
        elif it[0] == "image":
            import base64
            _, img, ix, iy, iw, ih = it
            png = _png_bytes(img)
            out.append('<image x="%.2f" y="%.2f" width="%.2f" height="%.2f" preserveAspectRatio="none" '
                       'xlink:href="data:image/png;base64,%s"/>' % (ix, iy, iw, ih, base64.b64encode(png).decode()))
    return out


def _png_bytes(img):
    """Minimal PNG encoder (RGB8) with zlib, no dependencies."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _ps_escape(s):
    s = s.encode("latin-1", "replace").decode("latin-1")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(dl, eps=False):
    """PDF (or PostScript) drawing operators in pt with y up."""
    k = 72.0 / dl.dpi
    Hpt = dl.H * k
    o = []
    if eps:
        M, L, S, RG, rg, W_, J, F = "m", "l", "S", "RG", "rg", "w", "J", "f"
    else:
        M, L, S, RG, rg, W_, J, F = "m", "l", "S", "RG", "rg", "w", "J", "f*"
    o.append("1 j")
    if dl.bg:
        o.append("%.4f %.4f %.4f %s" % (tuple(dl.bg) + (rg,)))
        o.append("0 0 %.2f %.2f re %s" % (dl.W * k, Hpt, F if not eps else "f"))
    images = []

    def pts(p):
        return [(x * k, Hpt - y * k) for x, y in p]

    for it in dl.items:
        if it[0] == "lines":
            _, pls, rgb, lw, dash, cap = it
            o.append("%.4f %.4f %.4f %s %.3f %s %d %s" % (rgb[0], rgb[1], rgb[2], RG, lw, W_,
                                                     1 if cap == "round" else 0, J))
            o.append(("[%s] 0 d" % " ".join("%g" % v for v in dash)) if dash else "[] 0 d")
            buf = []
            for p in pls:
                q = pts(p)
                buf.append("%.2f %.2f %s" % (q[0][0], q[0][1], M))
                buf.extend("%.2f %.2f %s" % (x, y, L) for x, y in q[1:])
                buf.append(S)
            o.append("\n".join(buf))
        elif it[0] == "fill":
            _, rings, rgb, stroke, lw, alpha = it
            o.append("%.4f %.4f %.4f %s" % (rgb[0], rgb[1], rgb[2], rg))
            buf = []
            for r in rings:
                q = pts(r)
                buf.append("%.2f %.2f %s" % (q[0][0], q[0][1], M))
                buf.extend("%.2f %.2f %s" % (x, y, L) for x, y in q[1:])
                buf.append("h")
            if stroke is not None and lw > 0:
                o.append("%.4f %.4f %.4f %s %.3f %s" % (stroke[0], stroke[1], stroke[2], RG, lw, W_))
                buf.append("B*" if not eps else "gsave eofill grestore stroke")
            else:
                buf.append("f*" if not eps else "eofill")
            o.append("\n".join(buf))
        elif it[0] == "marker":
            _, kind, x, y, r, rgb = it
            (cxp, cyp), = pts([(x, y)])
            rp = r * k
            o.append("%.4f %.4f %.4f %s" % (rgb[0], rgb[1], rgb[2], rg))
            if kind == "circle":
                c = 0.5523 * rp
                o.append("%.2f %.2f m %.2f %.2f %.2f %.2f %.2f %.2f c %.2f %.2f %.2f %.2f %.2f %.2f c "
                         "%.2f %.2f %.2f %.2f %.2f %.2f c %.2f %.2f %.2f %.2f %.2f %.2f c %s"
                         % (cxp + rp, cyp, cxp + rp, cyp + c, cxp + c, cyp + rp, cxp, cyp + rp,
                            cxp - c, cyp + rp, cxp - rp, cyp + c, cxp - rp, cyp,
                            cxp - rp, cyp - c, cxp - c, cyp - rp, cxp, cyp - rp,
                            cxp + c, cyp - rp, cxp + rp, cyp - c, cxp + rp, cyp, "f" if not eps else "fill")
                         if not eps else
                         "newpath %.2f %.2f %.2f 0 360 arc fill" % (cxp, cyp, rp))
            else:
                o.append("%.2f %.2f %.2f %.2f re %s" % (cxp - rp, cyp - rp, 2 * rp, 2 * rp, "f")
                         if not eps else "%.2f %.2f %.2f %.2f rectfill" % (cxp - rp, cyp - rp, 2 * rp, 2 * rp))
        elif it[0] == "text":
            _, x, y, s, size, rgb, anchor, font = it
            (xp, yp), = pts([(x, y)])
            o.append("%.4f %.4f %.4f %s" % (rgb[0], rgb[1], rgb[2], rg))
            bold = _is_bold(font)
            w = text_width(s, size, bold)
            cx = xp - {"start": 0.0, "middle": 0.5 * w, "end": w}[anchor]
            for t, role in text_runs(s):
                f, shift = _RUN_STYLE[role]
                for fn, seg in _font_segments(t, bold, "latin-1" if eps else "cp1252"):
                    if fn == "G":                   # outlined glyph, filled with the text colour
                        adv, path = _glyph_outline(seg, bold)
                        ops = _outline_ops(path, cx, yp - shift * size, f * size / 1000.0, flip=True)
                        o.append(" ".join("".join("%.2f " % v for v in op[1:]) + _PS_PATH_OPS[op[0]]
                                          for op in ops) + " f")
                        cx += f * size * adv / 1000.0
                        continue
                    if eps:
                        o.append("/%s findfont %.2f scalefont setfont %.2f %.2f moveto (%s) show"
                                 % (_PS_FONTS[fn], f * size, cx, yp - shift * size, _ps_escape(seg)))
                    else:
                        o.append("BT /%s %.2f Tf %.2f %.2f Td (%s) Tj ET"
                                 % (fn, f * size, cx, yp - shift * size, _ps_escape(seg)))
                    cx += f * size * _segment_width(fn, seg, bold) / 1000.0
        elif it[0] == "image":
            _, img, ix, iy, iw, ih = it
            images.append(img)
            x0p, y0p, wp, hp = ix * k, Hpt - (iy + ih) * k, iw * k, ih * k
            if not eps:
                o.append("q %.2f 0 0 %.2f %.2f %.2f cm /Im%d Do Q" % (wp, hp, x0p, y0p, len(images) - 1))
            else:
                h, w, _ = img.shape
                o.append("gsave %.2f %.2f translate %.2f %.2f scale %d %d 8 [%d 0 0 %d 0 %d] currentfile "
                         "/ASCIIHexDecode filter false 3 colorimage" % (x0p, y0p, wp, hp, w, h, w, -h, h))
                o.append(img.tobytes().hex() + ">")
                o.append("grestore")
    return "\n".join(o), images


def write_pdf(dl, path):
    k = 72.0 / dl.dpi
    content, images = _content_stream(dl)
    data = zlib.compress(content.encode("latin-1", "replace"), 6)
    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    xobj = " ".join("/Im%d %d 0 R" % (i, 8 + i) for i in range(len(images)))
    objs.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.2f %.2f] /Contents 4 0 R "
                 "/Resources << /Font << /F1 5 0 R /F2 6 0 R /F3 7 0 R >> /XObject << %s >> >> >>"
                 % (dl.W * k, dl.H * k, xobj)).encode())
    objs.append(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(data) + data + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Symbol >>")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    for img in images:
        h, w, _ = img.shape
        d = zlib.compress(img.tobytes(), 6)
        objs.append(("<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB "
                     "/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n" % (w, h, len(d))).encode()
                    + d + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offs = []
    for i, ob in enumerate(objs):
        offs.append(len(out))
        out += b"%d 0 obj\n" % (i + 1) + ob + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for of in offs:
        out += b"%010d 00000 n \n" % of
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info << /Producer (solid_lineview %s) >> >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, __version__.encode(), xref)
    with open(path, "wb") as fh:
        fh.write(out)


def write_eps(dl, path):
    k = 72.0 / dl.dpi
    content, _ = _content_stream(dl, eps=True)
    defs = "/m {moveto} bind def /l {lineto} bind def /S {stroke} bind def /h {closepath} bind def\n" \
           "/RG {setrgbcolor} bind def /rg {setrgbcolor} bind def /w {setlinewidth} bind def\n" \
           "/J {setlinecap} bind def /j {setlinejoin} bind def /d {setdash} bind def\n" \
           "/re {4 2 roll moveto 1 index 0 rlineto 0 exch rlineto neg 0 rlineto closepath} bind def\n" \
           "/f {fill} bind def /c {curveto} bind def\n" \
           "/L1enc ISOLatin1Encoding dup length array copy\n" \
           "  dup 45 /hyphen put dup 39 /quotesingle put dup 96 /grave put def\n" \
           "/latin1 {findfont dup length dict begin {1 index /FID ne {def} {pop pop} ifelse} forall\n" \
           "  /Encoding L1enc def currentdict end definefont pop} bind def\n" \
           "/Helvetica-L1 /Helvetica latin1 /Helvetica-Bold-L1 /Helvetica-Bold latin1\n"
    content = content.replace(" m\n", " m\n")
    # PostScript needs 'newpath' only before fills built with re; moveto starts a new subpath anyway
    with open(path, "w", encoding="latin-1", errors="replace") as fh:
        fh.write("%%!PS-Adobe-3.0 EPSF-3.0\n%%%%BoundingBox: 0 0 %d %d\n%%%%HiResBoundingBox: 0 0 %.2f %.2f\n"
                 "%%%%Creator: solid_lineview %s\n%%%%EndComments\n" % (
                     math.ceil(dl.W * k), math.ceil(dl.H * k), dl.W * k, dl.H * k, __version__))
        fh.write(defs)
        fh.write(content.replace("re f\n", "re fill\n"))
        fh.write("\nshowpage\n%%EOF\n")


def write_png(dl, path, backend="auto"):
    if backend in ("auto", "cairo"):
        try:
            import cairo  # noqa: F401
            return _png_cairo(dl, path)
        except ImportError:
            if backend == "cairo":
                raise
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise RuntimeError("PNG output needs pycairo or matplotlib")
    return _png_matplotlib(dl, path)


def _png_cairo(dl, path):
    import cairo
    W, H = dl.W, dl.H
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    cr = cairo.Context(surf)
    cr.set_source_rgb(*(dl.bg or (1, 1, 1)))
    cr.paint()
    cairo_draw_items(cr, dl.items, dl.dpi)
    if dl.bg is None:                       # transparent: cairo's own (premultiplied) RGBA writer
        surf.write_to_png(path)
        return
    surf.flush()
    bgra = np.ndarray((H, W, 4), np.uint8, surf.get_data(), 0, (surf.get_stride(), 4, 1))
    rgb = bgra[:, :, 2::-1] if sys.byteorder == "little" else bgra[:, :, 1:]
    grey = bool((rgb[:, :, 0] == rgb[:, :, 1]).all() and (rgb[:, :, 1] == rgb[:, :, 2]).all())
    with open(path, "wb") as fh:
        fh.write(_png_encode(rgb[:, :, 0] if grey else rgb, dl.dpi))


def cairo_draw_items(cr, items, dpi):
    """Draw DrawList items on a cairo context whose user space is canvas pixels (what the PNG writer
    does; solid_evdisplay draws animation frames with it)."""
    import cairo
    pt = dpi / 72.0
    cr.set_line_join(cairo.LINE_JOIN_ROUND)
    cr.set_font_options(_cairo_font_options(cairo))
    for it in items:
        if it[0] == "lines":
            _, pls, rgb, lw, dash, cap = it
            cr.set_source_rgb(*rgb)
            cr.set_line_width(lw * pt)
            cr.set_line_cap(cairo.LINE_CAP_ROUND if cap == "round" else cairo.LINE_CAP_BUTT)
            cr.set_dash([v * pt for v in dash] if dash else [])
            for p in pls:
                cr.move_to(*p[0])
                for q in p[1:]:
                    cr.line_to(*q)
                cr.stroke()
        elif it[0] == "fill":
            _, rings, rgb, stroke, lw, alpha = it
            cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
            for r in rings:
                cr.move_to(*r[0])
                for q in r[1:]:
                    cr.line_to(*q)
                cr.close_path()
            cr.set_source_rgba(rgb[0], rgb[1], rgb[2], alpha)
            if stroke is not None and lw > 0:
                cr.fill_preserve()
                cr.set_source_rgb(*stroke)
                cr.set_line_width(lw * pt)
                cr.set_dash([])
                cr.stroke()
            else:
                cr.fill()
        elif it[0] == "marker":
            _, kind, x, y, r, rgb = it
            cr.set_source_rgb(*rgb)
            if kind == "circle":
                cr.arc(x, y, r, 0, TWO_PI)
            else:
                cr.rectangle(x - r, y - r, 2 * r, 2 * r)
            cr.fill()
        elif it[0] == "text":
            _, x, y, s, size, rgb, anchor, font = it
            bold = _is_bold(font)
            # 'Helvetica' resolves (fontconfig) to Nimbus Sans / Liberation Sans / Arial: Helvetica metrics;
            # characters that font lacks (ψ, − in the X11 bitmap Helvetica of some images) are outlines
            cr.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL,
                                cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL)
            pieces = []                     # (text or outline, font px, baseline shift px, advance px)
            for t, role in text_runs(s):
                f, shift = _RUN_STYLE[role]
                fs = f * size * pt
                cr.set_font_size(fs)
                glyphs = cr.get_scaled_font().text_to_glyphs(0.0, 0.0, t, False)
                missing = [g.index == 0 for g in glyphs] if len(glyphs) == len(t) else [False] * len(t)
                for c, miss in zip(t, missing):
                    g = _glyph_outline(c, bold) if miss else None
                    if g is not None:
                        pieces.append((g[1], fs, shift * size * pt, g[0] * fs / 1000.0))
                    elif pieces and isinstance(pieces[-1][0], str) and pieces[-1][1:3] == (fs, shift * size * pt):
                        pieces[-1] = (pieces[-1][0] + c,) + pieces[-1][1:3] + (None,)
                    else:
                        pieces.append((c, fs, shift * size * pt, None))
            for i, (p, fs, dy, a) in enumerate(pieces):
                if a is None:
                    cr.set_font_size(fs)
                    pieces[i] = (p, fs, dy, cr.text_extents(p).x_advance)
            total = sum(pc[3] for pc in pieces)
            cx = x - {"start": 0.0, "middle": 0.5 * total, "end": total}[anchor]
            cr.set_source_rgb(*rgb)
            for p, fs, dy, a in pieces:
                if isinstance(p, str):
                    cr.set_font_size(fs)
                    cr.move_to(cx, y + dy)
                    cr.show_text(p)
                else:
                    cr.new_path()
                    for op in _outline_ops(p, cx, y + dy, fs / 1000.0):
                        if op[0] == "Z":
                            cr.close_path()
                        else:
                            {"M": cr.move_to, "L": cr.line_to, "C": cr.curve_to}[op[0]](*op[1:])
                    cr.fill()
                cx += a
        elif it[0] == "image":
            _, img, ix, iy, iw, ih = it
            h, w, _ = img.shape
            argb = np.empty((h, w, 4), np.uint8)
            argb[:, :, 0] = img[:, :, 2]
            argb[:, :, 1] = img[:, :, 1]
            argb[:, :, 2] = img[:, :, 0]
            argb[:, :, 3] = 255
            stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_ARGB32, w)
            buf = np.zeros((h, stride // 4, 4), np.uint8)
            buf[:, :w] = argb
            isurf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_ARGB32, w, h, stride)
            cr.save()
            cr.translate(ix, iy)
            cr.scale(iw / w, ih / h)
            cr.set_source_surface(isurf, 0, 0)
            cr.paint()
            cr.restore()


def _png_encode(img, dpi=None):
    """PNG of an 8-bit grey (H, W) or RGB (H, W, 3) image: per-row adaptive filter (the libpng
    minimum-sum-of-absolute-differences heuristic), zlib level 9, pHYs chunk with the dpi."""
    h, w = img.shape[:2]
    bpp = 1 if img.ndim == 2 else img.shape[2]
    x = np.ascontiguousarray(img, np.uint8).reshape(h, w * bpp).astype(np.int16)
    a = np.zeros_like(x)
    a[:, bpp:] = x[:, :-bpp]                                # left
    b = np.zeros_like(x)
    b[1:] = x[:-1]                                          # up
    c = np.zeros_like(x)
    c[1:, bpp:] = x[:-1, :-bpp]                             # upper left
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    paeth = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
    filt = np.stack([x, x - a, x - b, x - ((a + b) >> 1), x - paeth]) & 0xFF
    best = np.abs(filt.astype(np.uint8).view(np.int8).astype(np.int32)).sum(2).argmin(0)
    raw = np.concatenate([best[:, None], filt[best, np.arange(h)]], 1).astype(np.uint8).tobytes()

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0 if bpp == 1 else 2, 0, 0, 0))
    if dpi:
        ppm = int(round(dpi / 0.0254))
        out += chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
    return out + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _png_matplotlib(dl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, Circle, Rectangle
    W, H, dpi = dl.W, dl.H, dl.dpi
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor(dl.bg or (1, 1, 1))
    z = 1
    for it in dl.items:
        z += 1
        if it[0] == "lines":
            _, pls, rgb, lw, dash, cap = it
            lc = LineCollection(pls, colors=[rgb], linewidths=lw, capstyle="round" if cap == "round" else "butt",
                                joinstyle="round", zorder=z)
            if dash:
                lc.set_linestyle((0, dash))
            ax.add_collection(lc)
        elif it[0] == "fill":
            _, rings, rgb, stroke, lw, alpha = it
            verts, codes = [], []
            for r in rings:
                verts.extend(r.tolist() + [r[0].tolist()])
                codes.extend([Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY])
            pp = PathPatch(Path(verts, codes), facecolor=rgb, alpha=alpha, zorder=z,
                           edgecolor=stroke if stroke is not None else "none",
                           linewidth=lw if stroke is not None else 0)
            ax.add_patch(pp)
        elif it[0] == "marker":
            _, kind, x, y, r, rgb = it
            p = Circle((x, y), r, color=rgb, zorder=z) if kind == "circle" else \
                Rectangle((x - r, y - r), 2 * r, 2 * r, color=rgb, zorder=z)
            ax.add_patch(p)
        elif it[0] == "text":
            _, x, y, s, size, rgb, anchor, font = it
            ax.text(x, y, s, fontsize=size, color=rgb, ha={"start": "left", "middle": "center", "end": "right"}[anchor],
                    va="baseline", zorder=z, fontweight="bold" if _is_bold(font) else "normal")
        elif it[0] == "image":
            _, img, ix, iy, iw, ih = it
            ax.imshow(img, extent=(ix, ix + iw, iy + ih, iy), interpolation="bilinear", zorder=z)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_outputs(dl, outputs, png_backend):
    for path in outputs:
        ext = os.path.splitext(path)[1].lower()
        with timed("write %s" % path):
            if ext == ".svg":
                write_svg(dl, path)
            elif ext == ".pdf":
                write_pdf(dl, path)
            elif ext in (".eps", ".ps"):
                write_eps(dl, path)
            elif ext == ".png":
                write_png(dl, path, png_backend)
            else:
                raise SystemExit("unknown output format: %s" % path)
        log("wrote %s (%.1f kB)" % (path, os.path.getsize(path) / 1024.0))


# =============================================================================
#  12.  DAWN compatibility helpers
# =============================================================================

def read_dawn_history(path):
    """.DAWN_1.history (written by G4DAWNFILEViewer::SendViewParameters / dawn_tweak)."""
    with open(path) as fh:
        L = [x.strip() for x in fh.read().splitlines()]

    def f(i, default=None):
        try:
            return float(L[i - 1])
        except (IndexError, ValueError):
            return default

    h = {"camera_distance": f(1), "polar": f(2, 0.0), "azimuth": f(3, 0.0),
         "auto_target": f(4, 0.0), "target": (f(5, 0.0), f(6, 0.0), f(7, 0.0)),
         "style": int(f(9, 4) or 4), "aux_threshold": f(24, 70.0)}
    mag = L[7] if len(L) > 7 else "1"
    h["mag"] = None if mag.startswith("fd") else float(mag)
    h["focal_distance"] = float(mag[2:]) if mag.startswith("fd") else None
    return h


DAWN_STYLE_TO_MODE = {1: "wireframe", 2: "wireframe", 3: "surface", 4: "hlr", 5: "surface"}


def dawncut_to_plane(a, b, c, d):
    """DAWNCUT 1.54a semantics (verified in its source): unit normal n=(a,b,c)/|..|, the
    part with n.P - d > 0 is erased.  Returned in solid_lineview convention (keep >= 0)."""
    n = np.array([a, b, c], float)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise SystemExit("dawncut: zero normal")
    n /= nn
    return (-n, float(d))


def plane_from_abcd(a, b, c, d):
    n = np.array([a, b, c], float)
    nn = np.linalg.norm(n)
    if nn == 0:
        raise SystemExit("zero plane normal")
    return (n / nn, float(d) / nn)


def parse_section_spec(spec):
    """'x=0', 'y=-10cm', 'z=2.5m' -> plane (n, d) with n along +axis."""
    m = re.fullmatch(r"\s*([xyz])\s*=\s*(.+)", spec)
    if not m:
        raise argparse.ArgumentTypeError("section spec must look like x=0 / z=2.5m")
    ax = "xyz".index(m.group(1))
    n = np.zeros(3)
    n[ax] = 1.0
    return (n, -parse_length(m.group(2)))


def parse_slab(spec):
    """'y=-1:1' (mm) or 'x=-5cm:5cm' -> two keep-planes."""
    m = re.fullmatch(r"\s*([xyz])\s*=\s*([^:]+):(.+)", spec)
    if not m:
        raise argparse.ArgumentTypeError("slab spec must look like y=-1:1")
    ax = "xyz".index(m.group(1))
    lo, hi = parse_length(m.group(2)), parse_length(m.group(3))
    n = np.zeros(3)
    n[ax] = 1.0
    return [(n.copy(), -lo), (-n, hi)]


def parse_trim(spec):
    """'Beamline=auto', 'ExitPipe=z<5m', 'Pipe=z>-3.5m,z<5m' -> (regex, [terms]); a term is
    'auto' or a keep-plane (n, d)."""
    rx, sep, rest = spec.rpartition("=")
    if not sep or not rx:
        raise argparse.ArgumentTypeError("trim must look like REGEX=auto or REGEX=z<5m")
    try:
        crx = re.compile(rx)
    except re.error as exc:
        raise argparse.ArgumentTypeError("bad trim regex '%s': %s" % (rx, exc))
    terms = []
    for t in rest.split(","):
        t = t.strip()
        if t == "auto":
            terms.append("auto")
            continue
        m = re.fullmatch(r"([xyz])\s*([<>])\s*(.+)", t)
        if not m:
            raise argparse.ArgumentTypeError("trim term '%s' must be auto, AXIS<LEN or AXIS>LEN" % t)
        n = np.zeros(3)
        v = parse_length(m.group(3))
        n["xyz".index(m.group(1))] = -1.0 if m.group(2) == "<" else 1.0
        terms.append((n, v if m.group(2) == "<" else -v))
    return crx, terms


def parse_label(spec):
    """'REGEX=Text' (split at the first '=')."""
    rx, sep, txt = spec.partition("=")
    if not sep or not rx:
        raise argparse.ArgumentTypeError("label must look like REGEX=Text")
    try:
        re.compile(rx)
    except re.error as exc:
        raise argparse.ArgumentTypeError("bad label regex '%s': %s" % (rx, exc))
    return rx, txt.strip()


# =============================================================================
#  13.  CLI
# =============================================================================

EPILOG = r"""
examples (scene.npz and g4_XXXX.prim are interchangeable as input)
  # exact y-z section at x=0 from the scene written by `solid_scene_dump -o scene.npz`
  solid_lineview scene.npz --mode section --view side --fill -o section_yz.svg

  # DAWN-like hidden-line side view of the half detector x>0 (camera at -x), PDF + PNG
  solid_lineview g4_0000.prim --view side --cut 1 0 0 0 -o side.pdf -o side.png

  # exact y-z section at x=0, coloured fills
  solid_lineview g4_0000.prim --mode section --view side --fill -o section_yz.svg

  # what the old views/top_view/generate_eps of this repository did (side view part):
  #   dawncut -1 0 0 1 ; dawn_tweak --mag 10 --draw 3 --theta 165 --phi 75
  solid_lineview g4_0000.prim --dawncut -1 0 0 1 --dawn-angles 75 165 --mode hlr -o view.pdf

  # read camera + style from an existing .DAWN_1.history
  solid_lineview g4_0000.prim --dawn-history .DAWN_1.history -o top.pdf

  # labelled detector overview: cutaway, beam line trimmed to the detector, beam arrow, SoLID labels
  solid_lineview g4_0000.prim --dawncut -1 0 0 0 --view-dir -0.8 0.3 -0.52 --perspective 40m \
      --trim 'Beamline=auto' --beam-arrow --labels solid --title 'SoLID-Lite J/ψ' -o overview.svg

conventions
  lengths are mm unless a unit is given (e.g. 2.5m, -30cm); --cut a b c d keeps a*x+b*y+c*z+d >= 0
  (Geant4 /vis/viewer/addCutawayPlane convention); --dawncut a b c d reproduces DAWNCUT 1.54a exactly
  (erases the side n.x > d, n = unit(a,b,c)).  View presets: side (camera at -x, z right, y up),
  side-x, top (camera at +y, z right, x up), bottom, beam (camera at +z: x right, y up), front, iso.
"""


def build_parser():
    p = argparse.ArgumentParser(
        prog="solid_lineview",
        description="Clean DAWN-style vector line drawings (SVG/PDF/EPS/PNG) of Geant4 geometry: from a scene.npz "
                    "(solid_scene_dump, or ddsim with SOLID_SCENE) or from a DAWNFILE .prim file.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prim", metavar="scene", help="input scene: scene.npz (Geant4 polyhedra with names and volume "
                   "IDs) or a .prim file (Geant4 DAWNFILE output, g4.prim format 2.4); told apart by content")
    p.add_argument("-o", "--output", action="append", default=[],
                   help="output file(s); format from extension: .svg .pdf .eps .png (repeatable)")
    p.add_argument("--mode", choices=["hlr", "section", "wireframe", "surface"], default="hlr")
    g = p.add_argument_group("camera")
    g.add_argument("--view", default="side", help="preset: %s (default side)" % ", ".join(VIEW_PRESETS))
    g.add_argument("--dawn-angles", nargs=2, type=float, metavar=("POLAR", "AZIMUTH"),
                   help="DAWN camera angles in deg (= .DAWN_1.history lines 2,3 = dawn_tweak --phi/--theta "
                        "= Geant4 viewpointThetaPhi)")
    g.add_argument("--view-dir", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
                   help="direction from the target towards the camera")
    g.add_argument("--up", nargs=3, type=float, default=None, metavar=("UX", "UY", "UZ"),
                   help="up vector (default +y)")
    g.add_argument("--target", nargs=3, type=parse_length, metavar=("X", "Y", "Z"),
                   help="target point (default: centre of the drawn geometry)")
    g.add_argument("--perspective", type=parse_length, metavar="DIST",
                   help="perspective projection with the camera at this distance from the target")
    g.add_argument("--mag", type=float, help="DAWN-compatible magnification relative to the /BoundingBox fit "
                                             "(disables auto-fit and auto-crop)")
    g.add_argument("--dawn-history", metavar="FILE", help="take camera, mag, style and aux-edge angle from a "
                                                          ".DAWN_1.history file")
    g.add_argument("--fit-box", nargs=4, type=parse_length, metavar=("X0", "X1", "Y0", "Y1"),
                   help="fit this window (screen axes, mm, relative to the target) instead of the content; "
                        "sections and line drawings are clipped to it")
    g.add_argument("--fit-volumes", metavar="REGEX",
                   help="fit the view to the box around the volumes whose name matches, e.g. '(^|!)Target' for "
                        "a close-up; everything else inside the window is drawn too, clipped at its edge "
                        "(sections and line drawings)")
    g.add_argument("--fit-pad", type=parse_length, default=0.0, metavar="LEN",
                   help="widen the fitted window by LEN on every side, e.g. 1m")
    g = p.add_argument_group("cuts and sections")
    g.add_argument("--cut", nargs=4, type=float, action="append", metavar=("A", "B", "C", "D"),
                   help="keep a*x+b*y+c*z+d >= 0 (mm); repeatable (intersection of half spaces)")
    g.add_argument("--dawncut", nargs=4, type=float, action="append", metavar=("A", "B", "C", "D"),
                   help="DAWNCUT semantics: erase n.x > d with n = unit(a,b,c); repeatable")
    g.add_argument("--slab", action="append", type=parse_slab, metavar="AXIS=LO:HI",
                   help="keep LO <= axis <= HI (e.g. y=-1:1), like the dawncut pair in generate_eps")
    g.add_argument("--section", type=parse_section_spec, metavar="AXIS=VALUE",
                   help="section plane for --mode section (default: from the view preset through the target)")
    g.add_argument("--plane", nargs=4, type=float, metavar=("A", "B", "C", "D"),
                   help="general section plane a*x+b*y+c*z+d=0 for --mode section")
    g.add_argument("--plane-snap", type=parse_length, default=1e-3, metavar="LEN",
                   help="vertices closer than LEN to a cut or section plane are moved off it by LEN, each solid "
                        "to one side, so that faces lying in the plane (sector sides at x = 0) do not depend on "
                        "rounding noise: a solid touching a section plane with a face is cut along that face, "
                        "a solid touching a cut plane from the removed side is removed (default 1e-3 mm, "
                        "above the coordinate precision of a DAWNFILE .prim; 0 = off, as versions before 0.5.0)")
    g.add_argument("--trim", action="append", type=parse_trim, metavar="REGEX=SPEC",
                   help="cut only the volumes matching REGEX; SPEC is 'auto' (z range of all other volumes, "
                        "+- --trim-margin) or a comma list of AXIS<LEN / AXIS>LEN, e.g. 'Beamline=z<5m'; repeatable")
    g.add_argument("--trim-margin", type=float, default=0.03,
                   help="margin of --trim auto as a fraction of the z length of the other volumes (default 0.03)")
    g.add_argument("--no-caps", action="store_true", help="do not close cut solids with caps (hlr/surface)")
    g.add_argument("--cap-fill", default="none", help="fill of cut faces in hlr/surface: none (default), white, "
                                                    "color, or a colour")
    g = p.add_argument_group("selection")
    g.add_argument("--include", action="append", metavar="REGEX", help="only volumes whose PV name matches")
    g.add_argument("--exclude", action="append", metavar="REGEX", help="drop volumes whose PV name matches")
    g.add_argument("--transparent", action="append", metavar="REGEX",
                   help="draw these volumes but let them hide nothing (like ForceWireframe)")
    g.add_argument("--match-path", action="store_true", help="match regexes against the volume path "
                                                           "(mother/daughter) instead of the name")
    g.add_argument("--min-depth", type=int, help="drop volumes shallower than this hierarchy depth")
    g.add_argument("--max-depth", type=int, help="drop volumes deeper than this hierarchy depth")
    g.add_argument("--exclude-color", action="append", type=parse_color, metavar="RGB")
    g.add_argument("--only-color", action="append", type=parse_color, metavar="RGB")
    g.add_argument("--mothers", choices=["opaque", "transparent", "hide"], default="opaque",
                   help="volumes with drawn daughters: opaque (physical, default), transparent, hide")
    g.add_argument("--wireframe-volumes", choices=["transparent", "hide"], default="transparent",
                   help="volumes flagged /ForceWireframe 1 (default: drawn, non-occluding)")
    g.add_argument("--drawing-style", action="store_true",
                   help="scene.npz only: also treat volumes whose DD4hep VisAttr has drawingStyle=\"wireframe\" "
                        "as wireframe volumes (DDG4 does not pass drawingStyle to Geant4, so by default they "
                        "are drawn solid, as in a DAWNFILE .prim)")
    g = p.add_argument_group("edges and style")
    g.add_argument("--crease", type=float, default=35.0,
                   help="feature-edge angle in deg between face normals (default 35; DAWN uses 70)")
    g.add_argument("--no-silhouettes", action="store_true")
    g.add_argument("--all-edges", action="store_true", help="wireframe: draw every polygon edge")
    g.add_argument("--lw", type=float, default=0.35, help="line width in pt (default 0.35)")
    g.add_argument("--lw-outline", type=float, default=None, help="silhouette line width (default = --lw)")
    g.add_argument("--lw-cut", type=float, default=None, help="cut/section outline width (default 1.4 x --lw)")
    g.add_argument("--depth-lw-factor", type=float, default=1.0,
                   help="multiply the width by this factor per hierarchy level (e.g. 0.8)")
    g.add_argument("--color-lines", action="store_true", help="draw lines in the volume colours")
    g.add_argument("--line-color", type=parse_color, default=(0, 0, 0), help="monochrome line colour")
    g.add_argument("--line-color-scale", type=float, default=0.8, help="darken volume colours for lines")
    g.add_argument("--fill", action="store_true", help="section mode: fill outlines with the volume colour")
    g.add_argument("--fill-lighten", type=float, default=0.35, help="blend fills towards white (0..1)")
    g.add_argument("--fill-alpha", type=float, default=1.0)
    g.add_argument("--background", type=parse_color, default=(1, 1, 1), help="'none' for transparent")
    g.add_argument("--light", nargs=3, type=float, default=(-0.35, 0.55, 0.75), metavar=("R", "U", "E"),
                   help="surface mode light direction in camera axes (right, up, towards camera)")
    g.add_argument("--ambient", type=float, default=0.35)
    g.add_argument("--surface-vector", action="store_true",
                   help="surface mode: painter-sorted vector polygons instead of an embedded raster")
    g = p.add_argument_group("trajectories, hits, text")
    g.add_argument("--no-traj", action="store_true")
    g.add_argument("--no-hits", action="store_true")
    g.add_argument("--no-text", action="store_true")
    g.add_argument("--traj-lw", type=float, default=0.6)
    g.add_argument("--traj-color", type=parse_color, default=None)
    g.add_argument("--traj-hlr", action="store_true", help="hide trajectory parts behind solids")
    g.add_argument("--clip-traj", action="store_true", help="apply the cut planes to trajectories/hits too")
    g.add_argument("--fit-traj", action="store_true", help="include trajectories in the auto-fit")
    g.add_argument("--marker-scale", type=float, default=1.0)
    g.add_argument("--mono-text", action="store_true")
    g = p.add_argument_group("canvas and decorations")
    g.add_argument("--size", default="3000x2000", help="canvas in px (default 3000x2000); the content is "
                                                     "fitted and the canvas cropped to it unless --no-crop")
    g.add_argument("--dpi", type=float, default=300.0, help="px per inch for vector page size and line widths")
    g.add_argument("--margin", type=float, default=0.03, help="margin as a fraction of the smaller side")
    g.add_argument("--no-crop", action="store_true")
    g.add_argument("--hlr-scale", type=float, default=1.0, help="depth-buffer resolution / canvas resolution")
    g.add_argument("--sample-step", type=float, default=0.5, help="edge sampling step in depth-buffer px")
    g.add_argument("--slope-tol", type=float, default=1.0, help="depth tolerance in units of the occluder's "
                                                              "depth change per px")
    g.add_argument("--no-cull", action="store_true", help="rasterise back faces too")
    g.add_argument("--keep-overdraw", action="store_true",
                   help="draw lines that lie on top of each other more than once, as before 0.6.0 "
                        "(default: overlapping parts of lines on one straight line are drawn once)")
    g.add_argument("--scale-bar", type=parse_length, nargs="?", const=0.0, default=None, metavar="LEN",
                   help="draw a scale bar (auto length if no value)")
    g.add_argument("--axes", action="store_true", help="draw a small x/y/z axis triad")
    g.add_argument("--title", default=None, help="title; Unicode sub/superscripts and Greek are typeset "
                                                 "(e.g. 'J/ψ, NH₃ target')")
    g.add_argument("--title-size", type=float, default=None, help="title size in pt (default 1.3 x --font-size)")
    g.add_argument("--title-bold", action="store_true")
    g.add_argument("--subtitle", default=None, help="second, smaller grey line under the title")
    g.add_argument("--font-size", type=float, default=9.0)
    g.add_argument("--png-backend", choices=["auto", "cairo", "matplotlib"], default="auto")
    g = p.add_argument_group("magnetic field (section mode)")
    g.add_argument("--field", metavar="FILE",
                   help="field drawn under the section (|B| in colour bands, field lines, direction arrows): "
                        "field.npz of solid_field_dump (the total field of a configuration on this plane) or a "
                        "GEMC 2-D field map (R Z Br Bz, axisymmetric about z)")
    g.add_argument("--field-scale", type=float, default=1.0, metavar="F",
                   help="factor applied to the map, as the scale attribute of the field fragment (-1 flips it)")
    g.add_argument("--field-lines", type=int, default=24, metavar="N",
                   help="number of field lines (default 24, 0 = none). Where the in-plane field is axisymmetric "
                        "about z: N lines with equal flux between neighbours (needs the python module "
                        "contourpy). Otherwise lines along the field direction, spaced by 1/N of the height")
    g.add_argument("--field-lw", type=float, default=0.45, help="field line width in pt (default 0.45)")
    g.add_argument("--field-arrows", type=parse_length, default=500.0, metavar="STEP",
                   help="spacing of the field-direction arrows (default 50cm, 0 = none)")
    g.add_argument("--field-iron", default=r"(^|!)MagnetYoke", metavar="REGEX",
                   help="volumes that are iron: no field lines or arrows inside, and the flux function is not "
                        "integrated through them (default '(^|!)MagnetYoke')")
    g = p.add_argument_group("labels and beam arrow")
    g.add_argument("--labels", choices=sorted(LABEL_PRESETS), default=None,
                   help="built-in label set; 'solid': Target, Yoke endcap, CLEO-II yoke, Coil + cryostat, "
                        "Baffles, GEM, LGC, HGC, CO2 Cherenkov, LAEC, LASPD, FAEC, FASPD, MRPC, Muon system "
                        "(only those present and visible are drawn)")
    g.add_argument("--label", action="append", type=parse_label, metavar="REGEX=TEXT",
                   help="callout label for the volumes whose PV name (or path with --match-path) matches REGEX; "
                        "repeatable, first match wins; replaces the preset entry with the same REGEX, an "
                        "empty TEXT removes it")
    g.add_argument("--label-size", type=float, default=None, help="label size in pt (default --font-size)")
    g.add_argument("--label-rows", choices=["both", "top", "bottom"], default="both",
                   help="label rows above and/or below the drawing (default both)")
    g.add_argument("--label-color", type=parse_color, default=(0.0, 0.0, 0.0), help="leader and dot colour")
    g.add_argument("--label-lw", type=float, default=0.5, help="leader width in pt (default 0.5)")
    g.add_argument("--label-dot", type=float, default=1.4, help="anchor dot radius in pt (default 1.4)")
    g.add_argument("--beam-arrow", nargs="?", const="e⁻ beam", default=None, metavar="TEXT",
                   help="arrow along +z on the beam axis, upstream of the drawing, labelled TEXT "
                        "(default 'e⁻ beam'; '' for none)")
    g.add_argument("--beam-arrow-length", type=float, default=0.15,
                   help="beam arrow length as a fraction of the drawn z length (default 0.15)")
    g = p.add_argument_group("information")
    g.add_argument("--info", action="store_true", help="print the volume tree and statistics, no drawing")
    g.add_argument("--profile", action="store_true", help="print stage timings")
    g.add_argument("-v", "--verbose", action="count", default=0)
    g.add_argument("-q", "--quiet", action="store_true")
    g.add_argument("--version", action="version", version="solid_lineview " + __version__)
    return p


def trimmed_centre(sc, o):
    """Default camera target with --trim: centre of the selected volumes, trimmed ones clipped
    to their axis-aligned trim planes (a long beam pipe must not pull the camera downstream)."""
    rd = Renderer.__new__(Renderer)
    rd.sc, rd.o = sc, o
    sel, _ = rd.select_solids()
    trims = rd.resolve_trims(sel, quiet=True)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for s in sc.solids:
        if not sel[s.index]:
            continue
        b0, b1 = s.bmin.copy(), s.bmax.copy()
        for n, d, m in trims:
            k = int(np.argmax(np.abs(n)))
            if m[s.index] and abs(n[k]) > 0.999:
                if n[k] > 0:
                    b0[k] = max(b0[k], -d / n[k])
                else:
                    b1[k] = min(b1[k], d / -n[k])
        if np.all(b1 >= b0):
            lo, hi = np.minimum(lo, b0), np.maximum(hi, b1)
    return 0.5 * (lo + hi) if np.all(np.isfinite(lo)) else 0.5 * (sc.V.min(0) + sc.V.max(0))


def print_info(sc, mesh):
    print("solids: %d   vertices: %d   facets: %d   (closed: %d)" % (
        len(sc.solids), len(sc.V), len(sc.F), sum(s.closed for s in sc.solids)))
    print("kinds:", ", ".join("%s=%d" % kv for kv in sorted(sc.counts.items())))
    if sc.bbox is not None:
        print("/BoundingBox:", " ".join("%.6g" % v for v in sc.bbox))
    if len(sc.V):
        print("geometry extent [mm]: x %.1f..%.1f  y %.1f..%.1f  z %.1f..%.1f" % (
            sc.V[:, 0].min(), sc.V[:, 0].max(), sc.V[:, 1].min(), sc.V[:, 1].max(),
            sc.V[:, 2].min(), sc.V[:, 2].max()))
    print("edges: %d (feature %d, boundary %d, non-manifold %d, inconsistent orientation %d)" % (
        len(mesh.e0), int(mesh.efeature.sum()), int(mesh.eboundary.sum()), int(mesh.enonmanifold.sum()),
        mesh.n_inconsistent))
    if sc.clip_comments:
        print("DAWNCUT planes recorded in file:", sc.clip_comments)
    for w, c in sc.warnings.most_common():
        print("warning (%dx): %s" % (c, w))
    names = Counter()
    for s in sc.solids:
        names[(s.depth, s.name.rsplit(":", 1)[0], s.kind, s.color, s.wire, s.closed)] += 1
    print("\n%-5s %-6s %-28s %-12s %-17s %s" % ("depth", "count", "volume", "kind", "colour", "flags"))
    seen = set()
    for s in sc.solids:
        k = (s.depth, s.name.rsplit(":", 1)[0], s.kind, s.color, s.wire, s.closed)
        if k in seen:
            continue
        seen.add(k)
        flags = ("wire " if s.wire else "") + ("" if s.closed else "open")
        print("%-5d %-6d %-28s %-12s %-17s %s" % (s.depth, names[k], ("  " * max(s.depth - 1, 0)) + k[1],
                                                  s.kind, "%.2f,%.2f,%.2f" % s.color, flags))


def main(argv=None):
    global VERBOSE
    args = build_parser().parse_args(argv)
    VERBOSE = 0 if args.quiet else 1 + args.verbose + (1 if args.profile else 0)
    try:
        W, H = [int(v) for v in args.size.lower().split("x")]
    except ValueError:
        raise SystemExit("--size must look like 3000x2000")
    args.width, args.height = W, H
    if args.lw_outline is None:
        args.lw_outline = args.lw
    if args.lw_cut is None:
        args.lw_cut = 1.4 * args.lw

    t_start = time.perf_counter()
    with timed("parse %s" % args.prim):
        sc = load_scene(args.prim, drawing_style=args.drawing_style)
    log("parsed %s: %d solids, %d facets, %d polylines, %d markers" % (
        os.path.basename(args.prim), len(sc.solids), len(sc.F), len(sc.polylines), len(sc.markers)))
    for w, c in sc.warnings.most_common(10):
        log("  warning (%dx): %s" % (c, w))

    # ---- DAWN history -> defaults
    mode = args.mode
    if args.dawn_history:
        h = read_dawn_history(args.dawn_history)
        if args.dawn_angles is None:
            args.dawn_angles = (h["polar"], h["azimuth"])
        if args.target is None and not h["auto_target"]:
            args.target = h["target"]
        if args.mag is None and h["mag"] is not None:
            args.mag = h["mag"]
        if "--mode" not in (argv if argv is not None else sys.argv):
            mode = DAWN_STYLE_TO_MODE.get(h["style"], "hlr")
        log("  .DAWN_1.history: angles %.4g %.4g, target %s, mag %s, style %d -> mode %s" % (
            h["polar"], h["azimuth"], h["target"], h["mag"], h["style"], mode))

    if args.info:
        mesh = Mesh(sc, args.crease)
        print_info(sc, mesh)
        return 0
    if not args.output:
        raise SystemExit("no output given (-o file.svg|pdf|eps|png), or use --info")

    # ---- camera
    up = args.up if args.up is not None else (0, 1, 0)
    if args.dawn_angles is not None:
        eye = dawn_angles_to_eye(*args.dawn_angles)
        preset = None
    elif args.view_dir is not None:
        eye = _unit(args.view_dir)
        preset = None
    else:
        if args.view not in VIEW_PRESETS:
            raise SystemExit("unknown view preset %s" % args.view)
        eye, up0 = VIEW_PRESETS[args.view]
        if args.up is None:
            up = up0
        preset = args.view
    if args.target is not None:
        target = np.array(args.target, float)
    elif len(sc.V):
        target = 0.5 * (sc.V.min(0) + sc.V.max(0))
        if args.mag is not None and sc.bbox is not None:
            target = 0.5 * (sc.bbox[:3] + sc.bbox[3:])
        elif args.trim:
            target = trimmed_centre(sc, args)
    else:
        target = np.zeros(3)
    cam = Camera(eye, up, target, args.perspective)

    # ---- planes
    planes = []
    for c in args.cut or []:
        planes.append(plane_from_abcd(*c))
    for c in args.dawncut or []:
        planes.append(dawncut_to_plane(*c))
    for pair in args.slab or []:
        planes.extend(pair)
    cap_planes = [] if args.no_caps else list(planes)

    if mode == "section":
        if args.plane is not None:
            sec = plane_from_abcd(*args.plane)
        elif args.section is not None:
            sec = args.section
        else:
            ax = SECTION_FOR_VIEW.get(preset)
            if ax is None:
                ax = tuple(cam.e)
            n = _unit(ax)
            sec = (n, -float(n @ target) if args.target is not None else 0.0)
            if args.target is None:
                # through the origin of the hall frame, the natural choice for SoLID
                sec = (n, 0.0)
        log("section plane: n=(%.3g, %.3g, %.3g) d=%.4g mm" % (sec[0][0], sec[0][1], sec[0][2], sec[1]))
        with timed("render section"):
            dl, rd = render_section(sc, args, cam, sec, planes)
    else:
        rd = Renderer(sc, args)
        with timed("render %s" % mode):
            rd._fitinfo = None
            # _fit is needed by cap rasterisation (unprojection); compute it inside _finish
            orig_fit = rd._fit

            def fit_and_store(cam_, pts):
                r = orig_fit(cam_, pts)
                rd._fitinfo = r
                return r

            rd._fit = fit_and_store
            dl = rd.render(cam, planes, cap_planes, mode)
        log("  %d triangles after cuts, %d visible edge segments" % (rd.n_tri, rd.n_vis_segments))

    if args.beam_arrow is not None:
        place_beam_arrow(dl, rd, args)
    labels = label_list(args)
    if labels:
        with timed("place %d candidate labels" % len(labels)):
            place_labels(dl, rd, args, labels)
    if labels or args.beam_arrow is not None:
        finish_canvas(dl, args)
    draw_title(dl, args)
    if args.background is None:
        dl.bg = None
    write_outputs(dl, args.output, args.png_backend)
    log("total %.2f s" % (time.perf_counter() - t_start))
    if args.profile:
        print("\nstage timings:", file=sys.stderr)
        for lab, dt in _TIMINGS:
            print("  %8.3f s  %s" % (dt, lab), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
