#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test of the field views and of the line clean-up in solid_lineview (ctest: field_view).

usage: test_field_view.py [--tool PATH/solid_field_dump] [--scene-tool PATH/solid_scene_dump] [--config NAME]
                          [--workdir DIR]

Checks, in this order:
  lines     merge_collinear: lines drawn on top of each other are drawn once (duplicates, partial overlaps, the
            wider line wins, crossing and parallel lines are left alone, end points are input points);
            clip_segments_rect and clip_polygon_rect (window of --fit-volumes / --fit-box)
  dump      solid_field_dump on the configuration (default solid_JPsi_lite): grid on multiples of the step with
            the beam axis on it, and the same field as DD4hep's OverlayedField at sample points
  symmetry  the sampled solenoid field in the x = 0 plane is recognised as axisymmetric (equal-flux lines), a
            plane across the axis is not
  drawing   the x = 0 section drawn with the sampled field equals the one drawn with the GEMC map of the
            configuration (when that map is in $DETECTOR_PATH/fieldmaps; at most 0.3 % of the pixels may differ,
            along the edge of the map), and --fit-volumes clips the drawing to its window
The first block needs numpy only. The others need DETECTOR_PATH, the plugin libraries on LD_LIBRARY_PATH and the
dd4hep python module (thissolid.sh, or the ctest environment). Exit status 0 if every check passes.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'python'))
FAILED = []


def check(ok, what):
    print('%s  %s' % ('ok  ' if ok else 'FAIL', what))
    if not ok:
        FAILED.append(what)


def test_lines():
    from solid_lineview.core import clip_polygon_rect, clip_polyline_rect, clip_segments_rect, merge_collinear

    def run(segs, prio=None, style=None):
        s = np.array(segs, float)
        n = len(s)
        prio = np.zeros(n) if prio is None else np.array(prio, float)
        style = np.zeros(n, np.int64) if style is None else np.array(style, np.int64)
        x0, y0, x1, y1, src = merge_collinear(s[:, 0], s[:, 1], s[:, 2], s[:, 3], prio, style)
        return np.stack([x0, y0, x1, y1], 1), src

    def length(o):
        return float(np.hypot(o[:, 2] - o[:, 0], o[:, 3] - o[:, 1]).sum())

    o, _ = run([(0, 0, 10, 0), (0, 0, 10, 0), (10, 0, 0, 0)])
    check(len(o) == 1 and abs(length(o) - 10) < 1e-9, 'three copies of one segment are drawn once')
    o, _ = run([(0, 0, 10, 5), (4, 2, 16, 8)])
    check(len(o) == 1 and abs(length(o) - np.hypot(16, 8)) < 1e-9, 'two overlapping pieces of one line are joined')
    o, src = run([(0, 0, 10, 0), (5, 0, 15, 0)], prio=[0.5, 0.9], style=[0, 1])
    o = o[np.argsort(np.minimum(o[:, 0], o[:, 2]))]
    check(len(o) == 2 and np.allclose(np.sort(o[0, [0, 2]]), [0, 5]) and np.allclose(np.sort(o[1, [0, 2]]), [5, 15])
          and sorted(src.tolist()) == [0, 1], 'where two widths overlap the wider line is kept')
    o, _ = run([(0, 0, 10, 0), (20, 0, 30, 0)])
    check(len(o) == 2 and abs(length(o) - 20) < 1e-9, 'a gap between two pieces of one line stays open')
    o, _ = run([(0, 0, 10, 10), (0, 10, 10, 0), (0, 0.5, 10, 0.5), (0, 0, 10, 0)])
    check(len(o) == 4 and abs(length(o) - (2 * np.hypot(10, 10) + 20)) < 1e-9,
          'crossing lines and lines 0.5 px apart are left alone')
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 100, (40, 2))
    segs = [(p[i, 0], p[i, 1], p[i + 1, 0], p[i + 1, 1]) for i in range(39)]
    o, _ = run(segs + segs[::2])
    pts = {(round(a, 9), round(b, 9)) for a, b in np.concatenate([o[:, :2], o[:, 2:]]).tolist()}
    inp = {(round(a, 9), round(b, 9)) for a, b in p.tolist()}
    check(len(o) == 39 and pts <= inp and abs(length(o) - length(np.array(segs))) < 1e-9,
          'a polyline with half of its segments doubled comes back once, with its own corner points')

    x0, y0, x1, y1, src = clip_segments_rect(np.array([-5.0, 2.0, 20.0, 5.0]), np.array([5.0, 2.0, 5.0, -5.0]),
                                             np.array([5.0, 8.0, 30.0, 5.0]), np.array([5.0, 8.0, 5.0, 15.0]),
                                             (0.0, 0.0, 10.0, 10.0))
    check(src.tolist() == [0, 1, 3] and np.allclose([x0[0], x1[0]], [0, 5]) and np.allclose([y0[2], y1[2]], [0, 10]),
          'segments are clipped to a window (inside, crossing, outside)')
    pieces = clip_polyline_rect(np.array([[-5.0, 5.0], [5.0, 5.0], [5.0, 15.0], [8.0, 15.0], [8.0, 5.0], [20.0, 5.0]]),
                                (0.0, 0.0, 10.0, 10.0))
    check(len(pieces) == 2 and np.allclose(pieces[0], [[0, 5], [5, 5], [5, 10]])
          and np.allclose(pieces[1], [[8, 10], [8, 5], [10, 5]]), 'a polyline leaving and re-entering a window gives two pieces')
    sq = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    c = clip_polygon_rect(sq, (0.0, 0.0, 10.0, 10.0))
    area = 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], -1)) - np.dot(c[:, 1], np.roll(c[:, 0], -1)))
    check(abs(area - 25.0) < 1e-9 and len(clip_polygon_rect(sq + 100.0, (0.0, 0.0, 10.0, 10.0))) == 0,
          'polygons are clipped to a window')


def find_binary(name, given):
    for cand in (given or '', shutil.which(name) or '', os.path.join(REPO, 'build', 'bin', name)):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def run(cmd, log):
    with open(log, 'a') as fh:
        fh.write('+ %s\n' % ' '.join(cmd))
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode


def test_field(args, work):
    from solid_lineview.core import SampledField

    path = os.environ.get('DETECTOR_PATH', REPO)
    compact = os.path.join(path, args.config + '.xml')
    tool = find_binary('solid_field_dump', args.tool)
    scene_tool = find_binary('solid_scene_dump', args.scene_tool)
    check(tool is not None and scene_tool is not None and os.path.isfile(compact),
          'solid_field_dump, solid_scene_dump and %s are there' % compact)
    if FAILED:
        return
    log = os.path.join(work, 'test_field_view.log')
    field = os.path.join(work, 'field_x0.npz')
    rc = run([tool, compact, '--plane', 'x=0', '-o', field], log)
    check(rc == 0 and os.path.isfile(field), 'solid_field_dump writes the x = 0 plane (see %s)' % log)
    if FAILED:
        return
    f = np.load(field)
    u, v, B = f['u'], f['v'], f['B']
    check(B.shape == (len(u), len(v), 3) and np.allclose(np.diff(u), 10.0) and np.allclose(np.diff(v), 10.0)
          and np.any(np.abs(u) < 1e-9) and np.any(np.abs(v) < 1e-9),
          'the grid has a 1 cm step and holds the beam axis (%d x %d points)' % (len(u), len(v)))
    check(np.allclose(f['normal'], [1, 0, 0]) and np.allclose(f['u_dir'], [0, 0, 1]) and np.allclose(f['v_dir'], [0, 1, 0]),
          'plane x = 0 has the axes u = z, v = y')

    # the same field from DD4hep, at grid points and in between
    import dd4hep
    dd4hep.setPrintLevel(dd4hep.PrintLevel.WARNING)
    det = dd4hep.Detector.getInstance()
    det.fromXML(compact)
    fld = det.field()
    rng = np.random.default_rng(7)
    iu, iv = rng.integers(0, len(u), 300), rng.integers(0, len(v), 300)
    worst = 0.0
    for i, j in zip(iu, iv):
        b = fld.magneticField(dd4hep.Position(0.0, v[j] * dd4hep.mm, u[i] * dd4hep.mm))
        ref = np.array([b.x(), b.y(), b.z()]) / dd4hep.tesla
        worst = max(worst, float(np.abs(B[i, j] - ref).max()))
    bmax = float(np.linalg.norm(B, axis=2).max())
    check(worst < 1e-6 * max(bmax, 1e-3) and bmax > 0.5,
          'grid values equal DD4hep within 1e-6 of the largest field (%.3g T; largest deviation %.2g T)' % (bmax, worst))
    sf = SampledField(field)
    P = np.array([[0.0, 1234.5, 987.6], [0.0, -456.7, -1500.2], [0.0, 30.0, 0.0]])
    got, inside = sf.vector(P)
    ref = []
    for p in P:
        b = fld.magneticField(dd4hep.Position(*(p * dd4hep.mm)))
        ref.append(np.array([b.x(), b.y(), b.z()]) / dd4hep.gauss)
    dev = float(np.abs(got - np.array(ref)).max())
    check(inside.all() and dev < 2e-3 * bmax * 1e4,
          'between grid points the drawing interpolates to DD4hep within 0.2 %% of the largest field (%.2g G)' % dev)
    ax = sf.axial(np.array([1.0, 0.0, 0.0]), 0.0)
    check(ax is not None and abs(ax.r[0]) < 1e-9, 'the solenoid field in the x = 0 plane is recognised as axisymmetric')
    fz = os.path.join(work, 'field_z0.npz')
    rc = run([tool, compact, '--plane', 'z=0', '--u-range', '-2m', '2m', '--v-range', '-2m', '2m', '-o', fz], log)
    check(rc == 0 and SampledField(fz).axial(np.array([0.0, 0.0, 1.0]), 0.0) is None,
          'a plane across the beam axis (z = 0) gets no equal-flux lines')
    try:
        sf.check_plane(np.array([0.0, 1.0, 0.0]), 0.0)
        ok = False
    except SystemExit:
        ok = True
    check(ok, 'a field sampled on another plane is refused')

    # drawing: sampled field against the GEMC map of the configuration
    scene = os.path.join(work, 'scene.npz')
    rc = run([scene_tool, compact, '-o', scene], log)
    check(rc == 0 and os.path.isfile(scene), 'solid_scene_dump writes the scene')
    if FAILED:
        return
    lineview = [sys.executable, '-B', os.path.join(REPO, 'bin', 'solid_lineview')]
    base = [scene, '--mode', 'section', '--section', 'x=0', '--view', 'side', '--size', '1400x700', '-q']
    out = os.path.join(work, 'sampled.svg')
    rc = run(lineview + base + ['--field', field, '-o', out, '-o', out[:-3] + 'png'], log)
    svg = open(out).read() if rc == 0 and os.path.isfile(out) else ''
    check('<image ' in svg and svg.count('<path') > 4, 'solid_lineview draws the section with the sampled field')
    text = re.sub(r'<!--.*?-->', '', open(compact).read(), flags=re.S)
    gmap = None
    for ref in re.findall(r'<include\s+ref="([^"]*field[^"]*\.xml)"', text):
        try:
            frag = re.sub(r'<!--.*?-->', '', open(ref.replace('${DETECTOR_PATH}', path)).read(), flags=re.S)
        except OSError:
            continue
        m = re.search(r'field_map="([^"]*)"', frag)
        if m and 'coord_type="BrBz"' in frag:
            for d in ('.', path):
                if os.path.isfile(os.path.join(d, m.group(1))):
                    gmap = os.path.join(d, m.group(1))
    try:
        from matplotlib.image import imread
    except ImportError:
        imread = None
    if gmap and imread and svg:
        # the same picture from the map file itself. Not byte for byte: where the map ends inside the drawing,
        # the sampled field falls to zero over one grid step
        out2 = os.path.join(work, 'map.png')
        rc = run(lineview + base + ['--field', gmap, '-o', out2], log)
        ok, share = False, 1.0
        if rc == 0 and os.path.isfile(out2):
            a, b = imread(out[:-3] + 'png')[..., :3], imread(out2)[..., :3]
            if a.shape == b.shape:
                share = float((np.abs(a.astype(float) - b.astype(float)).max(-1) > 8.0 / 255.0).mean())
                ok = share < 0.003
        check(ok, 'the drawing with the sampled field equals the one with the map %s (%.3f %% of the pixels differ, '
                  'limit 0.3 %%)' % (os.path.basename(gmap), 100.0 * share))
    else:
        print('skip  comparison with a GEMC map (no 2-D map of this configuration in fieldmaps/, or no matplotlib)')
    out3 = os.path.join(work, 'target.svg')
    rc = run(lineview + base + ['--field', field, '--fit-volumes', '(^|!)Target_', '--fit-pad', '50cm', '-o', out3], log)
    ok = rc == 0 and os.path.isfile(out3)
    if ok:
        s3 = open(out3).read()
        vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', s3)
        W, H = float(vb.group(1)), float(vb.group(2))
        xs = [float(a) for d in re.findall(r' d="([^"]*)"', s3) for a, _ in re.findall(r'(-?[\d.]+)[ ,](-?[\d.]+)', d)]
        ok = bool(xs) and min(xs) > -1.0 and max(xs) < W + 1.0 and svg.count('<path') > 0
    check(ok, '--fit-volumes draws a close-up of the target, clipped to its window')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool', default=None, help='solid_field_dump (default: on PATH or build/bin)')
    ap.add_argument('--scene-tool', default=None, help='solid_scene_dump (default: on PATH or build/bin)')
    ap.add_argument('--config', default='solid_JPsi_lite')
    ap.add_argument('--workdir', default=None, help='keep the files here (default: a temporary directory)')
    ap.add_argument('--lines-only', action='store_true', help='only the checks that need numpy alone')
    args = ap.parse_args()
    test_lines()
    if not args.lines_only:
        work = args.workdir or tempfile.mkdtemp(prefix='test_field_view_')
        os.makedirs(work, exist_ok=True)
        try:
            test_field(args, work)
        finally:
            if not args.workdir:
                shutil.rmtree(work, ignore_errors=True)
    print('%d check%s failed' % (len(FAILED), '' if len(FAILED) == 1 else 's') if FAILED else 'all checks passed')
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
