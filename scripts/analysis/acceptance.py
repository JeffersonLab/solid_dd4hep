#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Acceptance maps in polar angle and momentum from particle tables, with a comparison to GEMC's stored maps.

usage:
    acceptance.py TABLE.npz [...] --config full|lite [-o MAPS.npz] [--plot FIG.png] [--gemc REF.root]
                  [--fa-segments none|4x24|C1,C2,..:WIDTH] [--la-segments ...]
                  [--theta 6:28:1] [--p 0.5:8:0.5] [--gem-min-fa 4] [--gem-min-la 3] [--ec-min 0.005] [--no-spd]
                  [--ec-r-fa MIN:MAX] [--ec-r-la MIN:MAX] [--p-loss-max GEV]
                  [--pdg ID] [--title TEXT]

A particle counts as detected in the forward (large-angle) arm when it left hits in at least --gem-min-fa
(--gem-min-la) GEM planes of that arm, a hit of its own in the scintillator pads in front of that arm's
calorimeter, and more than --ec-min GeV of visible energy in the calorimeter (shower plus preshower). --no-spd
drops the scintillator requirement. Tables come from particle_table.py. The default of 5 MeV visible says that
the particle reached the calorimeter; it holds for full showers and for the fast mode (SOLID_FAST=1) alike.

  --config        full: one six-plane GEM tracker (planes 2-6 forward, 1-4 large angle); lite: separate trackers
  --fa-segments   azimuthal segments of the forward arm, applied as a selection on the stored detector elements:
                  none (default), 4x24 (four 24-deg segments at 42, 138, 222, 318 deg), or centres and width in deg
  --la-segments   the same for the large-angle arm
  --ec-r-fa, --ec-r-la   keep particles that reach the calorimeter front between these radii (cm). The position is
                  the particle's hit in the detector just in front: the scintillator pads (forward), the last
                  large-angle GEM plane. GEMC's maps of 2017 were made with 105:220 and 83:127
  --p-loss-max E  the particle must reach the calorimeter front having lost less than E GeV of momentum
                  (default 0 = no requirement). A pion that crosses a tungsten collimator of the 3He target loses
                  150 MeV or more, one that does not loses 3 to 15 MeV: 0.05 separates them. GEMC's maps were
                  made with collimators that stop everything
  --gemc REF.root acceptance file of solid_gemc/analysis/acceptance (histograms acceptance_ThetaP_forwardangle and
                  acceptance_ThetaP_largeangle); it is averaged over the bins used here and compared
  --plot          maps as a figure; with --gemc also the reference and the difference
  -o              maps as .npz: theta_edges, p_edges, forward, forward_err, large, large_err, overall, generated

Needs numpy; uproot for --gemc; matplotlib for --plot.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

from solid_ana.accept import FOUR_BY_24, Segments, acceptance_map, arms  # noqa: E402
from solid_ana.table import concat_tables, load_table  # noqa: E402

# chart colours: one hue for magnitude, two hues around a neutral midpoint for differences
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f0a9a0", "#e34948", "#a32626"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def edges(text):
    lo, hi, step = (float(v) for v in text.split(":"))
    n = int(round((hi - lo) / step))
    return lo + step * np.arange(n + 1)


def segments(text):
    if text in ("none", "", None):
        return Segments()
    if text == "4x24":
        return FOUR_BY_24
    centres, width = text.split(":")
    return Segments(tuple(float(c) for c in centres.split(",")), float(width))


def read_gemc(path, name, theta_edges, p_edges):
    """GEMC acceptance histogram averaged over our bins (its particles were flat in theta and P)"""
    import uproot
    h = uproot.open(path)[name]
    v = h.values()
    te, pe = h.axis(0).edges(), h.axis(1).edges()
    tc, pc = 0.5 * (te[:-1] + te[1:]), 0.5 * (pe[:-1] + pe[1:])
    out = np.full((len(theta_edges) - 1, len(p_edges) - 1), np.nan)
    for i in range(len(theta_edges) - 1):
        ti = (tc > theta_edges[i]) & (tc < theta_edges[i + 1])
        for j in range(len(p_edges) - 1):
            pj = (pc > p_edges[j]) & (pc < p_edges[j + 1])
            if ti.any() and pj.any():
                out[i, j] = v[np.ix_(ti, pj)].mean()
    return out


def edge_angle(acc, theta_edges, rising=True):
    """polar angle (deg) where the acceptance, averaged over momentum, crosses half of its largest value"""
    with np.errstate(invalid="ignore"):
        prof = np.array([np.mean(r[np.isfinite(r)]) if np.isfinite(r).any() else 0.0 for r in acc])
    level = 0.5 * prof.max()
    if level <= 0:
        return float("nan")
    tc = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    idx = range(len(prof) - 1) if rising else range(len(prof) - 2, -1, -1)
    for i in idx:
        a, b = (prof[i], prof[i + 1])
        if rising and a < level <= b or (not rising) and a >= level > b:
            return float(tc[i] + (level - a) / (b - a) * (tc[i + 1] - tc[i]))
    return float("nan")


def plot(path, title, theta_edges, p_edges, maps, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    seq = LinearSegmentedColormap.from_list("acc", SEQ)
    div = LinearSegmentedColormap.from_list("diff", DIV)
    seq.set_bad(SURFACE)
    div.set_bad(SURFACE)
    rows = [("forward angle", "forward"), ("large angle", "large")]
    ncol = 3 if ref else 1
    fig, axs = plt.subplots(2, ncol, figsize=(4.6 * ncol + 1.2, 7.2), squeeze=False, facecolor=SURFACE)
    im_a = im_d = None
    for r, (label, key) in enumerate(rows):
        cols = [(maps[key], "this simulation")]
        if ref:
            cols += [(ref[key], "GEMC reference"), (maps[key] - ref[key], "difference")]
        for c, (m, what) in enumerate(cols):
            ax = axs[r][c]
            ax.set_facecolor(SURFACE)
            diff = what == "difference"
            im = ax.pcolormesh(theta_edges, p_edges, np.ma.masked_invalid(m.T), cmap=div if diff else seq,
                               vmin=-0.5 if diff else 0.0, vmax=0.5 if diff else 1.0, shading="flat", rasterized=True)
            if diff:
                im_d = im
            else:
                im_a = im
            ax.set_title("%s: %s" % (label, what), fontsize=10, color=INK, loc="left")
            if r == 1:
                ax.set_xlabel("polar angle at the vertex (deg)", color=INK2, fontsize=9)
            if c == 0:
                ax.set_ylabel("momentum (GeV)", color=INK2, fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=8, length=3)
            for s in ax.spines.values():
                s.set_color(GRID)
    fig.suptitle(title, fontsize=12, color=INK, x=0.02, ha="left")
    fig.subplots_adjust(left=0.07, right=0.90 if ref else 0.84, top=0.90, bottom=0.08, wspace=0.16, hspace=0.24)
    right = 0.915 if ref else 0.865
    cb = fig.colorbar(im_a, cax=fig.add_axes([right, 0.55 if ref else 0.08, 0.012, 0.35 if ref else 0.82]))
    cb.set_label("acceptance", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    if ref:
        cb = fig.colorbar(im_d, cax=fig.add_axes([right, 0.08, 0.012, 0.35]))
        cb.set_label("this simulation minus GEMC", color=INK2, fontsize=9)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.outline.set_edgecolor(GRID)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--config", required=True, choices=("full", "lite"))
    ap.add_argument("--fa-segments", default="none")
    ap.add_argument("--la-segments", default="none")
    ap.add_argument("--theta", default="6:28:1")
    ap.add_argument("--p", default="0.5:8:0.5")
    ap.add_argument("--gem-min-fa", type=int, default=4)
    ap.add_argument("--gem-min-la", type=int, default=3)
    ap.add_argument("--ec-min", type=float, default=0.005)
    ap.add_argument("--no-spd", action="store_true")
    ap.add_argument("--ec-r-fa")
    ap.add_argument("--ec-r-la")
    ap.add_argument("--p-loss-max", type=float, default=0.0)
    ap.add_argument("--pdg", type=int)
    ap.add_argument("--gemc")
    ap.add_argument("--plot")
    ap.add_argument("--title")
    ap.add_argument("-o", "--output")
    a = ap.parse_args()

    tabs = [load_table(f)[0] for f in a.tables]
    t = tabs[0] if len(tabs) == 1 else concat_tables(tabs)
    te, pe = edges(a.theta), edges(a.p)
    fa, la = arms(a.config, ec_min=a.ec_min, spd=not a.no_spd)
    fa.gem_min, la.gem_min = a.gem_min_fa, a.gem_min_la
    fa.p_loss_max = la.p_loss_max = a.p_loss_max
    if a.ec_r_fa:
        fa.ec_r = tuple(float(v) for v in a.ec_r_fa.split(":"))
    if a.ec_r_la:
        la.ec_r = tuple(float(v) for v in a.ec_r_la.split(":"))
    sel = np.ones(len(t["pdg"]), bool) if a.pdg is None else (t["pdg"] == a.pdg)
    dfa = fa.detected(t, segments(a.fa_segments))
    dla = la.detected(t, segments(a.la_segments))
    maps = {}
    maps["forward"], maps["forward_err"], gen = acceptance_map(t, dfa, te, pe, sel)
    maps["large"], maps["large_err"], _ = acceptance_map(t, dla, te, pe, sel)
    maps["overall"], maps["overall_err"], _ = acceptance_map(t, dfa | dla, te, pe, sel)
    n = int(sel.sum())
    print("%d particles; detected forward %d (%.1f %%), large angle %d (%.1f %%), both %d"
          % (n, (dfa & sel).sum(), 100.0 * (dfa & sel).sum() / n, (dla & sel).sum(), 100.0 * (dla & sel).sum() / n,
             (dfa & dla & sel).sum()))
    print("  bins: theta %g-%g deg in %d, p %g-%g GeV in %d; %.0f generated per bin on average"
          % (te[0], te[-1], len(te) - 1, pe[0], pe[-1], len(pe) - 1, gen.mean()))
    for key, rising_first in (("forward", True), ("large", True)):
        lo, hi = edge_angle(maps[key], te, rising=True), edge_angle(maps[key], te, rising=False)
        print("  %-8s plateau %.2f (largest bin %.2f); edges at half of the plateau, averaged over momentum: %.1f and %.1f deg"
              % (key, np.nanmean(maps[key][maps[key] > 0.5 * np.nanmax(maps[key])]), np.nanmax(maps[key]), lo, hi))

    ref = None
    if a.gemc:
        ref = {"forward": read_gemc(a.gemc, "acceptance_ThetaP_forwardangle", te, pe),
               "large": read_gemc(a.gemc, "acceptance_ThetaP_largeangle", te, pe)}
        print("comparison with %s" % os.path.basename(a.gemc))
        for key in ("forward", "large"):
            d = maps[key] - ref[key]
            both = np.isfinite(d) & ((maps[key] > 0.02) | (ref[key] > 0.02))
            inside = np.isfinite(d) & (maps[key] > 0.5) & (ref[key] > 0.5)
            lo, hi = edge_angle(ref[key], te, rising=True), edge_angle(ref[key], te, rising=False)
            print("  %-8s GEMC edges %.1f and %.1f deg; where either is above 2 %%: mean difference %+.3f, rms %.3f "
                  "(%d bins); where both are above 50 %%: mean %+.3f (%d bins)"
                  % (key, lo, hi, d[both].mean(), d[both].std(), both.sum(),
                     d[inside].mean() if inside.any() else float("nan"), inside.sum()))
    if a.output:
        np.savez_compressed(a.output, theta_edges=te, p_edges=pe, generated=gen, **maps,
                            **({"gemc_" + k: v for k, v in ref.items()} if ref else {}))
        print("wrote %s" % a.output)
    if a.plot:
        plot(a.plot, a.title or "Acceptance, %s" % ", ".join(os.path.basename(os.path.dirname(os.path.abspath(f))) or f
                                                             for f in a.tables[:1]), te, pe, maps, ref)
        print("wrote %s" % a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
