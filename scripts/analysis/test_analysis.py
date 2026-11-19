#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test of the analysis layer: single particles -> ddsim -> particle table -> acceptance (ctest: analysis).

usage (inside eic-shell / solid-shell; needs DETECTOR_PATH and the plugins): test_analysis.py [--outdir DIR] [--keep]

  1. gun_events.py: 80 electrons of 3 to 5 GeV, half at 9.5-12.5 deg (forward arm), half at 19-23 deg (large angle),
     from the J/psi target position; flat distributions inside the requested ranges
  2. ddsim on solid_JPsi_lite (forward detectors over the full azimuth), no optical photons
  3. particle table: one row per electron with the generated kinematics; GEM planes, scintillator and calorimeter
     filled for the arm the electron was sent to, and no forward energy from the others; visible energy near 20 % of
     the momentum in both arms (the large-angle showers are stored as particles of their own and must be added)
  4. detector elements against positions: the GEM sector and the scintillator pad stored for a hit contain the
     hit's azimuth (the numbering that the segment selection relies on)
  5. segments: a particle passes the 4 x 24 deg selection exactly when its GEM hits and its calorimeter cell lie
     inside a segment; about 27 % pass; the full-azimuth selection keeps every detected particle
  6. acceptance.py: maps and a figure are written
  8. fast mode (SOLID_FAST=1): the 40 forward electrons again, without low-energy shower particles; as many are
     detected as with full showers, give or take the electrons at the edges
  7. track time limit (SOLID_MAX_TIME): with a limit of 5 ns a 4 GeV electron is stopped 1.5 m from the target
     and leaves nothing in the forward calorimeter; the default limit stopped nothing in step 2

About one minute. Exit status 0 = passed.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))

from solid_ana.accept import FOUR_BY_24, arms  # noqa: E402
from solid_ana.table import build_table, load_table, save_table  # noqa: E402

FAILED = []


def check(ok, what):
    print("%s  %s" % ("ok  " if ok else "FAIL", what))
    if not ok:
        FAILED.append(what)
    return ok


def run(cmd, log, **kw):
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, **kw).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    out = a.outdir or tempfile.mkdtemp(prefix="solid_analysis_")
    os.makedirs(out, exist_ok=True)
    path = os.environ["DETECTOR_PATH"]
    n = 40

    # 1. events: two angular ranges in two files, joined by ddsim through two input files
    files = []
    for tag, theta, seed in (("fa", "9.5:12.5", 3), ("la", "19:23", 4)):
        f = os.path.join(out, "gun_%s.hepmc" % tag)
        rc = run([sys.executable, os.path.join(HERE, "gun_events.py"), "--pdg", "11", "-n", str(n), "--p", "3:5",
                  "--theta", theta, "--z=-322.5:-307.5", "--seed", str(seed), "--first-event", str(len(files) * n),
                  "-o", f], os.path.join(out, "gun_%s.log" % tag))
        files.append(f)
        check(rc == 0 and os.path.getsize(f) > 0, "gun_events.py wrote %d electrons at theta %s deg" % (n, theta))

    # 2. simulation
    env = dict(os.environ, SOLID_GUN="0", SOLID_OPTICAL="0")
    cmd = ["ddsim", "--steeringFile", os.path.join(path, "steering", "solid_sim.py"), "--compactFile",
           os.path.join(path, "solid_JPsi_lite.xml"), "--inputFiles"] + files + [
           "-N", str(2 * n), "--random.seed", "7", "--outputFile", "sim.edm4hep.root"]
    rc = run(cmd, os.path.join(out, "ddsim.log"), env=env, cwd=out)
    sim = os.path.join(out, "sim.edm4hep.root")
    if not check(rc == 0 and os.path.exists(sim), "ddsim simulated %d events (log: %s/ddsim.log)" % (2 * n, out)):
        return 1

    # 3. table
    t, info = build_table(sim)
    save_table(os.path.join(out, "table.npz"), t, info)
    t, info = load_table(os.path.join(out, "table.npz"))
    th = np.degrees(t["theta"])
    check(len(t["pdg"]) == 2 * n and (t["pdg"] == 11).all(), "the table has one row per electron (%d)" % len(t["pdg"]))
    fwd = th < 15.0
    check(fwd.sum() == n and (th[fwd] > 9.5).all() and (th[fwd] < 12.5).all() and (th[~fwd] > 19.0).all()
          and (th[~fwd] < 23.0).all() and (t["p"] > 3.0).all() and (t["p"] < 5.0).all()
          and (t["vz"] > -3225.0).all() and (t["vz"] < -3075.0).all(),
          "generated angles, momenta and vertices are inside the requested ranges")
    fa, la = arms("lite")
    nfa = t["GEMTrackerFAHits_hit"][:, 1:6].sum(axis=1)
    nla = t["GEMTrackerLAHits_hit"][:, 1:5].sum(axis=1)
    check((nfa[fwd] >= 4).mean() > 0.8 and (nla[~fwd] >= 3).mean() > 0.8,
          "GEM planes: %.0f %% of the forward electrons have 4 or 5 forward planes, %.0f %% of the others 3 or 4 "
          "large-angle planes" % (100 * (nfa[fwd] >= 4).mean(), 100 * (nla[~fwd] >= 3).mean()))
    # shower leakage from the large-angle calorimeter can cross the forward planes, and ddsim books the hits of
    # secondaries it does not store on their ancestor: a few such rows are expected, forward energy is not
    check((nfa[~fwd] > 0).mean() < 0.15 and (t["FAECShowerHits_e"][~fwd] < 0.05).all(),
          "large-angle electrons leave no energy in the forward calorimeter, and %d of %d have forward GEM hits "
          "(shower leakage)" % ((nfa[~fwd] > 0).sum(), (~fwd).sum()))
    dfa, dla = fa.detected(t), la.detected(t)
    r_fa = np.median(t["FAECShowerHits_e"][dfa] / t["p"][dfa])
    r_la = np.median(t["LAECShowerHits_e"][dla] / t["p"][dla])
    check(0.15 < r_fa < 0.25 and 0.15 < r_la < 0.25,
          "visible calorimeter energy over momentum: %.3f forward, %.3f large angle (shashlik sampling, about 0.2)"
          % (r_fa, r_la))
    check(dfa[fwd].mean() > 0.8 and dla[~fwd].mean() > 0.8 and not (dfa & dla).any(),
          "detected: %d of %d forward, %d of %d large angle, none in both" % (dfa.sum(), n, dla.sum(), n))

    # 4. elements against positions
    worst = 0.0
    for coll, arm in (("GEMTrackerFAHits", fa), ("GEMTrackerLAHits", la)):
        h = t[coll + "_hit"]
        phi = np.degrees(np.arctan2(t[coll + "_y"], t[coll + "_x"]))[h]
        centre = arm.gem_sector1 + arm.gem_dphi * (t[coll + "_elem"][h] - 1)
        worst = max(worst, float(np.abs((phi - centre + 180.0) % 360.0 - 180.0).max()))
    check(worst <= 6.0 + 1e-3, "every GEM hit lies in the sector stored for it (largest distance from the sector "
          "centre %.2f deg of 6)" % worst)
    worst = 0.0
    for coll, arm in (("FASPDHits", fa), ("LASPDHits", la)):
        h = t[coll + "_hit"]
        phi = np.degrees(np.arctan2(t[coll + "_y"], t[coll + "_x"]))[h]
        centre = arm.spd_phi0 + arm.spd_dphi * (t[coll + "_elem"][h] - 0.5)
        worst = max(worst, float(np.abs((phi - centre + 180.0) % 360.0 - 180.0).max()))
    check(worst <= 3.0 + 1e-3, "every scintillator hit lies in the pad stored for it (largest distance from the pad "
          "centre %.2f deg of 3)" % worst)

    # 5. segments
    sfa = fa.detected(t, FOUR_BY_24)
    inside = np.ones(len(sfa), bool)
    h = t["GEMTrackerFAHits_hit"][:, 1:6]
    phi = np.degrees(np.arctan2(t["GEMTrackerFAHits_y"], t["GEMTrackerFAHits_x"]))[:, 1:6]
    in_seg = FOUR_BY_24.contains(np.nan_to_num(phi, nan=1e9))
    ec_phi = np.degrees(np.arctan2(t["FAECShowerHits_y"], t["FAECShowerHits_x"]))
    # a hit 0-6 deg outside a window can still be in a kept sector only if the window ends on a sector edge: it does
    planes_in = (h & in_seg).sum(axis=1)
    expect = dfa & (planes_in >= fa.gem_min) & FOUR_BY_24.contains(np.nan_to_num(ec_phi, nan=1e9))
    check((sfa == expect).all(), "the 4 x 24 deg selection keeps exactly the electrons whose GEM hits and calorimeter "
          "cell are inside a segment (%d of %d detected forward electrons)" % (sfa.sum(), dfa.sum()))
    check(not (sfa & ~dfa).any() and 0.05 < sfa.sum() / max(1, dfa.sum()) < 0.60,
          "a subset passes: %.0f %% (4 x 24 / 360 = 27 %%, with %d electrons)" % (100.0 * sfa.sum() / max(1, dfa.sum()),
                                                                              dfa.sum()))

    # 6. the tool
    rc = run([sys.executable, os.path.join(HERE, "acceptance.py"), os.path.join(out, "table.npz"), "--config", "lite",
              "--theta", "8:24:2", "--p", "3:5:1", "--fa-segments", "4x24", "--ec-r-la", "83:127",
              "-o", os.path.join(out, "acc.npz"),
              "--plot", os.path.join(out, "acc.png")], os.path.join(out, "acceptance.log"))
    ok = rc == 0 and os.path.exists(os.path.join(out, "acc.npz"))
    if ok:
        z = np.load(os.path.join(out, "acc.npz"))
        ok = z["forward"].shape == (8, 2) and np.nanmax(z["large"]) > 0.8 and os.path.getsize(os.path.join(out, "acc.png")) > 5000
    check(ok, "acceptance.py wrote maps of 8 x 2 bins and a figure")

    # 7. track time limit
    log = open(os.path.join(out, "ddsim.log")).read()
    check("SoLIDTrackTimeLimit" in log and " 0 tracks stopped after 100000 ns" in log,
          "the default time limit of 100 us was active in step 2 and stopped no track")
    f = os.path.join(out, "gun_limit.hepmc")
    run([sys.executable, os.path.join(HERE, "gun_events.py"), "--pdg", "11", "-n", "1", "--p", "4:4", "--theta", "11:11",
         "--z=-315:-315", "-o", f], os.path.join(out, "gun_limit.log"))
    cmd = ["ddsim", "--steeringFile", os.path.join(path, "steering", "solid_sim.py"), "--compactFile",
           os.path.join(path, "solid_JPsi_lite.xml"), "--inputFiles", f, "-N", "1", "--outputFile", "limit.edm4hep.root"]
    rc = run(cmd, os.path.join(out, "limit.log"), env=dict(env, SOLID_MAX_TIME="5"), cwd=out)
    log = open(os.path.join(out, "limit.log")).read()
    t5, _ = build_table(os.path.join(out, "limit.edm4hep.root")) if rc == 0 else ({}, None)
    check(rc == 0 and "stopped e- with" in log and "after 5" in log and float(t5["FAECShowerHits_etot"][0]) == 0.0,
          "with SOLID_MAX_TIME=5 the electron is stopped after 5 ns and the forward calorimeter stays empty")

    # 8. fast mode
    cmd = ["ddsim", "--steeringFile", os.path.join(path, "steering", "solid_sim.py"), "--compactFile",
           os.path.join(path, "solid_JPsi_lite.xml"), "--inputFiles", files[0], "-N", str(n), "--random.seed", "7",
           "--outputFile", "fast.edm4hep.root"]
    rc = run(cmd, os.path.join(out, "fast.log"), env=dict(env, SOLID_FAST="1"), cwd=out)
    log = open(os.path.join(out, "fast.log")).read()
    tf, _ = build_table(os.path.join(out, "fast.edm4hep.root")) if rc == 0 else ({}, None)
    nfast = int(fa.detected(tf).sum()) if rc == 0 else -1
    size = os.path.getsize(os.path.join(out, "fast.edm4hep.root")) if rc == 0 else 0
    check(rc == 0 and "SoLIDShowerStop" in log and "were dropped" in log and abs(nfast - int(dfa[fwd].sum())) <= 4
          and float(np.median(tf["FAECShowerHits_e"][fa.detected(tf)])) < 0.5 * r_fa * 4.0,
          "fast mode: %d of %d forward electrons detected (%d with full showers); the file is %.1f MB"
          % (nfast, n, dfa[fwd].sum(), size / 1e6))

    print("%d check%s failed" % (len(FAILED), "" if len(FAILED) == 1 else "s") if FAILED else "all checks passed")
    if not a.keep and not a.outdir and not FAILED:
        shutil.rmtree(out, ignore_errors=True)
    else:
        print("files kept in %s" % out)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
