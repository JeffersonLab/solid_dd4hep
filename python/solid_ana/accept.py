# SPDX-License-Identifier: LGPL-3.0-or-later
"""Acceptance from particle tables: what counts as detected, azimuthal segments, maps in (theta, p).

A particle is detected in an arm (forward or large angle) when it left hits in enough GEM planes of that arm, a
hit of its own in the scintillator pads in front of the arm's calorimeter, and energy in the calorimeter (shower
plus preshower). The numbers are options of `Arm`; the defaults are loose (acceptance, not efficiency).

Azimuthal segments are a selection on what the table stores: the GEM sector of every plane hit, the scintillator
pad and the calorimeter cell. A particle passes when all of them lie in a segment, exactly as if only those
detector elements had been built. This holds as long as the removed elements were not needed to reach the kept
ones (docs/analysis.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segments:
    """Azimuthal segments: `width` degrees around each centre (degrees). No centres = full azimuth."""
    centers: tuple = ()
    width: float = 24.0

    def contains(self, phi_deg):
        phi_deg = np.asarray(phi_deg, float)
        if not self.centers:
            return np.ones(phi_deg.shape, bool)
        ok = np.zeros(phi_deg.shape, bool)
        for c in self.centers:
            d = (phi_deg - c + 180.0) % 360.0 - 180.0
            ok |= np.abs(d) < 0.5 * self.width
        return ok

    @property
    def fraction(self):
        return 1.0 if not self.centers else len(self.centers) * self.width / 360.0


FOUR_BY_24 = Segments((42.0, 138.0, 222.0, 318.0), 24.0)      # compact/configs/lite_FA_4seg24.xml


@dataclass
class Arm:
    """One detector arm and what detection in it requires."""
    name: str
    gem: str                         # GEM collection
    gem_layers: tuple                # its planes that belong to the arm
    gem_min: int                     # planes that must be hit
    ec: str = ""                     # calorimeter shower collection ("" = no calorimeter required)
    ps: str = ""                     # preshower collection, added to the shower energy ("" = shower alone)
    ec_min: float = 0.0              # GeV visible in shower plus preshower from the particle and its descendants
    ec_total: bool = False           # use the energy of the whole event instead (single-particle samples)
    ec_r: tuple = ()                 # (rmin, rmax) in cm of the particle at the calorimeter front; () = no cut
    front: tuple = ()                # (tracker collection, layer) whose hit gives that position; () = use the cell
    p_loss_max: float = 0.0          # GeV: largest momentum loss up to the front hit; 0 = no requirement
    spd: str = ""                    # scintillator collection ("" = not required)
    gem_sector1: float = 96.0        # azimuth (deg) of the centre of GEM sector 1; sectors of gem_dphi follow
    gem_dphi: float = 12.0
    spd_phi0: float = 90.0           # scintillator sector s covers phi0 + (s-1) dphi ... phi0 + s dphi
    spd_dphi: float = 6.0
    segments: Segments = field(default_factory=Segments)

    def detected(self, t, segments=None):
        """boolean array over the rows of table t"""
        seg = self.segments if segments is None else segments
        n = len(t["pdg"])
        if self.gem + "_hit" not in t:
            return np.zeros(n, bool)
        layers = [l for l in self.gem_layers if l < t[self.gem + "_hit"].shape[1]]
        hit = t[self.gem + "_hit"][:, layers]
        if seg.centers:
            sector = t[self.gem + "_elem"][:, layers]
            hit = hit & seg.contains(self.gem_sector1 + self.gem_dphi * (sector - 1))
        ok = hit.sum(axis=1) >= self.gem_min
        if self.ec:
            if self.ec + "_e" not in t:
                return np.zeros(n, bool)
            key = "_etot" if self.ec_total else "_e"
            e = t[self.ec + key].astype(float)
            cx, cy = t[self.ec + "_x"], t[self.ec + "_y"]
            if self.ps and self.ps + "_e" in t:
                e = e + t[self.ps + key]
                miss = ~np.isfinite(cx)                  # no shower cell: take the preshower cell
                cx, cy = np.where(miss, t[self.ps + "_x"], cx), np.where(miss, t[self.ps + "_y"], cy)
            ok &= e > self.ec_min
            if seg.centers:
                phi = np.degrees(np.arctan2(cy, cx))
                ok &= seg.contains(np.nan_to_num(phi, nan=1e9)) & np.isfinite(phi)
            if self.ec_r:
                if self.front and self.front[0] + "_x" in t and self.front[1] < t[self.front[0] + "_x"].shape[1]:
                    x, y = t[self.front[0] + "_x"][:, self.front[1]], t[self.front[0] + "_y"][:, self.front[1]]
                else:                # the cell with most energy: deeper in the calorimeter, so at a larger radius
                    x, y = cx, cy
                r = np.nan_to_num(np.hypot(x, y), nan=-1.0) / 10.0
                ok &= (r > self.ec_r[0]) & (r < self.ec_r[1])
        if self.p_loss_max > 0.0 and self.front and self.front[0] + "_p" in t:
            pf = t[self.front[0] + "_p"][:, self.front[1]]
            ok &= t["p"] - np.nan_to_num(pf, nan=-1e9) < self.p_loss_max
        if self.spd:
            if self.spd + "_hit" not in t:
                return np.zeros(n, bool)
            h = t[self.spd + "_hit"]
            pad = h.any(axis=1)
            if seg.centers:
                sector = t[self.spd + "_elem"]
                inside = seg.contains(self.spd_phi0 + self.spd_dphi * (sector - 0.5))
                pad = (h & inside).any(axis=1)
            ok &= pad
        return ok


def arms(config, ec_min=0.005, spd=True):
    """Forward and large-angle arms of a configuration family: 'lite' (separate GEM trackers) or 'full'."""
    if config == "lite":
        fa = Arm("forward", "GEMTrackerFAHits", (1, 2, 3, 4, 5), 4, "FAECShowerHits", "FAECPreShowerHits", ec_min, spd="FASPDHits" if spd else "",
                 front=("FASPDHits", 1))
        la = Arm("large", "GEMTrackerLAHits", (1, 2, 3, 4), 3, "LAECShowerHits", "LAECPreShowerHits", ec_min, spd="LASPDHits" if spd else "",
                 front=("GEMTrackerLAHits", 4))
    elif config == "full":
        # six planes: 1-4 serve the large angle, 2-6 the forward angle (pre-CDR table 11)
        fa = Arm("forward", "GEMTrackerHits", (2, 3, 4, 5, 6), 4, "FAECShowerHits", "FAECPreShowerHits", ec_min, spd="FASPDHits" if spd else "",
                 front=("FASPDHits", 1))
        la = Arm("large", "GEMTrackerHits", (1, 2, 3, 4), 3, "LAECShowerHits", "LAECPreShowerHits", ec_min, spd="LASPDHits" if spd else "",
                 front=("GEMTrackerHits", 4))
    else:
        raise ValueError("config is 'lite' or 'full'")
    return fa, la


def acceptance_map(t, detected, theta_edges, p_edges, select=None):
    """(acceptance, error, generated) in bins of polar angle (deg) and momentum (GeV); binomial errors"""
    th = np.degrees(t["theta"])
    sel = np.ones(len(th), bool) if select is None else select
    gen, _, _ = np.histogram2d(th[sel], t["p"][sel], bins=(theta_edges, p_edges))
    det, _, _ = np.histogram2d(th[sel & detected], t["p"][sel & detected], bins=(theta_edges, p_edges))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(gen > 0, det / gen, np.nan)
        err = np.where(gen > 0, np.sqrt(np.clip(acc * (1 - acc), 0, None) / gen), np.nan)
    return acc, err, gen
