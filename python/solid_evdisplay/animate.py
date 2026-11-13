# SPDX-License-Identifier: LGPL-3.0-or-later
"""Animations of an event view: frames with cached geometry layers, and their encoders.

The camera is fixed, so the detector drawing is rendered once into two layers (fills below the volume
colours, lines above them); a frame composes them with the volume colours, tracks and hits at its time.
The last frame is the static figure (no time limit, no clock).

Outputs
    GIF            Pillow; one global palette, unchanged pixels transparent
    animated SVG   one file with SMIL animation, no encoder needed
    WebP           when Pillow was built with WebP support
    MP4, WebM      through an ffmpeg program: $SOLID_FFMPEG, or ffmpeg on PATH
A format without an encoder is skipped with a message; that is not an error.
"""
import math
import os
import shutil
import subprocess
import time

import numpy as np

from solid_lineview import core as lv

from .view import INK, REDS_RGB, PHOTON_RGB, STYLE


# =====================================================================================================
#  frames
# =====================================================================================================

class FrameRenderer:
    """RGB frames of an EventView at a reduced scale (needs pycairo)."""

    def __init__(self, view, scale=0.6):
        import cairo
        self.cairo = cairo
        self.view, self.scale = view, scale
        self.W, self.H = int(view.W * scale) // 2 * 2, int(view.H * scale) // 2 * 2      # even, for video
        self.bg = self._layer(view.fill_items, True)
        self.fg = self._layer(view.line_items, False)
        self.top = self._layer(view.top_items(), False)

    def _layer(self, items, opaque):
        cairo = self.cairo
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.W, self.H)
        cr = cairo.Context(surf)
        if opaque:
            cr.set_source_rgb(1, 1, 1)
            cr.paint()
        cr.scale(self.scale, self.scale)
        lv.cairo_draw_items(cr, items, self.view.dpi)
        return surf

    def frame(self, tmax):
        """(H, W, 3) uint8 at time tmax; None = the whole event, as in the static figure"""
        cairo, v = self.cairo, self.view
        surf = cairo.ImageSurface(cairo.FORMAT_RGB24, self.W, self.H)
        cr = cairo.Context(surf)
        cr.set_source_surface(self.bg, 0, 0)
        cr.paint()
        cr.save()
        cr.scale(self.scale, self.scale)
        lv.cairo_draw_items(cr, v.volume_items(tmax), v.dpi)
        cr.restore()
        cr.set_source_surface(self.fg, 0, 0)
        cr.paint()
        cr.save()
        cr.scale(self.scale, self.scale)
        lv.cairo_draw_items(cr, v.event_items(tmax), v.dpi)
        cr.restore()
        cr.set_source_surface(self.top, 0, 0)
        cr.paint()
        if tmax is not None:
            cr.scale(self.scale, self.scale)
            lv.cairo_draw_items(cr, v.clock_items(tmax), v.dpi)
        surf.flush()
        bgra = np.ndarray((self.H, self.W, 4), np.uint8, surf.get_data(), 0, (surf.get_stride(), 4, 1))
        return np.ascontiguousarray(bgra[:, :, 2::-1])

    def static(self):
        """the static figure drawn in one pass at the frame scale (reference for the last frame)"""
        cairo, v = self.cairo, self.view
        surf = cairo.ImageSurface(cairo.FORMAT_RGB24, self.W, self.H)
        cr = cairo.Context(surf)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()
        cr.scale(self.scale, self.scale)
        lv.cairo_draw_items(cr, v.drawlist(None).items, v.dpi)
        surf.flush()
        bgra = np.ndarray((self.H, self.W, 4), np.uint8, surf.get_data(), 0, (surf.get_stride(), 4, 1))
        return np.ascontiguousarray(bgra[:, :, 2::-1])


def frame_times(n, t_end):
    """times of the n frames: 0 .. t_end in equal steps, then None (the complete event) for the last"""
    if n < 2:
        return [None]
    return [t_end * i / (n - 1) for i in range(n - 1)] + [None]


# =====================================================================================================
#  encoders
# =====================================================================================================

def _reserved_colors():
    """palette entries that must exist whatever the frequent colours are: the particle colours and their
    blends with white (thin anti-aliased lines are rare colours), the energy ramp, the photon colour"""
    out = []
    for rgb in [st["rgb"] for st in STYLE.values()] + [INK, PHOTON_RGB]:
        c = np.array(rgb) * 255.0
        for w in (0.0, 0.35, 0.6, 0.8):
            out.append(np.round(c * (1 - w) + 255 * w))
    for u in np.linspace(0, 1, 40):
        x = 1 + u * (len(REDS_RGB) - 2)
        i = min(int(x), len(REDS_RGB) - 2)
        out.append(np.round(255 * (REDS_RGB[i] * (1 - (x - i)) + REDS_RGB[i + 1] * (x - i))))
    return np.unique(np.array(out, np.int64), axis=0)


class GifWriter:
    """GIF with one global palette of 255 colours (no flicker) + a transparent index for unchanged pixels.

    The palette holds the reserved colours and the most frequent colours of a reference frame (the last
    one, which shows everything).  Every pixel maps to the nearest palette colour, exactly.
    """

    def __init__(self, reference, fps, hold):
        key = self._key(reference)
        u, c = np.unique(key, return_counts=True)
        res = _reserved_colors()
        kk = (res[:, 0] << 16) | (res[:, 1] << 8) | res[:, 2]
        order = u[np.argsort(-c, kind="stable")]
        order = order[~np.isin(order, kk)]
        top = np.concatenate([kk, order[:255 - len(kk)]])
        self.pal = np.stack([(top >> 16) & 255, (top >> 8) & 255, top & 255], 1).astype(np.int64)
        self.lut = np.full(1 << 24, -1, np.int16)
        self.fps, self.hold = fps, hold
        self.images, self.prev = [], None

    @staticmethod
    def _key(rgb):
        a = rgb.astype(np.int64)
        return (a[:, :, 0] << 16) | (a[:, :, 1] << 8) | a[:, :, 2]

    def add(self, rgb):
        key = self._key(rgb)
        idx = self.lut[key]
        miss = idx < 0
        if miss.any():
            uk = np.unique(key[miss])
            uc = np.stack([(uk >> 16) & 255, (uk >> 8) & 255, uk & 255], 1)
            for i in range(0, len(uk), 4096):
                d = ((uc[i:i + 4096, None, :] - self.pal[None, :, :]) ** 2).sum(2)
                self.lut[uk[i:i + 4096]] = d.argmin(1)
            idx = self.lut[key]
        idx = idx.astype(np.uint8)
        out = idx.copy()
        if self.prev is not None:
            out[idx == self.prev] = 255
        self.prev = idx
        self.images.append(out)

    def save(self, path, scale=1.0):
        from PIL import Image
        flat = [int(v) for v in self.pal.reshape(-1)] + [0] * (3 * (256 - len(self.pal)))
        ims = []
        for a in self.images:
            im = Image.fromarray(a, mode="P")
            im.putpalette(flat)
            ims.append(im)
        dur = [int(round(1000.0 / self.fps))] * len(ims)
        dur[-1] += int(self.hold * 1000)
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=dur, loop=0, optimize=False, disposal=1,
                    transparency=255)


def webp_available():
    try:
        from PIL import features
        return bool(features.check("webp"))
    except Exception:
        return False


def find_ffmpeg():
    """path of an ffmpeg program: $SOLID_FFMPEG if set (must exist), else ffmpeg on PATH, else None"""
    env = os.environ.get("SOLID_FFMPEG")
    if env:
        return env if (os.path.isfile(env) and os.access(env, os.X_OK)) else shutil.which(env)
    return shutil.which("ffmpeg")


class VideoWriter:
    """Raw RGB frames piped into ffmpeg; MP4 (H.264 High profile, yuv420p, even frame size, index at the
    start of the file: what PowerPoint, Keynote, QuickTime and browsers ask for) or WebM (VP9).

    The frames are converted to YUV with the BT.709 matrix in limited range, and the stream says so.
    Without these tags a player has to guess between BT.601 and BT.709 (most go by the frame height), and
    a wrong guess shifts the colours of the energy scale."""

    COLOR = ["-vf", "scale=out_color_matrix=bt709:out_range=tv,"
             "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"]
    CODEC = {".mp4": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"] + COLOR + ["-movflags", "+faststart"],
             ".webm": ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "32", "-b:v", "0"] + COLOR}

    def __init__(self, ffmpeg, path, width, height, fps, hold):
        ext = os.path.splitext(path)[1].lower()
        self.path, self.fps, self.hold, self.last = path, fps, hold, None
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
               "%dx%d" % (width, height), "-r", "%g" % fps, "-i", "-", "-an"] + self.CODEC[ext] + [path]
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.failed = False

    def add(self, rgb):
        self.last = rgb
        if not self.failed:
            try:
                self.proc.stdin.write(rgb.tobytes())
            except (BrokenPipeError, OSError):
                self.failed = True

    def close(self):
        """True if the file was written"""
        if self.last is not None:
            for _ in range(int(round(self.hold * self.fps))):
                self.add(self.last)
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        err = self.proc.stderr.read().decode(errors="replace")
        rc = self.proc.wait()
        if rc != 0 or self.failed or not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            print("ffmpeg failed for %s (exit status %d): %s" % (self.path, rc, err.strip()[-300:]))
            return False
        return True


# =====================================================================================================
#  animated SVG (SMIL): static geometry, growing tracks, hits and volumes switching on
# =====================================================================================================

def write_svg_animation(view, path, t_end, duration, hold=2.0):
    """Tracks: stroke-dashoffset animated with keyTimes that follow the point times; dashed tracks (photons,
    neutrals) appear at their start time; hits: opacity set at the hit time; volumes: colour steps at 12
    times between their first and last deposit; clock: a text per ns.  The file loops."""
    k = duration / t_end
    total = duration + hold
    dpi, pt = view.dpi, view.dpi / 72.0
    a = view.a
    hexc = lv._rgbhex
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
           'width="%.1fpt" height="%.1fpt" viewBox="0 0 %d %d" data-t-end-ns="%.4f" data-duration-s="%.4f" '
           'data-loop-s="%.4f">' % (view.W * 72 / dpi, view.H * 72 / dpi, view.W, view.H, t_end, duration, total),
           '<!-- generated by solid_evview: %.1f ns in %.0f s, then %.0f s hold; SMIL animation -->'
           % (t_end, duration, hold),
           '<rect width="%d" height="%d" fill="#ffffff"/>' % (view.W, view.H)]
    out += lv.svg_items(view.fill_items, dpi)
    loop = 'dur="%.3fs" repeatCount="indefinite"' % total
    dep = view.dep
    for s, rings in [(s, [h]) for s, h in view.ghosts.items()] + list(view.rings.items()):     # ghosts below
        final = view.volume_color(s, None)
        if final is None:
            continue
        ghost = s not in view.rings
        d = "".join("M" + lv._fmt(r[0]) + "L" + " ".join(lv._fmt(q) for q in r[1:]) + "Z" for r in rings)
        times = []
        if s in dep.energy:
            ts = dep.energy[s][0]
            times += list(np.linspace(ts[0], min(ts[-1], t_end), 12))
        if s in dep.photons:
            times.append(float(dep.photons[s][0]))
        vals, keys = [], []
        for tt in sorted(set(times)):
            c = view.volume_color(s, tt)
            if c is not None and tt <= t_end:
                vals.append(hexc(c))
                keys.append(min(tt * k / total, 0.9990))
        # whatever arrives after the animated time span shows in the hold phase, as in the last frame
        vals.append(hexc(final))
        keys.append(max(duration / total, keys[-1] + 1e-4) if keys else duration / total)
        vals = [vals[0]] + vals
        keys = [0.0] + keys
        out.append('<path fill="%s" stroke="%s" stroke-width="%.2f"%s fill-rule="evenodd" opacity="0" d="%s">'
                   '<animate attributeName="opacity" calcMode="discrete" values="0;1" keyTimes="0;%.4f" %s/>'
                   '<animate attributeName="fill" calcMode="discrete" values="%s" keyTimes="%s" %s/></path>'
                   % (vals[0], "#52514e" if ghost else "#0b0b0b", (0.25 if ghost else 0.3) * pt,
                      ' fill-opacity="0.55"' if ghost else "", d, keys[1], loop, ";".join(vals),
                      ";".join("%.4f" % x for x in keys), loop))
    out += lv.svg_items(view.line_items, dpi)
    # tracks: the polylines of the complete event with the times of their points
    for kind, prim, xy, T in view.track_pieces():
        if len(xy) < 2:
            continue
        seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
        L = np.concatenate([[0], np.cumsum(seg)])
        if L[-1] < 0.5:
            continue
        frac = L / L[-1]
        st = STYLE[kind]
        lw = a.traj_lw * (2.6 if prim else 1.0) * pt
        tt = np.clip(T, 0, t_end)
        sel = np.unique(np.concatenate([[0], np.linspace(0, len(tt) - 1, min(len(tt), 24)).astype(int)]))
        kt = np.maximum.accumulate(np.array([0.0] + list(np.clip(tt[sel] * k / total, 0, 1)) + [1.0]))
        vals = [1.0] + list(1 - frac[sel]) + [0.0]
        if T[-1] > t_end:                      # the rest of the track appears in the hold phase
            kt[-2] = min(kt[-2], duration / total)
        dpath = "L".join("%.1f %.1f" % (xi, yi) for xi, yi in xy)
        if st["dash"] and not prim:
            # a dashed line cannot grow by the dash-offset trick: switch it on at its start time
            out.append('<path fill="none" stroke="%s" stroke-width="%.2f" stroke-dasharray="%s" opacity="0" d="M%s">'
                       '<animate attributeName="opacity" calcMode="discrete" values="0;1;1" keyTimes="0;%.4f;1" '
                       '%s/></path>' % (hexc(st["rgb"]), lw, ",".join("%g" % (v * pt) for v in st["dash"]), dpath,
                                        min(max(kt[1], 1e-4), 0.9999), loop))
            continue
        out.append('<path fill="none" stroke="%s" stroke-width="%.2f" stroke-linecap="round" pathLength="1" '
                   'stroke-dasharray="1 1" stroke-dashoffset="1" d="M%s"><animate attributeName="stroke-dashoffset" '
                   'values="%s" keyTimes="%s" %s/></path>'
                   % (hexc(st["rgb"]), lw, dpath, ";".join("%.4f" % v for v in vals),
                      ";".join("%.4f" % v for v in kt), loop))
    # hits
    if len(view.hitT):
        x, y, _ = view.to_px(view.hitP)
        r = a.hit_size * pt
        style = ('fill="none" stroke="#0b0b0b" stroke-width="%.2f"' % (0.7 * pt) if view.ring_hits else
                 'fill="#0b0b0b" stroke="#ffffff" stroke-width="%.2f"' % (0.6 * pt))
        for xi, yi, th in zip(x, y, view.hitT):
            kt = min(max(min(th, t_end) * k / total, 1e-4), 0.999)
            out.append('<circle cx="%.1f" cy="%.1f" r="%.2f" %s opacity="0"><animate attributeName="opacity" '
                       'calcMode="discrete" values="0;1" keyTimes="0;%.4f" %s/></circle>' % (xi, yi, r, style, kt, loop))
    for s, (px, py) in zip(view.photon_solids, view.photon_px):
        kt = min(max(min(float(dep.photons[s][0]), t_end) * k / total, 1e-4), 0.999)
        out.append('<circle cx="%.1f" cy="%.1f" r="%.2f" fill="%s" stroke="#0b0b0b" stroke-width="%.2f" opacity="0">'
                   '<animate attributeName="opacity" calcMode="discrete" values="0;1" keyTimes="0;%.4f" %s/></circle>'
                   % (px, py, 2.85 * pt, hexc(PHOTON_RGB), 0.7 * pt, kt, loop))
    out += lv.svg_items(view.top_items(), dpi)
    # clock: one text per ns, switched on in turn; none in the hold phase (as in the last frame)
    fs = a.font_size
    steps = np.arange(0, t_end, 1.0)
    for t0 in steps:
        k0 = max(t0 * k / total, 1e-4)
        k1 = min(min(t0 + 1.0, t_end) * k / total, 0.9999)
        if k1 <= k0:
            continue
        out.append('<text x="%.1f" y="%.1f" font-family="Helvetica, Arial, sans-serif" font-size="%.1f" '
                   'fill="#0b0b0b" opacity="0">t = %.0f ns<animate attributeName="opacity" calcMode="discrete" '
                   'values="0;1;0" keyTimes="0;%.4f;%.4f" %s/></text>'
                   % (view.W0 + 14 * pt, view.H - 14 * pt, fs * 1.4 * pt, t0, k0 if t0 > 0 else 0.0, k1, loop))
    out.append("</svg>")
    # the first clock text must be on from time 0: values 0;1;0 with keyTimes 0;0;k1 is not valid SMIL
    text = "\n".join(out) + "\n"
    text = text.replace('values="0;1;0" keyTimes="0;0.0000;', 'values="1;0" keyTimes="0;', 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# =====================================================================================================
#  one view -> all requested animation files
# =====================================================================================================

def animate_view(view, base, a):
    """Write base.gif, base.svg, ... for an EventView according to the options of solid_evview.
    Returns 0, or 1 if a format that should work failed."""
    formats = [f.strip().lower() for f in a.anim_formats.split(",") if f.strip()]
    t_end = a.t_end or view.end_time()
    status = 0
    if "svg" in formats:
        t0 = time.perf_counter()
        write_svg_animation(view, base + ".svg", t_end, duration=(a.frames - 1) / a.fps, hold=a.hold)
        lv.log("wrote %s (%.1f kB, %.2f s)" % (base + ".svg", os.path.getsize(base + ".svg") / 1024.0,
                                                time.perf_counter() - t0))
    raster = [f for f in formats if f in ("gif", "webp", "mp4", "webm")]
    unknown = [f for f in formats if f not in ("gif", "webp", "mp4", "webm", "svg")]
    if unknown:
        print("solid_evview: unknown animation format %s" % ", ".join(unknown))
        status = 1
    if not raster:
        return status
    try:
        import cairo  # noqa: F401
    except ImportError:
        print("solid_evview: animation frames need pycairo; only the animated SVG was written")
        return 1
    try:
        import PIL  # noqa: F401
        have_pil = True
    except ImportError:
        have_pil = False
    if "gif" in raster and not have_pil:
        print("solid_evview: GIF output needs Pillow; skipped")
        raster.remove("gif")
        status = 1
    if "webp" in raster and not webp_available():
        print("solid_evview: this Pillow cannot write WebP; %s.webp skipped" % os.path.basename(base))
        raster.remove("webp")
    ffmpeg = find_ffmpeg()
    for f in ("mp4", "webm"):
        if f in raster and not ffmpeg:
            print("solid_evview: no ffmpeg found (set SOLID_FFMPEG or put ffmpeg on PATH); %s.%s skipped"
                  % (os.path.basename(base), f))
            raster.remove(f)
    if not raster:
        return status
    scale = a.frame_scale
    while True:
        t0 = time.perf_counter()
        fr = FrameRenderer(view, scale)
        times = frame_times(a.frames, t_end)
        last = fr.frame(None)
        gif = GifWriter(last, a.fps, a.hold) if "gif" in raster else None
        videos = [VideoWriter(ffmpeg, base + "." + f, fr.W, fr.H, a.fps, a.hold) for f in ("mp4", "webm")
                  if f in raster]
        webp = [] if "webp" in raster else None
        for t in times:
            rgb = last if t is None else fr.frame(t)
            if gif:
                gif.add(rgb)
            for v in videos:
                v.add(rgb)
            if webp is not None:
                webp.append(rgb)
        t_frames = time.perf_counter() - t0
        lv.log("animation %s: %d frames %dx%d in %.1f s (%.3f s per frame), 0 to %.1f ns" % (
            view.kind, len(times), fr.W, fr.H, t_frames, t_frames / len(times), t_end))
        for v in videos:
            if v.close():
                lv.log("wrote %s (%.1f kB)" % (v.path, os.path.getsize(v.path) / 1024.0))
            else:
                status = 1
        if webp is not None:
            from PIL import Image
            t1 = time.perf_counter()
            ims = [Image.fromarray(x) for x in webp]
            dur = [int(round(1000.0 / a.fps))] * len(ims)
            dur[-1] += int(a.hold * 1000)
            ims[0].save(base + ".webp", save_all=True, append_images=ims[1:], duration=dur, loop=0, lossless=True,
                        method=4)
            lv.log("wrote %s (%.1f kB, %.1f s)" % (base + ".webp", os.path.getsize(base + ".webp") / 1024.0,
                                                    time.perf_counter() - t1))
        if not gif:
            break
        t1 = time.perf_counter()
        gif.save(base + ".gif")
        kb = os.path.getsize(base + ".gif") / 1024.0
        lv.log("wrote %s (%.1f kB, %.1f s)" % (base + ".gif", kb, time.perf_counter() - t1))
        if not a.gif_max_kb or kb <= a.gif_max_kb or scale < 0.2:
            break
        # too large: render again smaller (only the GIF; the other formats are done)
        scale *= max(0.5, min(0.9, math.sqrt(a.gif_max_kb / kb) * 0.95))
        raster = ["gif"]
        lv.log("GIF above %.0f kB: again at frame scale %.2f" % (a.gif_max_kb, scale))
    return status
