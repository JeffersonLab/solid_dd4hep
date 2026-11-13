#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Measure what the trajectory recorder (SOLID_TRAJ) costs in simulation time.

usage (inside eic-shell / solid-shell):
    recorder_overhead.py [-c CONFIG] [-n EVENTS] [--reps R] [--input EVENTS.hepmc] [--seed S] [-o DIR]
    recorder_overhead.py --jobs [--chunks K] ...

Default: recorded and unrecorded events inside one job
    Two ddsim jobs simulate the same EVENTS + 1 events with the same seed. The recorder records the even
    events in job A and the odd events in job B (SOLID_TRAJ_EVERY=2:0 and 2:1) and writes the CPU time of the
    thread at the first step of every event (SOLID_TRAJ_TIMING). The time between two such stamps is one
    event with everything that belongs to it: tracking, output, the recorder's write, reading the next event.
    Every event is so simulated once with and once without recording, and inside each job the two kinds
    alternate, so that both see the same machine load. With the sums A_even, A_odd, B_even, B_odd of the
    event times

        recorded / unrecorded = sqrt( (A_even / B_even) * (B_odd / A_odd) )

    in which a speed difference between the two jobs cancels. This is repeated R times (default 3); the
    result is the mean of the R values with their standard deviation. What it measures is recording against
    an idle recorder: the unrecorded events still pay one call of the stepping action and one comparison per
    step.

--jobs: whole jobs with and without the recorder
    The same events are simulated R times without and R times with the recorder, alternating (off on, on
    off, ...). Per run it takes the user + system time of the Geant4 event loop (the "User=...s Real=...s
    Sys=...s" line of /run/verbose 1) and of the whole process, and prints means, minima and the mean of the
    paired differences with its standard error. With --chunks K, pair r simulates the events r mod K of the
    input. This compares against no recorder at all, but run-to-run differences of the machine enter in
    full: on a shared machine expect a few per cent of scatter per pair.

The trajectory file is deleted after each run; its size is reported. SOLID_OPTICAL=0 unless the environment
says otherwise. Results also go to DIR/overhead.json.
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

G4_TIMER = re.compile(r'User=([0-9.eE+-]+)s\s+Real=([0-9.eE+-]+)s\s+Sys=([0-9.eE+-]+)s')


def ddsim(a, tag, events, env_extra, skip=0, seed=None):
    """one ddsim run; returns (log text, rusage, wall time)"""
    path = os.environ.get('DETECTOR_PATH', '.')
    out = os.path.join(a.outdir, 'run.edm4hep.root')
    log = os.path.join(a.outdir, 'ddsim_%s.log' % tag)
    cmd = ['ddsim', '--steeringFile', os.path.join(path, 'steering', 'solid_sim.py'), '--compactFile',
           os.path.join(path, a.config + '.xml'), '-N', str(events), '--random.seed',
           str(a.seed if seed is None else seed), '--outputFile', out, '--ui.commandsPreRun', '/run/verbose 1']
    env = dict(os.environ, DETECTOR_CONFIG=a.config)
    env.setdefault('SOLID_OPTICAL', '0')
    for k in ('SOLID_TRAJ', 'SOLID_SCENE', 'SOLID_TRAJ_EVERY', 'SOLID_TRAJ_TIMING'):
        env.pop(k, None)
    if a.input:
        cmd += ['--inputFiles'] + a.input + (['--skipNEvents', str(skip)] if skip else [])
        env['SOLID_GUN'] = '0'
    env.update(env_extra)
    t0 = time.perf_counter()
    with open(log, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=a.outdir)
        _, status, ru = os.wait4(p.pid, 0)
    wall = time.perf_counter() - t0
    text = open(log, errors='replace').read()
    if status != 0 or not os.path.isfile(out):
        sys.exit('recorder_overhead: ddsim failed, see %s' % log)
    size = os.path.getsize(out)
    os.remove(out)
    return text, ru, wall, size


def sd(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


# ---------------------------------------------------------------------------------------------------------
def inside_one_job(a):
    traj, timing = os.path.join(a.outdir, 'traj.bin'), os.path.join(a.outdir, 'timing.txt')
    reps = []
    for rep in range(a.reps):
        sums, nrec, tbytes = {}, 0, 0
        for job, rem in (('A', 0), ('B', 1)) if rep % 2 == 0 else (('B', 1), ('A', 0)):
            for f in (traj, timing):
                if os.path.isfile(f):
                    os.remove(f)
            ddsim(a, '%s%d' % (job, rep), a.events + 1,
                  dict(SOLID_TRAJ=traj, SOLID_TRAJ_EVERY='2:%d' % rem, SOLID_TRAJ_TIMING=timing))
            rows = [line.split() for line in open(timing)]
            if len(rows) != a.events + 1:
                sys.exit('recorder_overhead: %d timing lines for %d events (one thread only)' % (
                    len(rows), a.events + 1))
            for (num, rec, cpu, real), nxt in zip(rows[:-1], rows[1:]):
                key = (job, int(num) % 2)
                if (int(rec) == 1) != (int(num) % 2 == rem):
                    sys.exit('recorder_overhead: event %s has the wrong recording flag' % num)
                s = sums.setdefault(key, [0.0, 0.0, 0])
                s[0] += float(nxt[2]) - float(cpu)
                s[1] += float(nxt[3]) - float(real)
                s[2] += 1
                nrec += int(rec)
            tbytes += os.path.getsize(traj)
            os.remove(traj)
        r = dict(rep=rep, traj_bytes_per_recorded_event=tbytes / max(nrec, 1))
        for i, name in ((0, 'cpu'), (1, 'real')):
            ae, ao, be, bo = (sums[k][i] for k in (('A', 0), ('A', 1), ('B', 0), ('B', 1)))
            r[name] = 100.0 * (math.sqrt((ae / be) * (bo / ao)) - 1.0)
            r[name + '_recorded_s'] = (ae + bo) / (sums[('A', 0)][2] + sums[('B', 1)][2])
            r[name + '_unrecorded_s'] = (be + ao) / (sums[('B', 0)][2] + sums[('A', 1)][2])
            r[name + '_job_ratio'] = (ae + ao) / (be + bo)
        reps.append(r)
        print('repetition %d: recording costs %+.2f %% CPU, %+.2f %% real time; %.3f s CPU per unrecorded '
              'event; job A / job B = %.3f' % (rep, r['cpu'], r['real'], r['cpu_unrecorded_s'],
                                              r['cpu_job_ratio']), flush=True)
    res = dict(method='inside one job', reps=reps)
    for name in ('cpu', 'real'):
        v = [r[name] for r in reps]
        res[name + '_mean'], res[name + '_sd'] = statistics.mean(v), sd(v)
    res['cpu_unrecorded_s'] = statistics.mean(r['cpu_unrecorded_s'] for r in reps)
    res['traj_mb_per_event'] = statistics.mean(r['traj_bytes_per_recorded_event'] for r in reps) / 1e6
    print('\n%s, %d events, %d repetitions of two jobs, %s' % (
        a.config, a.events, a.reps, ', '.join(a.input) if a.input else 'particle gun'))
    print('recording against an idle recorder: CPU time %+.2f +- %.2f %%, real time %+.2f +- %.2f %% '
          '(mean and standard deviation of the repetitions)' % (res['cpu_mean'], res['cpu_sd'], res['real_mean'],
                                                              res['real_sd']))
    print('%.3f s CPU per unrecorded event; trajectory file %.2f MB per recorded event' % (
        res['cpu_unrecorded_s'], res['traj_mb_per_event']))
    return res


# ---------------------------------------------------------------------------------------------------------
def whole_jobs(a):
    traj = os.path.join(a.outdir, 'traj.bin')
    runs = []
    for rep in range(a.reps):
        chunk = rep % a.chunks
        for with_traj in ((False, True) if rep % 2 == 0 else (True, False)):
            tag = '%s%d' % ('on' if with_traj else 'off', rep)
            text, ru, wall, size = ddsim(a, tag, a.events, dict(SOLID_TRAJ=traj) if with_traj else {},
                                         skip=chunk * a.events, seed=a.seed if a.input else a.seed + chunk)
            m = G4_TIMER.findall(text)
            if not m:
                sys.exit('recorder_overhead: no run timer in the log of %s' % tag)
            user, real, syst = (float(v) for v in m[-1])
            r = dict(tag=tag, rep=rep, traj=with_traj, loop_cpu=user + syst, loop_real=real,
                     job_cpu=ru.ru_utime + ru.ru_stime, job_wall=wall, edm4hep_bytes=size)
            if with_traj:
                r['traj_bytes'] = os.path.getsize(traj)
                os.remove(traj)
            runs.append(r)
            print('%-8s loop CPU %8.2f s  loop real %8.2f s  job CPU %8.2f s  job wall %8.2f s%s' % (
                tag, r['loop_cpu'], r['loop_real'], r['job_cpu'], r['job_wall'],
                '  traj %.1f MB' % (r['traj_bytes'] / 1e6) if with_traj else ''), flush=True)
    off = sorted((r for r in runs if not r['traj']), key=lambda r: r['rep'])
    on = sorted((r for r in runs if r['traj']), key=lambda r: r['rep'])
    res = dict(method='whole jobs', chunks=a.chunks, runs=runs)
    print('\n%s, %d events per run (%d different sets), %d runs per mode, %s' % (
        a.config, a.events, min(a.chunks, a.reps), a.reps, ', '.join(a.input) if a.input else 'particle gun'))
    print('%-10s %22s %22s %12s %12s %12s' % ('', 'without recorder [s]', 'with recorder [s]', 'means',
                                              'minima', 'pairs'))
    for key, name in (('loop_cpu', 'loop CPU'), ('loop_real', 'loop real'), ('job_cpu', 'job CPU')):
        x, y = [r[key] for r in off], [r[key] for r in on]
        pairs = [100.0 * (q - p) / p for p, q in zip(x, y)]
        res[key] = dict(off_mean=statistics.mean(x), off_sd=sd(x), on_mean=statistics.mean(y), on_sd=sd(y),
                        overhead_means=100.0 * (statistics.mean(y) / statistics.mean(x) - 1.0),
                        overhead_minima=100.0 * (min(y) / min(x) - 1.0), overhead_pairs=statistics.mean(pairs),
                        overhead_pairs_sd=sd(pairs), overhead_pairs_sem=sd(pairs) / len(pairs) ** 0.5)
        v = res[key]
        print('%-10s %12.2f +- %6.2f %12.2f +- %6.2f %+11.2f%% %+11.2f%% %+6.2f +- %4.2f%%' % (
            name, v['off_mean'], v['off_sd'], v['on_mean'], v['on_sd'], v['overhead_means'],
            v['overhead_minima'], v['overhead_pairs'], v['overhead_pairs_sem']))
    tb = statistics.mean(r['traj_bytes'] for r in on)
    res.update(traj_mb_per_event=tb / 1e6 / a.events, edm4hep_bytes_off=off[0]['edm4hep_bytes'],
               edm4hep_bytes_on=on[0]['edm4hep_bytes'])
    print('+- is the standard deviation of the runs; for the pairs, the standard error of their mean')
    print('trajectory file: %.2f MB per event; EDM4hep file %d bytes without, %d with' % (
        res['traj_mb_per_event'], res['edm4hep_bytes_off'], res['edm4hep_bytes_on']))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', default=os.environ.get('DETECTOR_CONFIG', 'solid_JPsi_lite'))
    ap.add_argument('-n', '--events', type=int, default=100)
    ap.add_argument('--reps', type=int, default=3, help='repetitions (default 3)')
    ap.add_argument('--jobs', action='store_true', help='compare whole jobs with and without the recorder')
    ap.add_argument('--chunks', type=int, default=1, help='--jobs: number of different sets of EVENTS events')
    ap.add_argument('--input', nargs='+', default=None, help='generator file(s) for --inputFiles (default: '
                    'the particle gun of the steering file)')
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('-o', '--outdir', default='recorder_overhead')
    a = ap.parse_args()
    a.outdir = os.path.abspath(a.outdir)
    if a.input:
        a.input = [os.path.abspath(f) for f in a.input]
    os.makedirs(a.outdir, exist_ok=True)
    res = whole_jobs(a) if a.jobs else inside_one_job(a)
    res.update(config=a.config, events=a.events, input=a.input, seed=a.seed)
    json.dump(res, open(os.path.join(a.outdir, 'overhead.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
