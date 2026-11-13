// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 SoLID collaboration
//
// SoLIDTrajectoryRecorder: DDG4 stepping action that records thinned trajectory points with global time and
// kinetic energy, for event displays and animations (bin/solid_evexport, bin/solid_evview).
// steering/solid_sim.py adds it when SOLID_TRAJ is set; by hand:
//
//   ddsim ... --action.step '{"name": "SoLIDTrajectoryRecorder/TrajRec", "parameter": {"OutputFile": "traj.bin"}}'
//
// What is recorded
//   Every track except optical photons whose kinetic energy at its vertex is at least MinEkin (default 5 MeV);
//   primaries (parent 0) are always kept.  A track gets a point at its start, then after each step that moved it
//   more than MaxDistance (default 20 mm) from the last written point or turned it by more than MaxAngle
//   (default 0.04 rad), and at its end.  The recorder does not use or change DDG4's particle handler, so it is
//   independent of which MCParticles the simulation keeps.
//
// File format (little endian; python/solid_evdisplay/trajfile.py reads it with numpy)
//   40-byte header:  char[8] "SOLIDTRJ", int32 version (1), int32 record size (40), float32 MinEkin [MeV],
//                    float32 MaxDistance [mm], float32 MaxAngle [rad], 12 bytes reserved (0)
//   40-byte records: int32 event (see below), int32 track, int32 parent, int32 pdg,
//                    float32 x, y, z [mm], float32 t [ns, global time], float32 ekin [MeV],
//                    int32 flags (bit 0: first point of the track, bit 1: last point)
//   so numpy.fromfile(path, dtype, offset=40) gives all points.  The points of one event are contiguous, and
//   inside an event the points of one track are in time order (tracks interleave only as Geant4 suspends and
//   resumes them).  Events appear in the order they finish, which is not the event order in a multi-threaded
//   run: select them by the event field.
//
// Event number
//   The event field is the number ddsim writes into the EDM4hep EventHeader: the Geant4 event ID for the
//   particle gun, the event number of the input file when an input reader passed one on (HepMC3: the number
//   on the E line).  With input files it is therefore the generator's numbering, which must be unique for
//   solid_evexport to match trajectories and hits.  ddsim's --meta.eventNumberOffset is not applied here.
//
// Part of the events, and timing
//   Every = N records only the events whose number is Remainder modulo N (default 1: all); for the others the
//   action returns after one comparison per step.  TimingFile = FILE appends one text line per event at its
//   first step: event number, 1 if it is recorded, CPU time of the thread and real time in seconds.  The
//   difference of two consecutive lines is the cost of one event with everything between two events.
//   scripts/event_display/recorder_overhead.py uses both to compare recorded and unrecorded events inside
//   one job, where they see the same machine load.
//
// Threads
//   ddsim creates one instance per worker thread.  Each instance buffers the points of its current event and
//   appends them to the file in one write at the end of the event, under a lock shared by all instances that
//   write the same file.

#include <DD4hep/InstanceCount.h>
#include <DDG4/EventParameters.h>
#include <DDG4/Factories.h>
#include <DDG4/Geant4Context.h>
#include <DDG4/Geant4EventAction.h>
#include <DDG4/Geant4SteppingAction.h>

#include <G4Event.hh>
#include <G4EventManager.hh>
#include <G4OpticalPhoton.hh>
#include <G4Step.hh>
#include <G4StepPoint.hh>
#include <G4Track.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace solid {

namespace {

#pragma pack(push, 1)
  struct TrajHeader {
    char    magic[8];
    int32_t version;
    int32_t record_size;
    float   min_ekin;
    float   max_distance;
    float   max_angle;
    char    reserved[12];
  };
  struct TrajPoint {
    int32_t event, track, parent, pdg;
    float   x, y, z, t, ekin;
    int32_t flags;
  };
#pragma pack(pop)
  static_assert(sizeof(TrajHeader) == 40, "trajectory file header size");
  static_assert(sizeof(TrajPoint) == 40, "trajectory file record size");

  // One output file, shared by the recorder instances of all threads.
  class TrajFile {
  public:
    TrajFile(const std::string& path, const TrajHeader& header) : m_path(path)
    {
      m_fp = std::fopen(path.c_str(), "wb");
      if (m_fp != nullptr) {
        std::fwrite(&header, sizeof(header), 1, m_fp);
      }
    }
    ~TrajFile()
    {
      if (m_fp != nullptr) {
        std::fclose(m_fp);
      }
    }
    bool ok() const { return m_fp != nullptr; }
    void append(const std::vector<TrajPoint>& points)
    {
      if (points.empty() || m_fp == nullptr) {
        return;
      }
      std::lock_guard<std::mutex> lock(m_mutex);
      std::fwrite(points.data(), sizeof(TrajPoint), points.size(), m_fp);
      std::fflush(m_fp);
      m_points += static_cast<long>(points.size());
      ++m_events;
    }
    long points() const { return m_points; }
    long events() const { return m_events; }

    static std::shared_ptr<TrajFile> open(const std::string& path, const TrajHeader& header)
    {
      static std::mutex                                     registry_mutex;
      static std::map<std::string, std::weak_ptr<TrajFile>> registry;
      std::lock_guard<std::mutex>                           lock(registry_mutex);
      auto                                                  file = registry[path].lock();
      if (!file) {
        file           = std::make_shared<TrajFile>(path, header);
        registry[path] = file;
      }
      return file;
    }

  private:
    std::string m_path;
    std::FILE*  m_fp = nullptr;
    std::mutex  m_mutex;
    long        m_points = 0;
    long        m_events = 0;
  };

} // namespace

class SoLIDTrajectoryRecorder : public dd4hep::sim::Geant4SteppingAction {
public:
  SoLIDTrajectoryRecorder(dd4hep::sim::Geant4Context* ctxt, const std::string& nam)
    : dd4hep::sim::Geant4SteppingAction(ctxt, nam)
  {
    declareProperty("OutputFile", m_output = "solid_traj.bin");
    declareProperty("MinEkin", m_minEkin = 5.0);         // MeV, kinetic energy at the track vertex
    declareProperty("MaxDistance", m_maxDistance = 20.0); // mm
    declareProperty("MaxAngle", m_maxAngle = 0.04);       // rad
    declareProperty("Every", m_every = 1);                // record events with number % Every == Remainder
    declareProperty("Remainder", m_remainder = 0);
    declareProperty("TimingFile", m_timing = "");         // CPU and real time at the first step of each event
    dd4hep::InstanceCount::increment(this);
  }

  ~SoLIDTrajectoryRecorder() override
  {
    flush();
    if (m_file) {
      info("%ld steps seen, %ld points of %ld tracks buffered by this instance; %s holds %ld points of %ld events",
           m_nsteps, m_npoints, m_ntracks, m_output.c_str(), m_file->points(), m_file->events());
    }
    dd4hep::InstanceCount::decrement(this);
  }

  void operator()(const G4Step* step, G4SteppingManager*) override
  {
    const G4Track* track = step->GetTrack();
    if (track->GetDefinition() == G4OpticalPhoton::Definition()) {
      return;
    }
    ++m_nsteps;
    const G4Event* event = G4EventManager::GetEventManager()->GetConstCurrentEvent();
    const int      evt   = event != nullptr ? event->GetEventID() : -1;
    if (evt != m_event) {
      begin_event(evt);
    }
    if (!m_record) {
      return;
    }
    const G4StepPoint* pre  = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();
    const int          id   = track->GetTrackID();
    if (id != m_track || track->GetCurrentStepNumber() == 1) {
      // a new track, or one that Geant4 resumes after tracking others
      m_track = id;
      m_skip  = track->GetParentID() != 0 && track->GetVertexKineticEnergy() < m_minEkin;
      if (m_skip) {
        return;
      }
      if (track->GetCurrentStepNumber() == 1) {
        ++m_ntracks;
        put(track, pre->GetPosition(), pre->GetGlobalTime(), pre->GetKineticEnergy(), 1);
      }
      m_lastPos = pre->GetPosition();
      m_lastDir = pre->GetMomentumDirection();
    }
    if (m_skip) {
      return;
    }
    const G4ThreeVector& pos  = post->GetPosition();
    const G4ThreeVector& dir  = post->GetMomentumDirection();
    const bool           last = track->GetTrackStatus() != fAlive && track->GetTrackStatus() != fSuspend;
    if (last || (pos - m_lastPos).mag2() > m_maxDistance * m_maxDistance || dir.dot(m_lastDir) < m_cosMaxAngle) {
      put(track, pos, post->GetGlobalTime(), post->GetKineticEnergy(), last ? 2 : 0);
      m_lastPos = pos;
      m_lastDir = dir;
    }
  }

private:
  void begin_event(int evt)
  {
    flush();
    if (!m_file) {
      TrajHeader header;
      std::memset(&header, 0, sizeof(header));
      std::memcpy(header.magic, "SOLIDTRJ", 8);
      header.version      = 1;
      header.record_size  = sizeof(TrajPoint);
      header.min_ekin     = static_cast<float>(m_minEkin);
      header.max_distance = static_cast<float>(m_maxDistance);
      header.max_angle    = static_cast<float>(m_maxAngle);
      m_file              = TrajFile::open(m_output, header);
      if (!m_file->ok()) {
        // a drawing aid must not stop a simulation job
        error("cannot write %s: no trajectories are recorded", m_output.c_str());
      }
      m_cosMaxAngle = std::cos(m_maxAngle);
      // write the event when it ends, not only when the next one starts
      context()->eventAction().callAtEnd(this, &SoLIDTrajectoryRecorder::end_event);
    }
    m_event = evt;
    m_track = -1;
    // the number Geant4Output2EDM4hep gives this event: the one an input reader passed on, else the Geant4 ID
    const auto* par = context()->event().extension<dd4hep::sim::EventParameters>(false);
    m_number        = par != nullptr ? par->eventNumber() : evt;
    m_record        = m_every <= 1 || ((m_number % m_every) + m_every) % m_every == m_remainder;
    if (!m_timing.empty()) {
      timespec cpu, real;
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu);
      clock_gettime(CLOCK_MONOTONIC, &real);
      if (std::FILE* f = std::fopen(m_timing.c_str(), "a")) {
        std::fprintf(f, "%d %d %.6f %.6f\n", m_number, m_record ? 1 : 0, cpu.tv_sec + 1e-9 * cpu.tv_nsec,
                     real.tv_sec + 1e-9 * real.tv_nsec);
        std::fclose(f);
      }
    }
  }

  void end_event(const G4Event*)
  {
    flush();
    m_event = -1;
  }

  void flush()
  {
    if (m_file && !m_buffer.empty()) {
      m_file->append(m_buffer);
    }
    m_buffer.clear();
  }

  void put(const G4Track* track, const G4ThreeVector& p, double t, double ekin, int flags)
  {
    m_buffer.push_back(TrajPoint{m_number, track->GetTrackID(), track->GetParentID(),
                                 track->GetDefinition()->GetPDGEncoding(), float(p.x()), float(p.y()), float(p.z()),
                                 float(t), float(ekin), flags});
    ++m_npoints;
  }

  std::string               m_output;
  double                    m_minEkin     = 5.0;
  double                    m_maxDistance = 20.0;
  double                    m_maxAngle    = 0.04;
  double                    m_cosMaxAngle = 1.0;
  int                       m_every       = 1;
  int                       m_remainder   = 0;
  std::string               m_timing;
  bool                      m_record      = true;
  std::shared_ptr<TrajFile> m_file;
  std::vector<TrajPoint>    m_buffer;
  int                       m_event  = -1; // Geant4 event ID of the event being recorded
  int                       m_number = -1; // its number in the output file
  int                       m_track = -1;
  bool                      m_skip  = false;
  G4ThreeVector             m_lastPos, m_lastDir;
  long                      m_nsteps = 0, m_npoints = 0, m_ntracks = 0;
};

} // namespace solid

DECLARE_GEANT4ACTION_NS(solid, SoLIDTrajectoryRecorder)
