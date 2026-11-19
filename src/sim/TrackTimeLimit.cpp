// SPDX-License-Identifier: LGPL-3.0-or-later
//
// SoLIDTrackTimeLimit: a DDG4 stepping action that stops particles which are still being tracked long after the
// event.
//
// Why: a low-energy electron or positron born in a vacuum volume inside the solenoid field (beam pipe, cryostat
// gap, target chamber) can be trapped between two regions of stronger field. It loses no energy there, every step
// is short and completes, so Geant4's looper protection never acts and the event does not end: one such event
// stopped a production job for good (2026-11). All detector signals of interest arrive within a few hundred
// nanoseconds, so a particle that is still moving after MaxTime is of no interest to any read-out.
//
// Properties:
//   MaxTime      global time in ns after which a track is stopped (default 1e5 ns = 100 us)
//   ChargedOnly  act on charged particles only (default true): trapping is magnetic, and slow neutrons, whose
//                capture photons come late, are left alone
//   MaxPrint     number of stopped tracks that are reported in the log (default 10); a summary is always printed
//
// steering/solid_sim.py adds the action unless SOLID_MAX_TIME=0; SOLID_MAX_TIME=<ns> changes the limit.
// A stopped track deposits nothing further: its kinetic energy is dropped.

#include <DD4hep/InstanceCount.h>
#include <DD4hep/Printout.h>
#include <DDG4/Factories.h>
#include <DDG4/Geant4Context.h>
#include <DDG4/Geant4SteppingAction.h>

#include <G4ParticleDefinition.hh>
#include <G4Step.hh>
#include <G4SystemOfUnits.hh>
#include <G4Track.hh>
#include <G4VPhysicalVolume.hh>

#include <atomic>
#include <string>

namespace solid {

class SoLIDTrackTimeLimit : public dd4hep::sim::Geant4SteppingAction {
public:
  SoLIDTrackTimeLimit(dd4hep::sim::Geant4Context* ctx, const std::string& nam)
      : dd4hep::sim::Geant4SteppingAction(ctx, nam)
  {
    declareProperty("MaxTime", m_maxTime = 1.0e5);
    declareProperty("ChargedOnly", m_chargedOnly = true);
    declareProperty("MaxPrint", m_maxPrint = 10);
    dd4hep::InstanceCount::increment(this);
  }

  ~SoLIDTrackTimeLimit() override
  {
    if (!s_reported.exchange(true)) {
      dd4hep::printout(dd4hep::INFO, "SoLIDTrackTimeLimit", "%ld track%s stopped after %g ns%s", (long)s_stopped.load(),
                       s_stopped.load() == 1 ? "" : "s", m_maxTime, m_chargedOnly ? " (charged particles only)" : "");
    }
    dd4hep::InstanceCount::decrement(this);
  }

  void operator()(const G4Step* step, G4SteppingManager*) override
  {
    G4Track* trk = step->GetTrack();
    if (trk->GetGlobalTime() <= m_maxTime * CLHEP::ns)
      return;
    const G4ParticleDefinition* def = trk->GetDefinition();
    if (m_chargedOnly && def->GetPDGCharge() == 0.)
      return;
    trk->SetTrackStatus(fStopAndKill);
    const long n = ++s_stopped;
    if (n <= m_maxPrint) {
      const G4ThreeVector& x = trk->GetPosition();
      const G4VPhysicalVolume* pv = trk->GetVolume();
      dd4hep::printout(dd4hep::WARNING, "SoLIDTrackTimeLimit",
                       "stopped %s with %.4g MeV after %.3g ns and %.3g m at (%.1f, %.1f, %.1f) cm in %s", def->GetParticleName().c_str(),
                       trk->GetKineticEnergy() / CLHEP::MeV, trk->GetGlobalTime() / CLHEP::ns, trk->GetTrackLength() / CLHEP::m,
                       x.x() / CLHEP::cm, x.y() / CLHEP::cm, x.z() / CLHEP::cm, pv ? pv->GetName().c_str() : "?");
    }
  }

private:
  double m_maxTime = 1.0e5;
  bool m_chargedOnly = true;
  int m_maxPrint = 10;
  static std::atomic<long> s_stopped;
  static std::atomic<bool> s_reported;
};

std::atomic<long> SoLIDTrackTimeLimit::s_stopped{0};
std::atomic<bool> SoLIDTrackTimeLimit::s_reported{false};

} // namespace solid

DECLARE_GEANT4ACTION_NS(solid, SoLIDTrackTimeLimit)
