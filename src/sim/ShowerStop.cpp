// SPDX-License-Identifier: LGPL-3.0-or-later
//
// SoLIDShowerStop: a DDG4 stacking action for acceptance studies. It drops the low-energy secondaries that are born
// inside the shower modules of a calorimeter, so that only the energetic core of a shower develops there.
//
// Why: a shower in the shashlik modules (194 layers of lead and scintillator per module) takes most of the
// simulation time of an electron, nearly all of it in particles of a few MeV, and an acceptance map only asks
// whether the particle reached the calorimeter. With this action the incoming particles and their secondaries
// above MinEkin are still tracked and leave their ionization in the scintillator; the energy of the dropped
// secondaries is lost. Everything in front of the shower modules is simulated in full: target, windows, GEM
// planes, Cherenkov gas, scintillator pads, and the preshower with its lead radiator. The radiator has to stay:
// a low-energy electron often ends in it, and then only its shower reaches the scintillator behind
// (with the radiator included, 3 % of the electrons that arrive leave no signal; docs/analysis.md).
//
// NOT for anything that needs the calorimeter response: energy, E/p, shower shape, trigger thresholds, particle
// identification, backgrounds from albedo. For those run without it.
//
// Properties:
//   Volumes   name fragments; a secondary is a candidate when the name of the volume it is born in contains one
//             of them (default "ECShower_": the shower modules of the forward and large-angle calorimeters)
//   MinEkin   MeV; candidates born with less kinetic energy are dropped (default 20)
//
// steering/solid_sim.py adds the action with SOLID_FAST=1 (SOLID_FAST_MIN_EKIN changes MinEkin).

#include <DD4hep/InstanceCount.h>
#include <DD4hep/Printout.h>
#include <DDG4/Factories.h>
#include <DDG4/Geant4Context.h>
#include <DDG4/Geant4StackingAction.h>

#include <G4LogicalVolume.hh>
#include <G4SystemOfUnits.hh>
#include <G4Track.hh>
#include <G4VPhysicalVolume.hh>
#include <G4VTouchable.hh>

#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

namespace solid {

class SoLIDShowerStop : public dd4hep::sim::Geant4StackingAction {
public:
  SoLIDShowerStop(dd4hep::sim::Geant4Context* ctx, const std::string& nam)
      : dd4hep::sim::Geant4StackingAction(ctx, nam)
  {
    declareProperty("Volumes", m_fragments = {"ECShower_"});
    declareProperty("MinEkin", m_minEkin = 20.0);
    dd4hep::InstanceCount::increment(this);
  }

  ~SoLIDShowerStop() override
  {
    if (!s_reported.exchange(true)) {
      dd4hep::printout(dd4hep::INFO, "SoLIDShowerStop",
                       "%ld secondaries below %g MeV born in calorimeter shower modules were dropped: NO calorimeter response "
                       "in this run", (long)s_dropped.load(), m_minEkin);
    }
    dd4hep::InstanceCount::decrement(this);
  }

  dd4hep::sim::TrackClassification classifyNewTrack(G4StackManager*, const G4Track* track) override
  {
    if (track->GetParentID() == 0 || track->GetKineticEnergy() >= m_minEkin * CLHEP::MeV)
      return dd4hep::sim::TrackClassification();
    const G4VTouchable* touchable = track->GetTouchable();
    const G4VPhysicalVolume* pv = touchable ? touchable->GetVolume() : nullptr;
    if (pv == nullptr)
      return dd4hep::sim::TrackClassification();
    const G4LogicalVolume* lv = pv->GetLogicalVolume();
    auto it = m_cache.find(lv);
    if (it == m_cache.end()) {
      bool match = false;
      const std::string name = lv->GetName();
      for (const auto& f : m_fragments)
        match = match || name.find(f) != std::string::npos;
      it = m_cache.emplace(lv, match).first;
    }
    if (!it->second)
      return dd4hep::sim::TrackClassification();
    ++s_dropped;
    return dd4hep::sim::TrackClassification(fKill);
  }

private:
  std::vector<std::string> m_fragments;
  double m_minEkin = 20.0;
  std::unordered_map<const G4LogicalVolume*, bool> m_cache;
  static std::atomic<long> s_dropped;
  static std::atomic<bool> s_reported;
};

std::atomic<long> SoLIDShowerStop::s_dropped{0};
std::atomic<bool> SoLIDShowerStop::s_reported{false};

} // namespace solid

DECLARE_GEANT4ACTION_NS(solid, SoLIDShowerStop)
