# Event generators

| process | generator | where | output for ddsim |
|---|---|---|---|
| exclusive J/ψ, photo- and electroproduction | lAger 3.7.0 | `/opt/lager` in the `solid_xl` image | HepMC3 through `scripts/event_display/hepmc_prepare.py` ([event_display.md](event_display.md#generators)) |
| SIDIS with radiative corrections | SIDIS-RC EvGen (`sidisgen`) | `/opt/sidis` in the `solid_xl` image | HepMC3 through `scripts/generators/sidisgen2hepmc.py` (this page) |
| single particles for acceptance maps | `scripts/analysis/gun_events.py` | this repository | HepMC3 ([analysis.md](analysis.md)) |
| beam on target, for backgrounds | `scripts/event_display/beam_on_target.py` | this repository | HepMC3 |

Not yet here: the inclusive-electron and hadron-background generators of the GEMC studies (`evgen_inclusive_e`,
`evgen_bggen`) and a Bethe-Heitler generator.

## SIDIS: SIDIS-RC EvGen

D. Byer et al., Comput. Phys. Commun. 287 (2023) 108702, arXiv:2210.03785; https://github.com/duanebyer/sidis
(GPL-3.0). Semi-inclusive DIS on the proton, neutron or deuteron with the lowest-order QED radiative corrections,
polarized beam and target, structure functions in the Wandzura-Wilczek-type approximation. The image builds
commit `8f9881a` with one patch (below); outside the image, `cmake` with ROOT and GSL is enough.

```
cp <solid_dd4hep checkout>/scripts/generators/sidis_solid.params .
sidisgen --initialize sidis_solid.params       # once per parameter set: about 20 minutes, writes gen.root
sidisgen --generate sidis_solid.params         # 100 000 events in a few seconds, writes event.root
sidisgen2hepmc.py event.root -o sidis.hepmc --z=-370:-330 --summary sidis.json
SOLID_GUN=0 ddsim --steeringFile $DETECTOR_PATH/steering/solid_sim.py --inputFiles sidis.hepmc -N 100000 ...
```

`scripts/generators/sidis_solid.params` is the parameter set for SoLID at 11 GeV: π⁺ on the neutron, Q² > 1 GeV²,
W > 2.3 GeV, 0.3 < z < 0.7, scattered electron at 6°–28° and hadron at 6°–18° with 0.8–8 GeV each. The angular
and momentum cuts only keep the generator from producing events far outside the detector; the simulation decides
the acceptance. Inside these cuts the cross section for π⁺ on the proton is 5.8 nb, of which a third are events
with a radiated photon. Change `setup.target` (`p`, `n`, `d`), `setup.hadron` (`pi+`, `pi-`, `K+`, ...),
`setup.beam_energy`, `setup.beam_pol` and `setup.target_pol` for the other samples; every change needs its own
`--initialize`.

**The neutron needs our patch.** The generator's built-in structure functions are the proton's, and it used them
for every target: with `setup.target n` only the nucleon mass changed, and the cross sections on the neutron
equalled those on the proton (5.79 nb for π⁺ in the cuts above). `containers/patches/sidis-neutron-isospin.patch`
gives the neutron by isospin rotation of the proton's parton distributions (u ↔ d, ū ↔ d̄), for the unpolarized
and all polarized distributions; charges and fragmentation functions keep the flavour of the struck quark. In a
box at x = 0.2–0.3 the Born cross sections then stand as p π⁺ : p π⁻ : n π⁺ : n π⁻ = 7.2 : 3.9 : 4.3 : 3.7, and
the neutron-to-proton ratio summed over both pions is 0.73, as F₂ⁿ/F₂ᵖ there. The deuteron is not implemented:
the generator now says that it uses the proton set.

**Polarized ³He** is not a target of the generator. Generate on the neutron and on the proton and combine
afterwards: the asymmetry comes from the neutron with its effective polarization in ³He, and the two protons
dilute it.

### What the converter writes

`sidisgen2hepmc.py` needs the momenta in the event file (`file.write_momenta on`). Per event: the beam lepton and
the target nucleon at rest as incoming particles; the scattered lepton, the hadron and, for a radiative event,
the photon as final-state particles; a vertex uniform along the target; and the weight.

The weight is the cross section carried by the event in nanobarn: the generator's weight divided by the norm of
its event type (`stats/norm` in the file), times 0.3894 × 10⁶ nb GeV². The weights of all generated events add up
to the cross section inside the cuts. ddsim keeps the weight in the event header, so a rate is the luminosity
times the sum of the weights of the selected events.

First pass through the chain (200 events on the neutron, SoLID-Lite stage 2 at full azimuth, fast mode): the
header weights add up to the converter's cross section, and the particle table gives the electron and the pion of
each event for coincidence counts ([analysis.md](analysis.md)).
