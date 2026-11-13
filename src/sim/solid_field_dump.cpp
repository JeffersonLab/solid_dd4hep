// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (C) 2026 SoLID collaboration
//
// solid_field_dump: sample the magnetic field of a DD4hep compact file on a plane and write it as field.npz.
// The field is the sum of all <field> overlays of the configuration, as the simulation sees it (solenoid map,
// target magnet, ...). solid_lineview --field FILE.npz draws it under the section in the same plane.

#include <DD4hep/Detector.h>
#include <DD4hep/Fields.h>
#include <DD4hep/Printout.h>
#include <Evaluator/DD4hepUnits.h>

#include <TGeoBBox.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>

#include "NpzWriter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const char* const USAGE = R"(usage: solid_field_dump [COMPACT.xml] [-o FIELD.npz] [--plane x=0|y=0|z=VALUE]
                        [--u-range LO HI] [--v-range LO HI] [--step S] [-v]

Sample the magnetic field of the configuration (all <field> overlays summed, as in the simulation) on a plane.
solid_lineview --field FIELD.npz draws it under the section in that plane (make_solid_views --views field_yz).

  COMPACT.xml     compact file (default: $DETECTOR_PATH/$DETECTOR_CONFIG.xml, set by thissolid.sh)
  -o, --output    file to write (default: <configuration>.field.npz)
  --plane SPEC    x=VALUE (default x=0), y=VALUE or z=VALUE. In-plane axes (u, v): x -> (z, y), y -> (z, x),
                  z -> (x, y)
  --u-range LO HI range along u (default: the box around the detector plus 1.5 m on each side)
  --v-range LO HI range along v
  --step S        grid step (default 1cm); the grid is aligned to multiples of S, so it holds the beam axis
  -v, --verbose   print DD4hep messages of level INFO
  -h, --help      this text
Lengths are in mm unless they carry a unit: 10mm, 1cm, 2.5m.

The file holds: u, v (grid coordinates, mm), B (nu x nv x 3, tesla, hall-frame components), origin, u_dir,
v_dir, normal, meta (JSON).
)";

// "12.5", "1cm", "2.5m" -> mm
bool parseLength(const std::string& s, double& mm)
{
  char*        end = nullptr;
  const double v   = std::strtod(s.c_str(), &end);
  if (end == s.c_str()) {
    return false;
  }
  const std::string unit(end);
  if (unit.empty() || unit == "mm") {
    mm = v;
  }
  else if (unit == "cm") {
    mm = 10.0 * v;
  }
  else if (unit == "m") {
    mm = 1000.0 * v;
  }
  else {
    return false;
  }
  return true;
}

std::string jsonString(const std::string& s)
{
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
    }
    out += c;
  }
  return out + "\"";
}

// box around the daughters of the world volume, in mm: lo[3], hi[3]
void detectorBox(const dd4hep::Detector& description, double lo[3], double hi[3])
{
  std::fill(lo, lo + 3, 1e99);
  std::fill(hi, hi + 3, -1e99);
  TGeoVolume* world = description.worldVolume().ptr();
  for (int i = 0; i < world->GetNdaughters(); ++i) {
    TGeoNode* node = world->GetNode(i);
    auto*     box  = dynamic_cast<TGeoBBox*>(node->GetVolume()->GetShape());
    if (box == nullptr) {
      continue;
    }
    const double  d[3] = {box->GetDX(), box->GetDY(), box->GetDZ()};
    const double* o    = box->GetOrigin();
    for (int c = 0; c < 8; ++c) {
      const double local[3] = {o[0] + ((c & 1) ? d[0] : -d[0]), o[1] + ((c & 2) ? d[1] : -d[1]),
                               o[2] + ((c & 4) ? d[2] : -d[2])};
      double       master[3];
      node->LocalToMaster(local, master);
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], master[k] / dd4hep::mm);
        hi[k] = std::max(hi[k], master[k] / dd4hep::mm);
      }
    }
  }
}

} // namespace

int main(int argc, char** argv)
{
  std::string compact, output, plane = "x=0";
  double      step = 10.0, urange[2] = {0, 0}, vrange[2] = {0, 0};
  bool        haveU = false, haveV = false, verbose = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a     = argv[i];
    auto              value = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "solid_field_dump: " << what << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    auto length = [&](const char* what) -> double {
      double            mm = 0;
      const std::string s  = value(what);
      if (!parseLength(s, mm)) {
        std::cerr << "solid_field_dump: " << what << ": '" << s << "' is not a length (e.g. 0, 10mm, 1cm, 2.5m)\n";
        std::exit(2);
      }
      return mm;
    };
    if (a == "-h" || a == "--help") {
      std::cout << USAGE;
      return 0;
    }
    else if (a == "-o" || a == "--output") {
      output = value("--output");
    }
    else if (a == "--plane") {
      plane = value("--plane");
    }
    else if (a == "--u-range") {
      urange[0] = length("--u-range");
      urange[1] = length("--u-range");
      haveU     = true;
    }
    else if (a == "--v-range") {
      vrange[0] = length("--v-range");
      vrange[1] = length("--v-range");
      haveV     = true;
    }
    else if (a == "--step") {
      step = length("--step");
    }
    else if (a == "-v" || a == "--verbose") {
      verbose = true;
    }
    else if (!a.empty() && a[0] == '-' && !(a.size() > 1 && (std::isdigit(a[1]) || a[1] == '.'))) {
      std::cerr << "solid_field_dump: unknown option " << a << "\n" << USAGE;
      return 2;
    }
    else if (compact.empty()) {
      compact = a;
    }
    else {
      std::cerr << "solid_field_dump: more than one compact file given\n";
      return 2;
    }
  }
  if (compact.empty()) {
    const char* path   = std::getenv("DETECTOR_PATH");
    const char* config = std::getenv("DETECTOR_CONFIG");
    if (path == nullptr || config == nullptr) {
      std::cerr << "solid_field_dump: no compact file given and DETECTOR_PATH/DETECTOR_CONFIG are not set\n" << USAGE;
      return 2;
    }
    compact = std::string(path) + "/" + config + ".xml";
  }
  // plane: axis = value
  const std::size_t eq = plane.find('=');
  double            offset = 0;
  if (eq != 1 || std::string("xyz").find(plane[0]) == std::string::npos || !parseLength(plane.substr(2), offset)) {
    std::cerr << "solid_field_dump: --plane must look like x=0, y=-10cm or z=2.5m\n";
    return 2;
  }
  const int axis = static_cast<int>(std::string("xyz").find(plane[0]));
  // in-plane axes: x -> (z, y), y -> (z, x), z -> (x, y)
  const int iu = axis == 2 ? 0 : 2;
  const int iv = axis == 0 ? 1 : (axis == 1 ? 0 : 1);
  if (!(step > 0)) {
    std::cerr << "solid_field_dump: --step must be positive\n";
    return 2;
  }
  if ((haveU && urange[1] <= urange[0]) || (haveV && vrange[1] <= vrange[0])) {
    std::cerr << "solid_field_dump: a range needs LO < HI\n";
    return 2;
  }
  const std::string config = std::filesystem::path(compact).stem().string();
  if (output.empty()) {
    output = config + ".field.npz";
  }

  try {
    dd4hep::setPrintLevel(verbose ? dd4hep::INFO : dd4hep::WARNING);
    dd4hep::Detector& description = dd4hep::Detector::getInstance();
    description.fromXML(compact);
    if (!haveU || !haveV) {
      double lo[3], hi[3];
      detectorBox(description, lo, hi);
      auto widen = [&](int k, double r[2]) {
        const double m = 1500.0; // mm: room for close-ups around a detector at the edge (--fit-pad)
        r[0]           = lo[k] - m;
        r[1]           = hi[k] + m;
      };
      if (!haveU) {
        widen(iu, urange);
      }
      if (!haveV) {
        widen(iv, vrange);
      }
    }
    // grid on multiples of the step
    auto axisValues = [&](const double r[2]) {
      std::vector<double> v;
      const long          k0 = static_cast<long>(std::floor(r[0] / step + 1e-9));
      const long          k1 = static_cast<long>(std::ceil(r[1] / step - 1e-9));
      for (long k = k0; k <= k1; ++k) {
        v.push_back(k * step);
      }
      return v;
    };
    const std::vector<double> u = axisValues(urange), v = axisValues(vrange);
    if (u.size() < 2 || v.size() < 2 || u.size() * v.size() > 60'000'000UL) {
      std::cerr << "solid_field_dump: " << u.size() << " x " << v.size()
                << " grid points: choose another range or step\n";
      return 2;
    }
    dd4hep::OverlayedField field = description.field();
    std::vector<float>     B(u.size() * v.size() * 3);
    double                 pos[3] = {0, 0, 0}, b[3];
    pos[axis]                     = offset * dd4hep::mm;
    double bmax                   = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
      pos[iu] = u[i] * dd4hep::mm;
      for (std::size_t j = 0; j < v.size(); ++j) {
        pos[iv] = v[j] * dd4hep::mm;
        b[0] = b[1] = b[2] = 0;
        field.magneticField(pos, b);
        float* out = &B[(i * v.size() + j) * 3];
        for (int k = 0; k < 3; ++k) {
          out[k] = static_cast<float>(b[k] / dd4hep::tesla);
        }
        bmax = std::max(bmax, std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) / dd4hep::tesla);
      }
    }
    std::array<double, 3> origin{0, 0, 0}, udir{0, 0, 0}, vdir{0, 0, 0}, normal{0, 0, 0};
    origin[axis] = offset;
    udir[iu]     = 1;
    vdir[iv]     = 1;
    normal[axis] = 1;
    std::ostringstream meta;
    meta << "{\"format\": \"solid_field\", \"version\": 1, \"config\": " << jsonString(config)
         << ", \"compact\": " << jsonString(compact) << ", \"plane\": " << jsonString(plane)
         << ", \"step_mm\": " << step << ", \"unit_B\": \"tesla\", \"unit_length\": \"mm\", \"B_max_T\": " << bmax
         << "}";
    solid::NpzWriter npz;
    npz.add("u", u);
    npz.add("v", v);
    npz.add("B", B, {u.size(), v.size(), 3});
    npz.add("origin", std::vector<double>(origin.begin(), origin.end()));
    npz.add("u_dir", std::vector<double>(udir.begin(), udir.end()));
    npz.add("v_dir", std::vector<double>(vdir.begin(), vdir.end()));
    npz.add("normal", std::vector<double>(normal.begin(), normal.end()));
    npz.addScalar("meta", meta.str());
    npz.write(output);
    std::cout << "solid_field_dump: " << config << ", plane " << plane << ": " << u.size() << " x " << v.size()
              << " points, step " << step << " mm, u " << u.front() << " .. " << u.back() << " mm, v " << v.front()
              << " .. " << v.back() << " mm, |B| up to " << bmax << " T -> " << output << "\n";
  }
  catch (const std::exception& e) {
    std::cerr << "solid_field_dump: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
