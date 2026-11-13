# SPDX-License-Identifier: LGPL-3.0-or-later
# solid_xl: the EIC software image (eic_xl) + ffmpeg, gmsh, a working lAger + the SoLID geometry (solid_dd4hep),
# optional Gaudi/k4FWCore.
#
#   docker build -f containers/Dockerfile -t solid_xl:dev .                       # from the repository root
#   docker build -f containers/Dockerfile --build-arg EIC_TAG=26.11.0-stable -t solid_xl:dev .
#   docker build -f containers/Dockerfile --build-arg WITH_GAUDI=1 -t solid_xl:dev-gaudi .
#   apptainer build solid_xl.sif docker-daemon://solid_xl:dev                        # for ifarm / laptops
#
# Published by .github/workflows/image.yml as ghcr.io/<repository owner>/solid_xl:<tag>; get it with
#   containers/install.sh --solid-xl
#
# solid_recon (EICrecon plugins) will be added here once it is re-based out of tree (docs/reconstruction.md).
ARG EIC_TAG=26.10.0-stable
FROM eicweb/eic_xl:${EIC_TAG}
ARG WITH_GAUDI=0
# Login shells: the eic_xl environment (python, cmake, DD4hep, Geant4 under /opt/local) is set in /etc/profile.d,
# not in the image's ENV, where PATH holds only the system directories.
SHELL ["/bin/bash", "-l", "-c"]

# (optional) Gaudi + k4FWCore in a separate spack environment/view, reusing the image's packages
# (verified to concretize to gaudi@39.4 + k4fwcore@1.5 with 12 new packages on eic_xl 26.10).
COPY containers/spack-gaudi.yaml /opt/spack-environment/solid-gaudi/spack.yaml
RUN if [ "${WITH_GAUDI}" = "1" ]; then \
      source /opt/spack/share/spack/setup-env.sh && \
      spack -e /opt/spack-environment/solid-gaudi install --fail-fast && \
      spack -e /opt/spack-environment/solid-gaudi view --dependencies no symlink /opt/k4 gaudi k4fwcore && \
      spack clean -a ; \
    fi

# Tools eic_xl does not ship:
#   ffmpeg                       MP4 output of the event-display animations
#   gmsh (+ two X11 libraries)   mesher of the field solver, magnet_field/solve_rz.py
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libxcursor1 libxinerama1 && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir --no-deps "gmsh>=4.8" && \
    ffmpeg -version | head -1 && \
    command -v python3 cmake ddsim && \
    python3 -c "import gmsh; gmsh.initialize(); print('gmsh', gmsh.__version__); gmsh.finalize()"

# lAger 3.7.0 (Argonne l/A event generator: exclusive J/psi and other vector mesons, used for the SoLID J/psi
# studies). eic_xl ships it, but that build ends in a segmentation fault before main() with ROOT 6.38 or newer
# (containers/patches/lager-3.7.0-root-6.40.patch has the reason). Build the same release with that patch and
# the one of the eic-spack recipe into /opt/lager: with its examples, and with an RPATH (not RUNPATH) so that
# it loads its own libraries although LD_LIBRARY_PATH holds /opt/local/lib with the ones that crash.
# profile.d/z20_solid.sh puts /opt/lager/bin first on PATH.
ARG LAGER_VERSION=3.7.0
ARG LAGER_SHA256=3ba025b744caae17e7109d8f8a19edd3e2ffdd79d07ff692373f15e92310f2f5
COPY containers/patches/lager-3.7.0-root-6.40.patch /tmp/lager/lager.patch
RUN cd /tmp/lager && \
    spack_pkg=/opt/spack-packages/repos/eic-spack/spack_repo/eic/packages/lager && \
    curl -fsSL -o lager.tar.gz \
         https://eicweb.phy.anl.gov/monte_carlo/lager/-/archive/${LAGER_VERSION}/lager-${LAGER_VERSION}.tar.gz && \
    echo "${LAGER_SHA256}  lager.tar.gz" | sha256sum -c - && \
    mkdir src && tar -xzf lager.tar.gz -C src --strip-components=1 --no-same-owner && \
    patch -d src -p1 < "${spack_pkg}/fix-photospp-includes.patch" && \
    patch -d src -p1 < lager.patch && \
    cmake -S src -B build -DCMAKE_INSTALL_PREFIX=/opt/lager -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_MODULE_PATH="${spack_pkg}" -DHEPMC_DIR=/opt/local -DPHOTOSPP_DIR=/opt/local \
          -DCMAKE_INSTALL_RPATH=/opt/lager/lib \
          -DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags -DCMAKE_SHARED_LINKER_FLAGS=-Wl,--disable-new-dtags && \
    cmake --build build -j"$(nproc)" && cmake --install build && \
    mkdir -p /opt/lager/share/lager && cp -r src/examples src/COPYING src/README.md lager.patch /opt/lager/share/lager/ && \
    cd / && rm -rf /tmp/lager && \
    /opt/lager/bin/lager --help > /dev/null && echo "lager: starts"

# SoLID geometry, built with the image's own toolchain
COPY . /opt/src/solid_dd4hep
RUN cmake -S /opt/src/solid_dd4hep -B /tmp/build -DCMAKE_INSTALL_PREFIX=/opt/solid \
          -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    cmake --build /tmp/build -j"$(nproc)" && cmake --install /tmp/build && rm -rf /tmp/build

COPY containers/profile.d/z20_solid.sh /etc/profile.d/z20_solid.sh
ARG IMAGE_SOURCE=https://github.com/JeffersonLab/solid_dd4hep
ARG EIC_TAG
LABEL org.opencontainers.image.source="${IMAGE_SOURCE}" \
      org.opencontainers.image.title="SoLID runtime image (eic_xl + solid_dd4hep)" \
      org.opencontainers.image.base.name="docker.io/eicweb/eic_xl:${EIC_TAG}"
