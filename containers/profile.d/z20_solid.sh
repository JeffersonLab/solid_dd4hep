# SoLID defaults inside the solid_xl image (runs before the image's z21_epic_main.sh, which only
# sets up ePIC when DETECTOR_PATH/DETECTOR_CONFIG are still unset).
export SOLID_PREFIX=/opt/solid
# Also when DETECTOR_PATH already points into /opt/solid: a nested login shell (the image's entry point runs one,
# and so does `bash -l`) starts from the inherited DETECTOR_PATH, but z10_spack_environment.sh has just reset
# LD_LIBRARY_PATH, which must get the SoLID plugin library again.
if [ -f /opt/solid/bin/thissolid.sh ]; then
  case "${DETECTOR_PATH:-}" in
    ""|/opt/solid/*) . /opt/solid/bin/thissolid.sh "${DETECTOR_CONFIG:-${SOLID_DEFAULT_CONFIG:-solid}}" ;;
  esac
fi
# lAger rebuilt with the ROOT patch (containers/Dockerfile), ahead of the one in /opt/local/bin that crashes
if [ -x /opt/lager/bin/lager ]; then
  case ":${PATH}:" in
    *":/opt/lager/bin:"*) ;;
    *) export PATH="/opt/lager/bin:${PATH}" ;;
  esac
fi
if [ -d /opt/k4 ]; then
  export CMAKE_PREFIX_PATH="/opt/k4${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
  export LD_LIBRARY_PATH="/opt/k4/lib:${LD_LIBRARY_PATH}" PYTHONPATH="/opt/k4/python:${PYTHONPATH}" PATH="/opt/k4/bin:${PATH}"
fi
