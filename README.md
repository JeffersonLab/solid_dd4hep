SoLID Detector Simulation
=========================
Detector simulation code for SoLID experiment

- [Wiki](https://solid.jlab.org/wiki/index.php/Meeting_solid_software#Using_EIC_software_for_SoLID_on_ifarm)

### Compilation

To configure, build, and install (to the `install` directory), use the following commands:
```bash
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=$EIC_SHELL_PREFIX
cmake --build build
cmake --install build
```
