# SPDX-License-Identifier: LGPL-3.0-or-later
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# CMP0074: find_package() uses <PackageName>_ROOT variables
cmake_policy(SET CMP0074 NEW)

project(solid
  VERSION 0.3.0
  LANGUAGES CXX
  DESCRIPTION "SoLID detector description and simulation (DD4hep)"
)

# DD4hep (>= 1.26) is built with C++20 in eic-shell; match it.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "install prefix" FORCE)
endif()
include(GNUInstallDirs)

find_package(DD4hep REQUIRED COMPONENTS DDCore DDRec DDG4)
find_package(ROOT REQUIRED COMPONENTS Core Geom GenVector)
find_package(fmt REQUIRED)

# Note: the plugins do not link against ACTS. ACTS only consumes the geometry (via the
# DD4hep_ParametersPlugin extensions set in the compact files), so it is not required here.
# The ACTS material-map tooling in scripts/material_map uses the ACTS python bindings at run time.

#-----------------------------------------------------------------------------------
set(a_lib_name ${PROJECT_NAME}_dd4hep)   # libsolid_dd4hep.so (+ .components for the plugin manager)

dd4hep_configure_output()
dd4hep_set_compiler_flags()

file(GLOB SOLID_PLUGIN_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)
dd4hep_add_plugin(${a_lib_name}
  SOURCES ${SOLID_PLUGIN_SOURCES}
  USES DD4hep::DDCore DD4hep::DDRec ROOT::Core ROOT::Geom ROOT::GenVector fmt::fmt
)
target_include_directories(${a_lib_name} PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(${a_lib_name} PRIVATE -Wall -Wextra -Wno-unused-parameter)

#-----------------------------------------------------------------------------------
# Simulation-side plugins (DDG4, Geant4): libsolid_ddg4.so. They are kept out of the geometry library so that
# loading the detector description (reconstruction, displays) does not pull in Geant4.
#   src/sim/SceneDump*: geometry scene for drawings (bin/solid_scene_dump, SOLID_SCENE in steering/solid_sim.py)
#   src/sim/TrajectoryRecorder.cpp: trajectory points with times for event displays (SOLID_TRAJ)
find_package(Geant4 REQUIRED)
find_package(ZLIB)
set(a_sim_lib_name ${PROJECT_NAME}_ddg4)
dd4hep_add_plugin(${a_sim_lib_name}
  SOURCES ${PROJECT_SOURCE_DIR}/src/sim/SceneDump.cpp ${PROJECT_SOURCE_DIR}/src/sim/SceneDumpPlugins.cpp
          ${PROJECT_SOURCE_DIR}/src/sim/TrajectoryRecorder.cpp
  USES DD4hep::DDCore DD4hep::DDG4 ROOT::Core ROOT::Geom ${Geant4_LIBRARIES}
)
target_include_directories(${a_sim_lib_name} PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/sim)
target_include_directories(${a_sim_lib_name} SYSTEM PRIVATE ${Geant4_INCLUDE_DIRS})
target_compile_definitions(${a_sim_lib_name} PRIVATE SOLID_VERSION="${PROJECT_VERSION}")
target_compile_options(${a_sim_lib_name} PRIVATE -Wall -Wextra -Wno-unused-parameter)
if(ZLIB_FOUND)
  # compressed scene files; without zlib they are written uncompressed
  target_link_libraries(${a_sim_lib_name} PRIVATE ZLIB::ZLIB)
  target_compile_definitions(${a_sim_lib_name} PRIVATE SOLID_HAVE_ZLIB)
endif()

# solid_scene_dump <compact> -o scene.npz: the scene without ddsim (calls the SoLID_SceneDump plugin)
add_executable(solid_scene_dump ${PROJECT_SOURCE_DIR}/src/sim/solid_scene_dump.cpp)
target_link_libraries(solid_scene_dump PRIVATE DD4hep::DDCore)
target_compile_options(solid_scene_dump PRIVATE -Wall -Wextra)
install(TARGETS solid_scene_dump RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# solid_field_dump <compact> --plane x=0 -o field.npz: the total magnetic field on a section plane, for the field
# views of make_solid_views (solid_lineview --field field.npz)
add_executable(solid_field_dump ${PROJECT_SOURCE_DIR}/src/sim/solid_field_dump.cpp)
target_link_libraries(solid_field_dump PRIVATE DD4hep::DDCore ROOT::Core ROOT::Geom)
target_compile_options(solid_field_dump PRIVATE -Wall -Wextra)
if(ZLIB_FOUND)
  target_link_libraries(solid_field_dump PRIVATE ZLIB::ZLIB)
  target_compile_definitions(solid_field_dump PRIVATE SOLID_HAVE_ZLIB)
endif()
install(TARGETS solid_field_dump RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#-----------------------------------------------------------------------------------
# Install the detector description: compact fragments, top-level configurations, field maps
set(SOLID_SHARE_DIR ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME})

install(DIRECTORY compact/
  DESTINATION ${SOLID_SHARE_DIR}/compact
  FILES_MATCHING PATTERN "*.xml" PATTERN "*.csv" PATTERN "*.txt" PATTERN "*.json"
)

file(GLOB SOLID_TOPLEVEL_XML CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/solid_*.xml)
install(FILES ${SOLID_TOPLEVEL_XML} DESTINATION ${SOLID_SHARE_DIR})
# the default configuration (solid.xml) is SIDIS on polarized 3He
install(FILES ${PROJECT_SOURCE_DIR}/solid_SIDIS_He3.xml DESTINATION ${SOLID_SHARE_DIR} RENAME solid.xml)

# field maps are downloaded on demand by the FileLoader plugin into <DETECTOR_PATH>/fieldmaps;
# create the directory so the cache is writable in a user install
install(DIRECTORY DESTINATION ${SOLID_SHARE_DIR}/fieldmaps)

# steering files, macros and helper scripts
install(DIRECTORY steering/ DESTINATION ${SOLID_SHARE_DIR}/steering OPTIONAL)
install(DIRECTORY macro/ DESTINATION ${SOLID_SHARE_DIR}/macro OPTIONAL)
install(DIRECTORY bin/ DESTINATION ${CMAKE_INSTALL_BINDIR} USE_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN "*" PATTERN "*.in" EXCLUDE)
# python packages (solid_lineview, solid_evdisplay): <prefix>/python, put on PYTHONPATH by thissolid.sh
install(DIRECTORY python/ DESTINATION python
  FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)

#-----------------------------------------------------------------------------------
# Environment script: source <prefix>/bin/thissolid.sh [configuration]
configure_file(${PROJECT_SOURCE_DIR}/templates/thissolid.sh.in
               ${PROJECT_BINARY_DIR}/thissolid.sh @ONLY)
install(PROGRAMS ${PROJECT_BINARY_DIR}/thissolid.sh DESTINATION ${CMAKE_INSTALL_BINDIR})

#-----------------------------------------------------------------------------------
# Tests: every configuration must load and be free of overlaps (TGeo check).
include(CTest)
if(BUILD_TESTING)
  file(GLOB SOLID_TEST_CONFIGS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/solid_*.xml)
  foreach(cfg ${SOLID_TEST_CONFIGS})
    get_filename_component(cfg_name ${cfg} NAME_WE)
    if(cfg_name STREQUAL "solid_beamtest")
      continue()
    endif()
    add_test(NAME load_${cfg_name}
      COMMAND ${CMAKE_COMMAND} -E env
              DETECTOR_PATH=${PROJECT_SOURCE_DIR}
              LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
              python3 ${PROJECT_SOURCE_DIR}/scripts/geometry/check_geometry.py ${cfg}
    )
    set_tests_properties(load_${cfg_name} PROPERTIES TIMEOUT 1800 LABELS geometry)
  endforeach()

  # scene producer: a small geometry with a doubly placed, a reflected and a parameterised volume
  add_test(NAME scene_dump
    COMMAND ${CMAKE_COMMAND} -E env
            DETECTOR_PATH=${PROJECT_SOURCE_DIR}
            LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
            PYTHONPATH=${PROJECT_SOURCE_DIR}/python:$ENV{PYTHONPATH}
            PYTHONDONTWRITEBYTECODE=1
            python3 ${PROJECT_SOURCE_DIR}/scripts/geometry/test_scene_dump.py
                    --tool $<TARGET_FILE:solid_scene_dump>
                    --compact ${PROJECT_SOURCE_DIR}/test/test_scene_dump.xml
  )
  set_tests_properties(scene_dump PROPERTIES TIMEOUT 300 LABELS scene)

  # field views: line clean-up and window clipping of solid_lineview, solid_field_dump against DD4hep's field,
  # the drawing from the sampled field against the drawing from the GEMC map
  add_test(NAME field_view
    COMMAND ${CMAKE_COMMAND} -E env
            DETECTOR_PATH=${PROJECT_SOURCE_DIR}
            LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
            PYTHONPATH=${PROJECT_SOURCE_DIR}/python:$ENV{PYTHONPATH}
            PYTHONDONTWRITEBYTECODE=1
            python3 ${PROJECT_SOURCE_DIR}/scripts/geometry/test_field_view.py
                    --tool $<TARGET_FILE:solid_field_dump> --scene-tool $<TARGET_FILE:solid_scene_dump>
                    --config solid_JPsi_lite
  )
  set_tests_properties(field_view PROPERTIES TIMEOUT 1800 LABELS "scene;field")

  # event display chain: 1 gun event with the trajectory recorder, solid_evexport, solid_evview (2 views and a
  # short animation). Uses ddsim, about 30 s.
  add_test(NAME event_display
    COMMAND ${CMAKE_COMMAND} -E env
            DETECTOR_PATH=${PROJECT_SOURCE_DIR}
            LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
            PYTHONPATH=${PROJECT_SOURCE_DIR}/python:$ENV{PYTHONPATH}
            PYTHONDONTWRITEBYTECODE=1
            python3 ${PROJECT_SOURCE_DIR}/scripts/event_display/test_event_display.py
                    --config solid_JPsi_lite --outdir ${PROJECT_BINARY_DIR}/event_display_test
  )
  set_tests_properties(event_display PROPERTIES TIMEOUT 1800 LABELS "scene;events")
endif()
