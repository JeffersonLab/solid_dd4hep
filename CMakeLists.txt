cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# CMP0074: find_package() uses <PackageName>_ROOT variables
cmake_policy(SET CMP0074 NEW)

PROJECT(solid
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "SoLID in ATHENA framework"
  )

set(CMAKE_CXX_STANDARD 17)
find_package( DD4hep REQUIRED COMPONENTS DDCore DDG4)

#find_package(Acts REQUIRED COMPONENTS Core PluginIdentification PluginTGeo PluginDD4hep )
find_package(Acts REQUIRED COMPONENTS Core PluginTGeo PluginDD4hep )

find_package(fmt)
#find_library(FMT_LIBRARY fmt)


#-----------------------------------------------------------------------------------
set(a_lib_name solid_dd4hep)

dd4hep_configure_output()
dd4hep_set_compiler_flags()

dd4hep_add_plugin(${a_lib_name} SOURCES 
  src/*.cpp
  USES ROOT::Core ROOT::Gdml
  )
target_link_libraries(${a_lib_name}
  PUBLIC DD4hep::DDCore  DD4hep::DDRec fmt::fmt
  )

#-----------------------------------------------------------------------------------
# Install the detector description files.
#install(DIRECTORY compact/
#    DESTINATION share/${PROJECT_NAME}/${a_lib_name}
#    FILES_MATCHING PATTERN "*.xml"
#    )
install(DIRECTORY compact/
    DESTINATION share/${PROJECT_NAME}/compact
    FILES_MATCHING PATTERN "*.xml"
    )
install(FILES solid.xml
    DESTINATION share/${PROJECT_NAME}/
    )
#-----------------------------------------------------------------------------------
# Install the detector calibration files.
# install(DIRECTORY calibrations/
#     DESTINATION share/${PROJECT_NAME}/calibrations
#     )
