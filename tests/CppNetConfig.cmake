
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was CppNetConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

# Find dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Check if CUDA was used when building
if(TRUE)
    find_package(CUDAToolkit REQUIRED)
endif()

# Check if OpenMP was used when building
find_package(OpenMP)

# Include the targets file
include("${CMAKE_CURRENT_LIST_DIR}/CppNetTargets.cmake")

# Set variables for users
set(CppNet_LIBRARIES CppNet::CppNet)
set(CppNet_INCLUDE_DIRS "")

check_required_components(CppNet)
