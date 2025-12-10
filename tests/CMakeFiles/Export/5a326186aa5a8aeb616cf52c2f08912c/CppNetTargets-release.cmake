#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CppNet::CppNet" for configuration "Release"
set_property(TARGET CppNet::CppNet APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(CppNet::CppNet PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libCppNet.a"
  )

list(APPEND _cmake_import_check_targets CppNet::CppNet )
list(APPEND _cmake_import_check_files_for_CppNet::CppNet "${_IMPORT_PREFIX}/lib/libCppNet.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
