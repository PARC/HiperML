project(HiperML)
cmake_minimum_required(VERSION 2.8.8)
set(PROJECT_NAME_STR HiperML)
project(${PROJECT_NAME_STR} C CXX)

set(CMAKE_MODULE_PATH ${HiperML_SOURCE_DIR}/cmake)

find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

if(MSVC)
  #vc 2012 fix for vararg templates
  set(MSVC_COMPILER_DEFS "-D_VARIADIC_MAX=10")
endif()

#-------------------
# set common include folder for module
#-------------------
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)

#-------------------
#macro for adding shared and static libraries
#-------------------
macro(add_hml_lib NAME)
cuda_add_library(${NAME} SHARED ${ARGN})
cuda_add_library(${NAME}_static STATIC ${ARGN})
set(${NAME}_LIB ${PROJECT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
set(${NAME}_STATIC_LIB ${PROJECT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${NAME}_static${CMAKE_STATIC_LIBRARY_SUFFIX})
endmacro(add_hml_lib)

# Add both shared and static libraries for ${NAME}
# For example, if ${NAME} = hml, then:
# 1. the share library target is called 'hml'
# 2. the static library target is called 'hml_static'
# 3. for target_link_libraries(), one can use ${HML_LIB} for
#    linking the shared library and ${HML_STATIC_LIB} for
#    linking the static library
macro(add_shared_and_static_libs NAME)
  cuda_add_library(${NAME} SHARED ${ARGN})
  cuda_add_library(${NAME}_static STATIC ${ARGN})
  #set_property(TARGET ${NAME} PROPERTY C_STANDARD 99)
  #set_property(TARGET ${NAME}_static PROPERTY C_STANDARD 99)
  set_property(TARGET ${NAME}_static PROPERTY POSITION_INDEPENT_CODE TRUE)
  #add_core_deps(${NAME})
  #add_core_deps(${NAME}_static)
  string(TOUPPER ${NAME}_lib SHARED_LIB)
  string(TOUPPER ${NAME}_static_lib STATIC_LIB)
  set(${SHARED_LIB} ${PROJECT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(${STATIC_LIB} ${PROJECT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${NAME}_static${CMAKE_STATIC_LIBRARY_SUFFIX})
endmacro(add_shared_and_static_libs)

#-------------------
# Module source
#-------------------
include_directories(${COMMON_INCLUDES})
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cu)
add_shared_and_static_libs(hml ${SRC_FILES})

macro(add_hml_executable NAME)
  cuda_add_executable(${NAME} ${ARGN})
  set_property(TARGET ${NAME} PROPERTY C_STANDARD 99)
  target_link_libraries(${NAME} ${HML_LIB} ${CMAKE_THREAD_LIBS_INIT})
  #add_core_deps(${NAME})
endmacro(add_hml_executable)

add_subdirectory(${PROJECT_SOURCE_DIR}/apps)


# Set installation rules
install(DIRECTORY include/
  DESTINATION include/HiperML
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".git" EXCLUDE)

install(TARGETS hml hml_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
