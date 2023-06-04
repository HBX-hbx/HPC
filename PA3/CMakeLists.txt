cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
include(CTest)
include_directories(./include)

# Set the path to the METIS and GKlib installation directories
set(METIS_DIR "/home/course/hpc/users/2020010944/local")
set(GKLIB_DIR "/home/course/hpc/users/2020010944/local")

# Include the METIS and GKlib headers
include_directories(${METIS_DIR}/include)
include_directories(${GKLIB_DIR}/include)

# Add the METIS and GKlib library
add_library(METIS SHARED IMPORTED)
add_library(GKLIB SHARED IMPORTED)
set_target_properties(METIS PROPERTIES IMPORTED_LOCATION ${METIS_DIR}/lib/libmetis.so)
set_target_properties(GKLIB PROPERTIES IMPORTED_LOCATION ${GKLIB_DIR}/lib/libGKlib.a)

add_library(expspmm STATIC
include/data.h
include/dbg.h
include/valid.h
include/util.h
include/args.hxx
include/spmm_cpu_ref.h
include/spmm_cpu_opt.h
src/data.cu
src/util.cu
src/valid.cu
src/spmm_ref.cu
src/spmm_opt.cu
src/spmm_cusparse.cu
src/spmm_cpu_ref.cpp
src/spmm_cpu_opt.cpp
  )

# Link against the METIS and GKLIB library
target_link_libraries(expspmm METIS GKLIB)

find_package(CUDA REQUIRED)

FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CUDA_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ENDIF()

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to particles
# will also build with -std=c++11
target_compile_features(expspmm PUBLIC cxx_std_11)

include_directories(/usr/local/cuda-11.1/targets/x86_64-linux/include/)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/googletest/)
# include_directories(${CMAKE_CURRENT_LIST_DIR}/third_party/)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( expspmm
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


set(CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp ${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -O3 --use_fast_math -lcurand  ")

# if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
#     message(STATUS "hah")
# endif()

# include(cmake/googletest.cmake)
# fetch_googletest(
#     ${PROJECT_SOURCE_DIR}/cmake
#     ${PROJECT_BINARY_DIR}/googletest
#     )

enable_testing()


add_subdirectory(test)



