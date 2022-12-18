cmake_minimum_required (VERSION 3.8)


if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

add_executable (SOD_Homework02 "main.cpp" "Generator.h" "Generator.cpp" "Solver.h" "Solver.cpp")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # using Clang
  target_compile_options(SOD_Homework02 PRIVATE
      # All warnings, warnings as errors, be pedantic.
      -Wall
      -Wextra
      -Werror
      -Wpedantic
  
      # Disable warnings about C++98 incompatibility. We're using C++17 features...
      -Wno-c++98-compat
      -Wno-c++98-compat-pedantic
  
  	# https://stackoverflow.com/questions/56707123/how-to-suppress-clang-warnings-in-third-party-library-header-file-in-cmakelists
      -Wno-exit-time-destructors 
      -Wno-global-constructors  
  )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  target_compile_options(SOD_Homework02 PRIVATE
      # All warnings, warnings as errors, be pedantic.
      -Wall
      -Wextra
      -Werror
      -Wpedantic
  
      # Disable warnings about C++98 incompatibility. We're using C++17 features...
      -Wno-c++98-compat
      -Wno-c++98-compat-pedantic
  
  	# https://stackoverflow.com/questions/56707123/how-to-suppress-clang-warnings-in-third-party-library-header-file-in-cmakelists
      -Wno-exit-time-destructors 
      -Wno-global-constructors  
  )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # using Intel C++
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # using Visual Studio C++
  target_compile_options(SOD_Homework02 PRIVATE)
endif()

# https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
	target_link_libraries(SOD_Homework02 PUBLIC OpenMP::OpenMP_CXX)
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET SOD_Homework02 PROPERTY CXX_STANDARD 20)
endif()
