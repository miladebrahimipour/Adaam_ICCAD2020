# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /snap/cmake/340/bin/cmake

# The command to remove a file.
RM = /snap/cmake/340/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/milad/OpenTimer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/milad/OpenTimer/build

# Include any dependencies generated for this target.
include CMakeFiles/utility.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/utility.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/utility.dir/flags.make

CMakeFiles/utility.dir/unittest/utility.cpp.o: CMakeFiles/utility.dir/flags.make
CMakeFiles/utility.dir/unittest/utility.cpp.o: ../unittest/utility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/milad/OpenTimer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/utility.dir/unittest/utility.cpp.o"
	/usr/bin/g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/utility.dir/unittest/utility.cpp.o -c /home/milad/OpenTimer/unittest/utility.cpp

CMakeFiles/utility.dir/unittest/utility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/utility.dir/unittest/utility.cpp.i"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/milad/OpenTimer/unittest/utility.cpp > CMakeFiles/utility.dir/unittest/utility.cpp.i

CMakeFiles/utility.dir/unittest/utility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/utility.dir/unittest/utility.cpp.s"
	/usr/bin/g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/milad/OpenTimer/unittest/utility.cpp -o CMakeFiles/utility.dir/unittest/utility.cpp.s

# Object files for target utility
utility_OBJECTS = \
"CMakeFiles/utility.dir/unittest/utility.cpp.o"

# External object files for target utility
utility_EXTERNAL_OBJECTS =

../unittest/utility: CMakeFiles/utility.dir/unittest/utility.cpp.o
../unittest/utility: CMakeFiles/utility.dir/build.make
../unittest/utility: ../lib/libOpenTimer.a
../unittest/utility: CMakeFiles/utility.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/milad/OpenTimer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../unittest/utility"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/utility.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/utility.dir/build: ../unittest/utility

.PHONY : CMakeFiles/utility.dir/build

CMakeFiles/utility.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/utility.dir/cmake_clean.cmake
.PHONY : CMakeFiles/utility.dir/clean

CMakeFiles/utility.dir/depend:
	cd /home/milad/OpenTimer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/milad/OpenTimer /home/milad/OpenTimer /home/milad/OpenTimer/build /home/milad/OpenTimer/build /home/milad/OpenTimer/build/CMakeFiles/utility.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/utility.dir/depend
