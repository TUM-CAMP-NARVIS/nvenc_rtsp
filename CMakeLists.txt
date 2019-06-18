cmake_minimum_required(VERSION 3.2)
project(ArtekMedNvPipe VERSION 1.0.0 LANGUAGES CXX C)

# **********************************************************************************
#                                                                                   #
# Copyright (c) 2019,                                                               #
# Research group MITI                                                               #
# Technical University of Munich                                                    #
#                                                                                   #
# All rights reserved.                                                              #
# Kevin Yu - kevin.yu@tum.de                                                        #
#                                                                                   #
# Redistribution and use in source and binary forms, with or without                #
# modification, are restricted to the following conditions:                         #
#                                                                                   #
#  * The software is permitted to be used internally only by the research group     #
#    MITI and CAMPAR and any associated/collaborating groups and/or individuals.    #
#  * The software is provided for your internal use only and you may                #
#    not sell, rent, lease or sublicense the software to any other entity           #
#    without specific prior written permission.                                     #
#    You acknowledge that the software in source form remains a confidential        #
#    trade secret of the research group MITI and therefore you agree not to         #
#    attempt to reverse-engineer, decompile, disassemble, or otherwise develop      #
#    source code for the software or knowingly allow others to do so.               #
#  * Redistributions of source code must retain the above copyright notice,         #
#    this list of conditions and the following disclaimer.                          #
#  * Redistributions in binary form must reproduce the above copyright notice,      #
#    this list of conditions and the following disclaimer in the documentation      #
#    and/or other materials provided with the distribution.                         #
#  * Neither the name of the research group MITI nor the names of its               #
#    contributors may be used to endorse or promote products derived from this      #
#    software without specific prior written permission.                            #
#                                                                                   #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   #
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     #
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE            #
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR   #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES    #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;      #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND       #
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS     #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                      #
#                                                                                   #
# ***********************************************************************************

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fPIC")

option(BUILD_SHARED_LIBS "Build shared library" ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

include(GNUInstallDirs)

#################################################################################################
# OPENCV
#(somehow getting it with conanfile.py it will have problems with some function like videocapture)
#################################################################################################

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.14/conan.cmake"
                 "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(REQUIRES opencv/4.1.0@conan/stable
		nvpipe/[>=0.1]@camposs/testing
		nvidia-video-codec-sdk/[>=9.0]@vendor/stable
                BASIC_SETUP 
                BUILD missing)

#################################################################################################
# CUDA
#################################################################################################

find_package(CUDA REQUIRED)

# Construct path to CUDA driver API lib (not provided by FindCUDA)
get_filename_component(CUDA_LIB_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)
find_library(CUDA_LIB NAMES cuda HINTS ${CUDA_LIB_DIR})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

include_directories(${CUDA_INCLUDE_DIRS})


#################################################################################################
# RTSP Server and Client
#################################################################################################

file(GLOB RTSPSERVERSRC 3rdParty/RTSPServer/source/*/*.cpp)

file(GLOB RTSPCLIENTSRC 3rdParty/RTSPClient/source/*.cpp 3rdParty/RTSPClient/source/*.c)

add_library(rtspConn STATIC ${RTSPSERVERSRC} ${RTSPCLIENTSRC})
SET_TARGET_PROPERTIES(rtspConn PROPERTIES COMPILE_FLAGS "-fPIC")

target_include_directories(rtspConn PUBLIC 3rdParty/RTSPServer/include 3rdParty/RTSPClient/include )


#################################################################################################

file(GLOB_RECURSE CORESOURCES src/*.cpp)

if(BUILD_SHARED_LIBS)
	add_library(nvenc_rtsp SHARED ${CORESOURCES} )
else()
	add_library(nvenc_rtsp STATIC ${CORESOURCES} )
endif()

target_include_directories(nvenc_rtsp PUBLIC include ${CUDA_INCLUDE_DIRS})
target_link_libraries(nvenc_rtsp PUBLIC
	rtspConn
	${OpenCV_LIBRARIES}
    	${CUDA_LIBRARIES}
    	${CUDA_LIB}
	${CONAN_LIBS}
	)
#################################################################################################

message(">> Run \"conan create . @artekmed/testing\"") 



