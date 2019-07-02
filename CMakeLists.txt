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

# XXX these flags only work on linux !!!
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -fPIC")

option(BUILD_SHARED_LIBS "Build shared library" ON)

include(GNUInstallDirs)

#################################################################################################
# Opencv
#################################################################################################

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

#################################################################################################
# CUDA & NVPipe Build Settings
#################################################################################################

find_package(CUDA REQUIRED)

# Construct path to CUDA driver API lib (not provided by FindCUDA)
get_filename_component(CUDA_LIB_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)
find_library(CUDA_LIB NAMES cuda HINTS ${CUDA_LIB_DIR})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Options
option(NVPIPE_WITH_ENCODER "Enables the NvPipe encoding interface." ON)
option(NVPIPE_WITH_DECODER "Enables the NvPipe decoding interface." ON)
option(NVPIPE_WITH_OPENGL "Enables the NvPipe OpenGL interface." OFF)
option(NVPIPE_BUILD_EXAMPLES "Builds the NvPipe example applications (requires both encoder and decoder)." OFF)

# Header
configure_file(3rdParty/NvPipe/NvPipe.h.in include/NvPipe/NvPipe.h @ONLY)
include_directories(${CUDA_INCLUDE_DIRS})

# NvPipe shared library
list(APPEND NVPIPE_SOURCES
    3rdParty/NvPipe/NvPipe.cu
    3rdParty/NvPipe/Video_Codec_SDK_9.0.20/Samples/Utils/ColorSpace.cu
    )
list(APPEND NVPIPE_LIBRARIES
    ${CMAKE_DL_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_LIB}
    )

if (NVPIPE_WITH_ENCODER)
    list(APPEND NVPIPE_SOURCES
        3rdParty/NvPipe/Video_Codec_SDK_9.0.20/Samples/NvCodec/NvEncoder/NvEncoder.cpp
        3rdParty/NvPipe/Video_Codec_SDK_9.0.20/Samples/NvCodec/NvEncoder/NvEncoderCuda.cpp
        )
endif()

if (NVPIPE_WITH_DECODER)    
    list(APPEND NVPIPE_SOURCES
        3rdParty/NvPipe/Video_Codec_SDK_9.0.20/Samples/NvCodec/NvDecoder/NvDecoder.cpp
        )
    list(APPEND NVPIPE_LIBRARIES
        nvcuvid
        )

    if (WIN32)
        if (CMAKE_SIZEOF_VOID_P EQUAL 8)
            link_directories(3rdParty/NvPipe/Video_Codec_SDK_9.0.20/Lib/x64)
        elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
            link_directories(3rdParty/NvPipe/Video_Codec_SDK_9.0.20/Lib/Win32)
        endif()    
    endif()
endif()

include(GNUInstallDirs)

cuda_add_library(CuNvPipe STATIC ${NVPIPE_SOURCES})
target_include_directories(CuNvPipe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
    )
target_include_directories(CuNvPipe PRIVATE
    $<BUILD_INTERFACE:3rdParty/NvPipe/Video_Codec_SDK_9.0.20/Samples 3rdParty/NvPipe/Video_Codec_SDK_9.0.20/Samples/NvCodec 3rdParty/NvPipe/Video_Codec_SDK_9.0.20/include ${CUDA_INCLUDE_DIRS}>
    )
target_link_libraries(CuNvPipe ${NVPIPE_LIBRARIES} )

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
    	CuNvPipe
	${OpenCV_LIBRARIES}
	)
#################################################################################################




