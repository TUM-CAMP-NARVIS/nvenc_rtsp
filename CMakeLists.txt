cmake_minimum_required(VERSION 3.2)
project(nvenc_rtsp VERSION 1.0.0 LANGUAGES CXX C)

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
# NVPipe
#################################################################################################

# Options
option(NVPIPE_WITH_ENCODER "Enables the NvPipe encoding interface." ON)
option(NVPIPE_WITH_DECODER "Enables the NvPipe decoding interface." ON)
option(NVPIPE_WITH_OPENGL "Enables the NvPipe OpenGL interface." OFF)
option(NVPIPE_BUILD_EXAMPLES "Builds the NvPipe example applications (requires both encoder and decoder)." OFF)

add_subdirectory(3rdParty/NvPipe)
if (WIN32)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        file(COPY 3rdParty/NvPipe/src/Video_Codec_SDK_9.0.20/Lib/x64/nvcuvid.lib DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
        file(COPY 3rdParty/NvPipe/src/Video_Codec_SDK_9.0.20/Lib/Win32/nvcuvid.lib DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    endif()    
endif()

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
	set_target_properties(nvenc_rtsp PROPERTIES COMPILE_DEFINITIONS "NVENCRTSP_DLL")
else()
	add_library(nvenc_rtsp STATIC ${CORESOURCES} )
endif()

target_include_directories(nvenc_rtsp PUBLIC include ${CUDA_INCLUDE_DIRS})
target_link_libraries(nvenc_rtsp PUBLIC
	rtspConn
    NvPipe
	${OpenCV_LIBRARIES}
	)

