#pragma once

#include "nvenc_rtsp_common.h"
#include "Encoder.h"
#include "xop/RtspServer.h"
#include "net/NetInterface.h"
#include <thread>
#include <mutex>

/* **********************************************************************************
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
*************************************************************************************/

namespace nvenc_rtsp
{
  class ServerPipeRTSP : public Encoder
  {
    public:
      ServerPipeRTSP(std::string ipAddress, int port, NvPipe_Format encFormat, NvPipe_Compression compression, NvPipe_Codec codec, float bitrateMbps, int targetFPS);
      ServerPipeRTSP(std::string ipAddress, int port, NvPipe_Format encFormat, NvPipe_Compression compression, NvPipe_Codec codec);
      ServerPipeRTSP(std::string ipAddress, int port, NvPipe_Format encFormat, NvPipe_Compression compression);

      virtual ByteObject send_frame(cv::Mat mat, uint32_t ts = 0) override;

      virtual void cleanUp() override;

    private:
      bool init_Loop(int _width, int _height, int _bytesPerPixel);

      ssize_t startCodePosition(uchar* buffer, int maxSize, size_t offset);
      void get_NalPackage(uchar* buffer, int maxSize, int *head, int *tail, int *length);

      std::string m_ipAddress;
      int m_port = 0;
      xop::MediaSessionId m_sessionId;
      xop::RtspServer* m_server;
      std::thread m_rtspThread;
      static std::mutex m_coutMutex;
  };
}
