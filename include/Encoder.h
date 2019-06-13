#pragma once

#include "nvenc_rtsp_common.h"

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
  struct ByteObject
  {
    uchar *data;
    uint64_t size;
  };

  class Encoder
  {
  public:
    Encoder(int width, int height, int bytesPerPixel, PurposeID purpose, NvPipe_Format encFormat, NvPipe_Compression compression, NvPipe_Codec codec, float bitrateMbps, int targetFPS);
    Encoder(int width, int height, int bytesPerPixel, PurposeID purpose, NvPipe_Format encFormat, NvPipe_Compression compression, NvPipe_Codec codec);
    Encoder(int width, int height, int bytesPerPixel, PurposeID purpose, NvPipe_Format encFormat, NvPipe_Compression compression);
	
    virtual ~Encoder();

    virtual ByteObject send_frame(cv::Mat mat){};

    virtual void cleanUp();

  protected:

    int m_width;
    int m_height;
    int m_bytesPerPixel;
    int m_dataSize;
    PurposeID m_purpose;
    float m_bitrateMbps;
    int m_targetFPS;

    bool m_forceIFrame{true};

    NvPipe *m_encoder;
    NvPipe_Codec m_codec;
    NvPipe_Format m_encFormat;
    NvPipe_Compression m_compression;

    std::vector<uint8_t> m_compressedBuffer;
    void *m_gpuDevice;

    Timer m_timer;
  };

} // namespace AM
