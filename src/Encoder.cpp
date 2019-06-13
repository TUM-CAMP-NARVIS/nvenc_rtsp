#include "nvenc_rtsp_common.h"
#include "Encoder.h"

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

using namespace nvenc_rtsp;

Encoder::Encoder(int _width, int _height, int _bytesPerPixel, PurposeID _purpose, NvPipe_Format _encFormat, NvPipe_Compression _compression, NvPipe_Codec _codec, float _bitrateMbps, int _targetFPS)
    : m_width(_width),
      m_height(_height),
      m_bytesPerPixel(_bytesPerPixel),
      m_purpose(_purpose),
      m_bitrateMbps(_bitrateMbps),
      m_targetFPS(_targetFPS),
      m_encFormat(_encFormat),
      m_compression(_compression),
      m_codec(_codec)
{
    m_dataSize = m_width * m_height * m_bytesPerPixel;
    m_compressedBuffer = std::vector<uint8_t>(m_dataSize);

    //m_encoder = NvPipe_CreateEncoder(m_encFormat, m_codec, m_compression, m_bitrateMbps * 1000 * 1000, m_targetFPS, m_width, m_height);
    m_encoder = NvPipe_CreateEncoder(m_encFormat, m_codec, m_compression, m_bitrateMbps * 1000 * 1000, m_targetFPS);
    if (!m_encoder)
        std::cerr << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;

    cudaMalloc(&m_gpuDevice, m_dataSize);
}

Encoder::Encoder(int _width, int _height, int _bytesPerPixel, PurposeID _purpose, NvPipe_Format _encFormat, NvPipe_Compression _compression, NvPipe_Codec _codec)
:Encoder(_width, _height, _bytesPerPixel, _purpose, _encFormat, _compression, _codec, 32, 90)
{
    
}

Encoder::Encoder(int _width, int _height, int _bytesPerPixel, PurposeID _purpose, NvPipe_Format _encFormat, NvPipe_Compression _compression)
:    Encoder(_width, _height, _bytesPerPixel, _purpose, _encFormat, _compression, CODEC, 32, 90)
{

}

Encoder::~Encoder(){};

void Encoder::cleanUp()
{
    cudaFree(m_gpuDevice);

    NvPipe_Destroy(m_encoder);
}
