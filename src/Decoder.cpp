#include "nvenc_rtsp_common.h"
#include "Decoder.h"

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

using namespace std;
using namespace nvenc_rtsp;

Decoder::Decoder(PurposeID _purpose, NvPipe_Format _decFormat, NvPipe_Codec _codec, RecvCallFn _recv_cb)
    : m_purpose(_purpose),
      m_decFormat(_decFormat),
      m_codec(_codec),
      m_recv_cb(_recv_cb)
{
    //m_decoder = NvPipe_CreateDecoder(m_decFormat, m_codec, m_width, m_height);
    m_decoder = NvPipe_CreateDecoder(m_decFormat, m_codec);
    if (!m_decoder)
        std::cerr << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;    
}

Decoder::Decoder(PurposeID _purpose, NvPipe_Format _decFormat, RecvCallFn _recv_cb)
: Decoder(_purpose, _decFormat, CODEC, _recv_cb)
{    
}

Decoder::~Decoder(){};

void Decoder::cleanUp()
{
    cudaFree(m_gpuDevice);

    NvPipe_Destroy(m_decoder);
}
