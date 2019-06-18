#include "nvenc_rtsp_common.h"
#include "nvenc_rtsp/ServerPipeRTSP.h"

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

std::mutex ServerPipeRTSP::m_coutMutex;

ServerPipeRTSP::ServerPipeRTSP(int _port, int _width, int _height, int _bytesPerPixel, NvPipe_Format _encFormat, NvPipe_Compression _compression, NvPipe_Codec _codec, float _bitrateMbps, int _targetFPS)
    : Encoder(_width, _height, _bytesPerPixel, _encFormat, _compression, _codec, _bitrateMbps, _targetFPS) 
{

    // RTSP ############################################################
    m_rtspThread = thread([&, _width, _height, _bytesPerPixel, _port]() {
        int clients = 0;
        std::string ip = xop::NetInterface::getLocalIPAddress();
        std::string rtspUrl;

        std::shared_ptr<xop::EventLoop> eventLoop(new xop::EventLoop());
        m_server = new xop::RtspServer(eventLoop.get(), ip, _port);

        xop::MediaSession *session = xop::MediaSession::createNew("live");
        rtspUrl = "rtsp://" + ip + ":" + std::to_string(_port) +  "/" + session->getRtspUrlSuffix();

        session->addMediaSource(xop::channel_0, xop::H264Source::createNew());
        session->setMediaDescribeSDPAddon("a=x-dimensions:" + std::to_string(_width) + "," + std::to_string(_height) + "," + std::to_string(_bytesPerPixel) + "\n");

        session->setNotifyCallback([&clients, &rtspUrl](xop::MediaSessionId sessionId, uint32_t numClients) {
            clients = numClients;
            std::lock_guard<std::mutex> lock(ServerPipeRTSP::m_coutMutex);
            std::cout << "[" << rtspUrl << "]" << " Online: " << clients << std::endl;
        });

        m_sessionId = m_server->addMediaSession(session);

        std::cout << "URL: " << rtspUrl << std::endl << std::endl;

        eventLoop->loop();
    });
    // END RTSP ########################################################
    }

ServerPipeRTSP::ServerPipeRTSP(int _port, int _width, int _height, int _bytesPerPixel, NvPipe_Format _encFormat, NvPipe_Compression _compression, NvPipe_Codec _codec)
:ServerPipeRTSP(_port, _width, _height, _bytesPerPixel, _encFormat, _compression, _codec, 32, 90)
{
    
}

ServerPipeRTSP::ServerPipeRTSP(int _port, int _width, int _height, int _bytesPerPixel, NvPipe_Format _encFormat, NvPipe_Compression _compression)
:    ServerPipeRTSP(_port, _width, _height, _bytesPerPixel, _encFormat, _compression, CODEC, 32, 90)
{

}

ByteObject ServerPipeRTSP::send_frame(cv::Mat mat)
{
    m_timer.reset();

    cudaMemcpy(m_gpuDevice, mat.data, m_dataSize, cudaMemcpyHostToDevice);
    uint64_t size = NvPipe_Encode(m_encoder, m_gpuDevice, m_width * m_bytesPerPixel, m_compressedBuffer.data(), m_dataSize, m_width, m_height, m_forceIFrame);
    m_forceIFrame = true;

    double encodeMs = m_timer.getElapsedMilliseconds();

    // RTSP ############################################################
    m_timer.reset();

        int tail = 0;
        while (tail < size)
        {
            int length;
            int head;
            get_NalPackage(m_compressedBuffer.data(), size, &head, &tail, &length);

            xop::AVFrame videoFrame = {0};
            videoFrame.type = 0;
            videoFrame.size = length;
            videoFrame.timestamp = xop::H264Source::getTimeStamp();
            videoFrame.buffer = &m_compressedBuffer.data()[head];
            m_server->pushFrame(m_sessionId, xop::channel_0, videoFrame);
        }

    double socketMs = m_timer.getElapsedMilliseconds();
    // END RTSP ############################################################

#ifdef DISPPIPETIME
    std::cout << size << std::setw(11) << encodeMs << std::setw(11) << socketMs << std::endl;
#endif

    ByteObject result = {m_compressedBuffer.data(), size};
    return result;
}

ssize_t ServerPipeRTSP::startCodePosition(uchar* buffer, int maxSize, size_t offset)
{
	assert(offset < maxSize);

    /* Simplified Boyer-Moore, inspired by gstreamer */
    while (offset < maxSize - 2)
    {
        switch((int)buffer[offset + 2])
        {
            case 1:
            {
                if ((int)buffer[offset] == 0 && (int)buffer[offset + 1] == 0 )
                {
                    return offset;
                }
                offset += 3;
                break;
            }
            case 0:
                offset++;
            break;
            default:
                offset += 3;
        }
    }

    return -1;
}

void ServerPipeRTSP::get_NalPackage(uchar* buffer, int maxSize, int *head, int *tail, int *length)
{
    int start = startCodePosition(buffer, maxSize, *tail);
    int end = startCodePosition(buffer, maxSize, start + 1);

    if(end < 0) end = maxSize;

    int packageLength = end - start;

    *head = start;
    *tail = end;
    *length = packageLength;
}

void ServerPipeRTSP::cleanUp()
{
    delete m_server;
    m_rtspThread.join();

    Encoder::cleanUp();

}
