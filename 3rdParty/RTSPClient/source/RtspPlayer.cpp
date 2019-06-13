//
//  RtspPlayer.cpp
//  toolForTest
//
//  Created by cx on 2018/9/6.
//  Copyright © 2018年 cx. All rights reserved.
//

#include "RtspPlayer.h"
#include <unistd.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define log(tag,fmt,...)\
do {\
    printf("[%s] > ", tag.c_str());\
    printf(fmt, ##__VA_ARGS__);\
    printf("\n");\
} while(0)

#define SetNextState(x) _PlayState = x;

namespace RK {

    RtspPlayer::RtspPlayer(RecvBufferFn recv_cb, std::string name)
    :   recv_cb(recv_cb)
    {
        TAG = "RTSP_" + name;

        _Terminated = false;
        _NetWorked = false;
        _PlayState = RtspIdle;
        _video_rtp_port = RTP_PORT;
        while(!PortIsOpen(_video_rtp_port)) _video_rtp_port ++;

        _video_rtcp_port = RTCP_PORT;
        while(!PortIsOpen(_video_rtcp_port)) _video_rtcp_port ++;

    }
    
    RtspPlayer::~RtspPlayer() {

    }
    
    bool RtspPlayer::getIPFromUrl(std::string url, char *ip, unsigned short *port) {
        unsigned int dstip[4] = {0};
        int dstport = 0;
        int field = sscanf(url.c_str(), "rtsp://%d.%d.%d.%d:%d", &dstip[0], &dstip[1], &dstip[2], &dstip[3], &dstport);
        if (field < 4) {
            log(TAG, "failed to get ip from url");
            return false;
        } else if (field == 4) {
            sprintf(ip, "%d.%d.%d.%d", dstip[0], dstip[1], dstip[2], dstip[3]);
            *port = dstport = 554;
        } else if (field == 5) {
            sprintf(ip, "%d.%d.%d.%d", dstip[0], dstip[1], dstip[2], dstip[3]);
            *port = dstport;
        } else {
            log(TAG, "failed to get ip from url");
            return false;
        }
        
        return true;
    }
    
    bool RtspPlayer::NetworkInit(const char *ip, const short port) {
        _RtspSocket = ::socket(AF_INET, SOCK_STREAM, 0);
        if (_RtspSocket < 0) {
            log(TAG, "network init failed");
            return false;
        }
        _Eventfd = std::max(_RtspSocket, _Eventfd);
        
        int ul = true;
        if (::ioctl(_RtspSocket, FIONBIO, &ul) < 0) {
            log(TAG, "set socket non block failed");
            return false;
        }
        
        struct sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        serverAddr.sin_addr.s_addr = inet_addr(ip);
        
        if (::connect(_RtspSocket, (struct sockaddr *)&serverAddr, (socklen_t)sizeof(serverAddr)) == 0) {
            log(TAG, "sync connect success");
        } else if (errno == EINPROGRESS){
            log(TAG, "async connecting...");
        } else {
            log(TAG, "invalid connect");
            return false;
        }
        
        _NetWorked = true;
        return true;
    }
    
    bool RtspPlayer::RTPSocketInit(int videoPort, int audioPort) {
        if (videoPort) {
            _RtpVideoSocket = ::socket(AF_INET, SOCK_DGRAM, 0);
            if (_RtpVideoSocket < 0) {
                log(TAG, "rtp video socket init failed");
                return false;
            }
            _Eventfd = std::max(_RtpVideoSocket, _Eventfd);
            
            int ul = true;
            if (::ioctl(_RtpVideoSocket, FIONBIO, &ul) < 0) {
                log(TAG, "failed to set rtp video socket non block");
                ::close(_RtpVideoSocket);
                return false;
            }
            
            _RtpVideoAddr.sin_family = AF_INET;
            _RtpVideoAddr.sin_addr.s_addr = INADDR_ANY;
            _RtpVideoAddr.sin_port = htons(videoPort);
            
            if (::bind(_RtpVideoSocket, (const struct sockaddr *)&_RtpVideoAddr, (socklen_t)sizeof(_RtpVideoAddr)) < 0) {
                log(TAG, "failed to bind rtp video socket error %d %s", errno, strerror(errno));
                return false;
            }

            return true;
        }

        return false;
    }
    
    void RtspPlayer::EventInit() {
        FD_ZERO(&_readfd);
        FD_ZERO(&_writefd);
        FD_ZERO(&_errorfd);
        
        FD_SET(_RtspSocket, &_readfd);
        FD_SET(_RtspSocket, &_writefd);
        FD_SET(_RtspSocket, &_errorfd);
    }
    
    std::vector<std::string> RtspPlayer::GetSDPFromMessage(const char *buffer, size_t length, const char *pattern) {
        char *tempBuffer = (char *)malloc(length + 1);
        strcpy(tempBuffer, buffer);
        
        std::vector<std::string> rvector;
        char* tmpStr = strtok(tempBuffer, pattern);
        while (tmpStr != NULL)
        {
            rvector.push_back(std::string(tmpStr));
            tmpStr = strtok(NULL, pattern);
        }
        
        free(tempBuffer);
        
        return rvector;
    }
    
    void RtspPlayer::SendDescribe(std::string url) {
        char buf[1024];
        sprintf(buf, "DESCRIBE %s RTSP/1.0\r\n"
                "Accept: application/sdp\r\n"
                "CSeq: %d\r\n"
                "User-Agent: Lavf58.12.100\r\n"
                "\r\n", url.c_str(), RTSPDESCRIBE);
        
        ::send(_RtspSocket, buf, strlen(buf), 0);
    }
    
    void RtspPlayer::HandleDescribe(const char *buf, ssize_t bufsize) {

        std::vector<std::string> rvector = GetSDPFromMessage(buf, bufsize, "\r\n");
        std::string sdp;
        for (auto substr : rvector) {
            if (strstr(substr.c_str(),"Session:")) {
                ::sscanf(substr.c_str(), "Session:%ld", &_RtspSessionID);
            } else if (strchr(substr.c_str(), '=')) {
                sdp.append(substr);
                sdp.append("\n");
            }
        }

        _SdpParser = sdp_parse(sdp.c_str());
    }
    
    void RtspPlayer::RtspSetup(const std::string url, int track, int CSeq, char *proto, short rtp_port, short rtcp_port) {
        char buf[1024];
        sprintf(buf, "SETUP %s/trackID=%d RTSP/1.0\r\n"
                "CSeq: %d\r\n"
                "User-Agent: Lavf58.12.100\r\n"
                "Transport: %s;unicast;client_port=%d-%d\r\n"
                "\r\n", url.c_str(), track, CSeq, proto, rtp_port, rtcp_port);
        
        ::send(_RtspSocket, buf, strlen(buf), 0);
    }
    
    void RtspPlayer::SendVideoSetup() {
        int i = 0, j = 0;
        int videoTrackID = 0;
        for (i = 0; i < _SdpParser->medias_count; i++) {        
            if (strcmp(_SdpParser->medias[i].info.type, "video") == 0) {
                for (j = 0; j < _SdpParser->medias[i].attributes_count; j++) {
                    if (strstr(_SdpParser->medias[i].attributes[j], "trackID")) {
                        ::sscanf(_SdpParser->medias[i].attributes[j], "control:trackID=%d", &videoTrackID);
                    }
                }
                RtspSetup(_rtspurl, videoTrackID, RTSPVIDEO_SETUP, _SdpParser->medias[i].info.proto, _video_rtp_port, _video_rtcp_port);
            }
        }
    }
    
    bool RtspPlayer::HandleVideoSetup(const char *buf, ssize_t bufsize) {
        int remote_port = 0;
        int remote_rtcp_port = 0;

        if (strstr(buf, "server_port="))
        {
            ::sscanf(strstr(buf, "server_port="), "server_port=%d-%d", &remote_port, &remote_rtcp_port);
        }

        if (!RTPSocketInit(_video_rtp_port, 0)) {
            log(TAG, "rtp socket init failed");
            return false;
        }
        
        struct sockaddr_in remoteAddr;
        remoteAddr.sin_family = AF_INET;
        remoteAddr.sin_port = htons(remote_port);
        remoteAddr.sin_addr.s_addr = inet_addr(_rtspip);
        
        const unsigned char natpacket[] = {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
        ::sendto(_RtpVideoSocket, natpacket, sizeof(natpacket), 0, (const struct sockaddr *)&remoteAddr, (socklen_t)sizeof(remoteAddr));
        
        return true;
    }
    
    void RtspPlayer::SendPlay(const std::string url) {
        char buf[1024];
        sprintf(buf, "PLAY %s RTSP/1.0\r\n"
                "CSeq: %u\r\n"
                "Session: %ld\r\n"
                "Range: npt=0.000-\r\n" // Range
                "User-Agent: Lavf58.12.100\r\n"
                "\r\n", url.c_str(), RTSPPLAY, _RtspSessionID);
        
        ::send(_RtspSocket, buf, strlen(buf), 0);
    }
    
    bool RtspPlayer::HandleRtspMsg(const char *buf, ssize_t bufsize) {
        int MsgType = 0;
        if (::sscanf(buf, "%*[^C]CSeq:%d", &MsgType) != 1) {
            log(TAG, "invalid rtsp message");
            return false;
        }
        
        switch (MsgType) {
            case RTSPOPTIONS:
                
                break;
            case RTSPDESCRIBE:
                HandleDescribe(buf, bufsize);
                SetNextState(RtspSendVideoSetup);
                break;
            case RTSPVIDEO_SETUP:
                if (HandleVideoSetup(buf, bufsize)) {
                    SetNextState(RtspSendPlay);
                }
                break;
            
            case RTSPAUDIO_SETUP:
                break;
            
            case RTSPPLAY:
                break;
                
            default:
                log(TAG, "unknow rtsp message");
                break;
        }
        
        return true;
    }

    void RtspPlayer::HandleRtspState() {
        switch (_PlayState.load()) {
            case RtspSendOptions:
                log(TAG, "rtsp send options");
                break;
            case RtspHandleOptions:
                log(TAG, "rtsp handle options");
                break;
            case RtspSendDescribe:
                log(TAG, "rtsp send describe");
                SendDescribe(_rtspurl);
                break;
            case RtspHandleDescribe:
                log(TAG, "rtsp handle describe");
                break;
            case RtspSendVideoSetup:
                log(TAG, "rtsp send video setup");
                SendVideoSetup();
                break;
            case RtspHandleVideoSetup:
                log(TAG, "rtsp handle video setup");
                break;
            case RtspSendAudioSetup:
                log(TAG, "rtsp send audio setup");
                break;
            case RtspHandleAudioSetup:
                log(TAG, "rtsp handle audio setup");
                break;
            case RtspSendPlay:
                log(TAG, "rtsp send play");
                SendPlay(_rtspurl);
                break;
            case RtspHandlePlay:
                log(TAG, "rtsp handle play");
                break;
            case RtspSendPause:
                log(TAG, "rtsp send pause");
                break;
            case RtspHandlePause:
                log(TAG, "rtsp handle pause");
                break;
            case RtspIdle:
                break;
            case RtspTurnOff:
                _Terminated = true;
                break;
            default:
                log(TAG, "unkonw rtsp state");
                break;
        }
        
        SetNextState(RtspIdle);
    }
    
    // void RtspPlayer::HandleRtpMsg(const char *buf, ssize_t bufsize) {
    //     char header[] = {0, 0, 0, 1};
    //     struct Nalu nalu = *(struct Nalu *)(buf + RTP_OFFSET);
        
    //     if (!fp) {
    //         fp = ::fopen("test.h264", "w+");
    //         if (!fp) {
    //             log(TAG, "failed to oepen test.h264");
    //             return;
    //         }
    //     }
        
    //     if (nalu.type >= 0 && nalu.type < 24) { //one nalu
    //         ::fwrite(header, 4, 1, fp);
    //         ::fwrite(buf + RTP_OFFSET, bufsize - RTP_OFFSET, 1, fp);
    //         ::fflush(fp);
    //     } else if (nalu.type == 28) { //fu-a slice
    //         struct FU fu;
    //         char in = buf[RTP_OFFSET + 1];
    //         fu.S = in >> 7;
    //         fu.E = (in >> 6) & 0x01;
    //         fu.R = (in >> 5) & 0x01;
    //         fu.type = in & 0x1f;
            
    //         if (fu.S == 1) {
    //             char naluType = nalu.forbidden_zero_bit << 7 | nalu.nal_ref_idc << 5 | fu.type;
    //             ::fwrite(header, 4, 1, fp);
    //             ::fwrite(&naluType, 1, 1, fp);
    //             ::fwrite(buf + FU_OFFSET, bufsize - FU_OFFSET, 1, fp);
    //             ::fflush(fp);
    //         } else if (fu.E == 1) {
    //             ::fwrite(buf + FU_OFFSET, bufsize - FU_OFFSET, 1, fp);
    //             ::fflush(fp);
    //         } else {
    //             ::fwrite(buf + FU_OFFSET, bufsize - FU_OFFSET, 1, fp);
    //             ::fflush(fp);
    //         }
    //     }
    // }
    
    bool RtspPlayer::Play(std::string url) {
        char ip[256];
        unsigned short port = 0;
        _rtspurl = url;

        if (!getIPFromUrl(url, ip, &port)) {
            log(TAG, "get ip and port failed");
            return false;
        }
        ::memcpy(_rtspip, ip, sizeof(ip));
        
        if (!NetworkInit(ip, port)) {
            log(TAG, "network uninitizial");
            return false;
        }
        
        EventInit();

        // internal rtsp play thread
        _PlayThreadPtr = std::make_shared<std::thread>([&] {
            uint8_t recvbuf[2048];

            while (!_Terminated)
            {
                FD_ZERO(&_readfd);
                FD_ZERO(&_errorfd);
                FD_SET(_RtspSocket, &_readfd);
                FD_SET(_RtspSocket, &_errorfd);

                // rtp video socket has connected
                if (_RtpVideoSocket)
                {
                    FD_SET(_RtpVideoSocket, &_readfd);
                }

                int r = ::select(_Eventfd + 1, &_readfd, &_writefd, &_errorfd, NULL);

                if (r < 0)
                {
                    log(TAG, "event error...");
                    break;
                }
                else if (r == 0)
                {
                    log(TAG, "event over time...");
                }
                else
                {
                    if (FD_ISSET(_RtspSocket, &_readfd))
                    {
                        ::memset(recvbuf, 0, sizeof(recvbuf));
                        ssize_t recvbytes = ::recv(_RtspSocket, recvbuf, sizeof(recvbuf), 0);
                        if (recvbytes <= 0)
                        {
                            //log(TAG, "Server not found");
                            break;
                        }
                        else
                        {
                            if (!HandleRtspMsg((char *)recvbuf, recvbytes))
                            {
                                log(TAG, "failed to handle rtsp msg");
                            }
                        }
                    }

                    if (FD_ISSET(_RtpVideoSocket, &_readfd))
                    {
                        socklen_t socklen = sizeof(_RtpVideoAddr);
                        ::memset(recvbuf, 0, sizeof(recvbuf));

                        ssize_t recvbytes = ::recvfrom(_RtpVideoSocket, recvbuf, sizeof(recvbuf), 0, (struct sockaddr *)&_RtpVideoAddr, &socklen);
                        //log(TAG, "recv rtp video packet %ld bytes", recvbytes);

                        if (recv_cb != NULL)
                        {
                            recv_cb(recvbuf, recvbytes);
                        }
                    }

                    if (FD_ISSET(_RtspSocket, &_writefd))
                    {
                        log(TAG, "async connect success");
                        SetNextState(RtspSendDescribe);
                        FD_CLR(_RtspSocket, &_writefd);
                    }

                    if (FD_ISSET(_RtspSocket, &_errorfd))
                    {
                        log(TAG, "event error occur");
                        break;
                    }
                }

                HandleRtspState();
            }

            //fclose(fp);
        });

        return true;
    }
    
    void RtspPlayer::Stop() {
        _PlayState = RtspTurnOff;
        _PlayThreadPtr->join();

    }

    bool RtspPlayer::PortIsOpen(int port)
    {
        const char *hostname = "127.0.0.1";

        int socket = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (socket < 0)
        {
            log(TAG, "rtp video socket init failed");
            return false;
        }
        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (::bind(socket, (const struct sockaddr *)&addr, (socklen_t)sizeof(addr)) < 0)
        {
            return false;
        }

        close(socket);
        return true;
    }
} // namespace RK
