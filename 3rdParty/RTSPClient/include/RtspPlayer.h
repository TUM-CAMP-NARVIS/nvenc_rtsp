//
//  RtspPlayer.hpp
//  toolForTest
//
//  Created by cx on 2018/9/6.
//  Copyright © 2018 cx. All rights reserved.
//

#ifndef RtspPlayer_h
#define RtspPlayer_h

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <unistd.h>
#include <string.h>
#include <functional>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <mutex>

extern "C" {
#include "sdp.h"
}
    
#define RTP_OFFSET (12)
#define FU_OFFSET (14)

#define RTP_PORT 12000
#define RTCP_PORT 13000

namespace RK {

    using RecvBufferFn = std::function<void(uint8_t*, ssize_t)>;
    using ImgPropRdyFn = std::function<void(int, int, int)>;

    enum RtspPlayerState {
        RtspSendOptions = 0,
        RtspHandleOptions,
        RtspSendDescribe,
        RtspHandleDescribe,
        RtspSendVideoSetup,
        RtspHandleVideoSetup,
        RtspSendAudioSetup,
        RtspHandleAudioSetup,
        RtspSendPlay,
        RtspHandlePlay,
        RtspSendPause,
        RtspHandlePause,
        RtspSendTerminate,
        RtspHandleTerminate,
        RtspIdle,
        RtspTurnOff,
    };
    
    enum RtspPlayerCSeq {
        RTSPOPTIONS = 1,
        RTSPDESCRIBE,
        RTSPVIDEO_SETUP,
        RTSPAUDIO_SETUP,
        RTSPPLAY,
        RTSPPAUSE,
        RTSPTEARDOWN,
    };
    
    // h264 nalu
    struct Nalu {
        unsigned type :5;
        unsigned nal_ref_idc :2;
        unsigned forbidden_zero_bit :1;
    };
    
    // h264 rtp fu
    struct FU {
        unsigned type :5;
        unsigned R :1;
        unsigned E :1;
        unsigned S :1;
    };

    struct ImgProps{
        int width;
        int height;
        int bytesPerPixel;
    };
    
    class RtspPlayer {
    public:
        typedef std::shared_ptr<RtspPlayer> Ptr;
        RtspPlayer(RecvBufferFn recv_cb = NULL, std::string name = "");
        ~RtspPlayer();
        bool Play(std::string url);
        void Stop();
        ImgProps GetImageProperties();

        RecvBufferFn recv_cb = NULL;
        ImgPropRdyFn imgPropRdy_cb = NULL;
    protected:
        bool NetworkInit(const char *ip, const short port);
        bool RTPSocketInit(int videoPort, int audioPort);
        bool getIPFromUrl(std::string url, char *ip, unsigned short *port);
        void EventInit();
        bool HandleRtspMsg(const char *buf, ssize_t bufsize);
        void HandleRtspState();
        
        //void HandleRtpMsg(const char *buf, ssize_t bufsize);
        
        // rtsp message send/handle function
        void SendDescribe(std::string url);
        void HandleDescribe(const char *buf, ssize_t bufsize);
        void RtspSetup(const std::string url, int track, int CSeq, char *proto, short rtp_port, short rtcp_port);
        void SendVideoSetup();
        bool HandleVideoSetup(const char *buf, ssize_t bufsize);
        void SendPlay(const std::string url);
        
        std::vector<std::string> GetSDPFromMessage(const char *buffer, size_t length, const char *pattern);
    private:
	    bool PortIsOpen(int port);

        struct ImgProps _ImgProps;

        std::atomic<bool> _Terminated;
        std::atomic<bool> _NetWorked;
        std::shared_ptr<std::thread> _PlayThreadPtr;
        std::atomic<RtspPlayerState> _PlayState;
        fd_set _readfd;
        fd_set _writefd;
        fd_set _errorfd;
        
        std::string _rtspurl;
        int _video_rtp_port;
        int _video_rtcp_port;

        char _rtspip[256];
        
        int _Eventfd = 0;
        int _RtspSocket = 0;
        int _RtpVideoSocket = 0;
        int _RtpAudioSocket = 0;
        struct sockaddr_in _RtpVideoAddr;
        
        struct sdp_payload *_SdpParser;
        
        long _RtspSessionID = 0;
        
        std::string TAG;

        static std::mutex _portMutex;
    };
    
} //namespace RK
#endif /* RtspPlayer_hpp */
