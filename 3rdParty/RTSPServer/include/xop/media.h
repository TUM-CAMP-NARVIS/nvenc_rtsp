// PHZ
// 2018-5-16

#ifndef XOP_MEDIA_H
#define XOP_MEDIA_H

#include <memory>

namespace xop
{

enum MediaType
{
    //PCMU = 0,	 
    PCMA = 8,
    H264 = 96,
    AAC  = 37,
    H265 = 265,   
    NONE
};	

enum FrameType
{
    VIDEO_FRAME_I = 0x01,	  
    VIDEO_FRAME_P = 0x02,
    VIDEO_FRAME_B = 0x03,    
    AUDIO_FRAME   = 0x11,   
};

struct AVFrame
{	
    AVFrame(uint32_t size = 0)
        :buffer(new uint8_t[size + 1])
    {
        this->size = size;
        type = 0;
        timestamp = 0;
    }

    uint8_t* buffer;
    uint32_t size;
    uint8_t  type;	
    uint32_t timestamp;	
};

#define MAX_MEDIA_CHANNEL 2

typedef  enum __MediaChannel_Id
{
    channel_0,
    channel_1
} MediaChannelId;

typedef uint32_t MediaSessionId;

}

#endif

