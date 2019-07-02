# nvenc_rtsp

nvenc_rtsp combines NvPipe with a rtsp connection.

## Requirements

Building NvStreamingLib requieres:

- Opencv (Installation guide: https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/)
- Cuda (with Display Driver >= 4.18)

Optionally, for using Kinect 360 as an input of video and depth stream, the library is needed:

- libfreenect (https://github.com/OpenKinect/libfreenect)
 
The code is easily extendable by reading any camera input into an OpenCV Mat.
Video frames should come as CV_8UC4 (RGBA) format and depth frames as CV_16UC1 format.

## Example

Example with libfreenect and Kinect 360. If libfreenect is not installed or the Kinect is not connected, the application will try to open a connected webcam instead. 

```c++


#include "nvenc_rtsp/ServerPipeRTSP.h"
#include "nvenc_rtsp/ClientPipeRTSP.h"

using namespace nvenc_rtsp;

int main(int argc, char *argv[])
{
    int key = 0;
 
    ServerPipeRTSP videoPipeServer(xop::NetInterface::getLocalIPAddress(), 55555, NVPIPE_RGBA32, NVPIPE_LOSSY);
    
    // video callback
    cv::Mat videoMat;
    bool videoMatUpdated = false;

    // Connect to client to local server
    ClientPipeRTSP videoPipeClient("rtsp://141.39.159.84:55555/live", NVPIPE_RGBA32,
                         [&](cv::Mat mat, uint64_t timestamp) {
                           videoMat = mat;
                           videoMatUpdated = true;
                         });
        
    // FILL OUT: Start camera image with OpenCV CV_8UC4 RGBA format

    // Main Loop ############################################################
    while (key != 27)
    {
        key = cv::waitKey(1);

        // Encode with nvenc h264 and send current videoframe
        videoPipeServer.send_frame( /* FILL OUT: Load current frame into a CV Mat*/ );

        // Show received image
        if(videoMatUpdated)
            cv::imshow("Client", videoMat);
    }

    videoPipeServer.cleanUp();
    videoPipeClient.cleanUp();
    
    return 0;
}


```

## Reference Third-Party

RTSP Server:
https://github.com/PHZ76/RtspServer (MIT Licence), adapted to project by Kevin Yu

RTSP Client:
https://github.com/sliver-chen/Simple-Rtsp-Client (no Licence), adapted to project by Kevin Yu

Nvidia video codec SDK:
https://developer.nvidia.com/nvidia-video-codec-sdk

NvPipe:
https://github.com/NVIDIA/NvPipe (licence under 3rdParty/NvPipe)

&copy; Research group MITI
