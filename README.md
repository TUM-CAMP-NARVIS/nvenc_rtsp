# nvenc_rtsp

nvenc_rtsp combines NvPipe with a rtsp connection.

## Requirements

Building nvenc_rtsp requires:

- Cuda (with Display Driver >= 4.18)
- Conan (refer to https://docs.conan.io/en/latest/installation.html)

## Creating Conan package

Use the following command inside the root folder.

```
conan create . @<user>/testing

```
with \<user\> being the remote name

If you want to upload to a server use:


```
conan upload nvenc_rtsp/<version>@<user>/testing --all -r artekmed

```


## Using the new Package

conanfile.txt (on the same level as your CMakeLists.txt)
```
[requires]
nvenc_rtsp/<version>@<user>/testing

[generators]
cmake
```

Inside CMakeLists.txt:
```
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

...

add_executable(MyApp [...] )
target_link_libraries(MyApp PRIVATE ${CONAN_LIBS} [...] }

```

Building project:
```
mkdir build && cd build
conan install .. --build nvenc_rtsp
cmake ..
make

```

Run ```conan profile update settings.compiler.libcxx="libstdc++11" default``` in case you encounter linking issues.

The executables are located inside the folder build/bin/

## Tuning Streaming Performance

If the resolution of the video stream is large and/or LOSSLESS H.264 compression is enabled, the network buffer of the UDP socket on the receiver side might reach its limit fairly quickly and starts to discard packages.
To prevent this, nvenc_rtsp increases the UDP receive buffer size automatically on Windows.
On Linux, the maximum size needs to be set on the OS.

To check the current UDP/IP receive buffer default and limit, type the following commands into the terminal:

```
$ sysctl net.core.rmem_max
net.core.rmem_max = 212992
$ sysctl net.core.rmem_default
net.core.rmem_default = 212992
```

For increasing the buffer size permanently, add the following lines into /etc/sysctl.conf and reboot:

```
net.core.rmem_max=26214400
net.core.rmem_default=26214400
```

For immediate effect, type into the terminal:
```
$ sudo sysctl -w net.core.rmem_max=26214400
net.core.rmem_max = 26214400
$ sudo sysctl -w net.core.rmem_default=26214400
net.core.rmem_default = 26214400
```

## Reference Third-Party

RTSP Server:
https://github.com/PHZ76/RtspServer (MIT Licence), adapted to project by Kevin Yu

RTSP Client:
https://github.com/sliver-chen/Simple-Rtsp-Client (no Licence), adapted to project by Kevin Yu

&copy; Research group MITI
