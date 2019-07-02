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

## Reference Third-Party

RTSP Server:
https://github.com/PHZ76/RtspServer (MIT Licence), adapted to project by Kevin Yu

RTSP Client:
https://github.com/sliver-chen/Simple-Rtsp-Client (no Licence), adapted to project by Kevin Yu

&copy; Research group MITI
