# nvenc_rtsp

nvenc_rtsp combines NvPipe with a rtsp connection.

## Requirements

Building nvenc_rtsp requires:

- Cuda (with Display Driver >= 4.18)
- Conan (refer to https://docs.conan.io/en/latest/installation.html)

## Creating Conan package

Use the following command inside the root folder.

```
conan create . @myuser/testing --build missing

```
In some cases, adding -s compiler.version=X or -s compiler.libcxx=libstdc++11 can solve build errors.

If you want to upload to a server, please refer to this site:

https://docs.conan.io/en/latest/uploading_packages/uploading_to_remotes.html


## Using the new Package

conanfile.txt (on the same level as your CMakeLists.txt)
```
[requires]
nvenc_rtsp/0.1@myuser/testing

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
conan install ..
cmake ..
make

```

The executables are located inside the folder build/bin/


&copy; Research group MITI
