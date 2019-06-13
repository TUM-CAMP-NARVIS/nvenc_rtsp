# AMStreamLib

AMStream combines NvPipe with a rtsp connection.

If not building for Conan, this lib can be built together with the examples.

In case Conan is desired, this lib should be built individually.
The following guide shows how we can generate a Conan package from this library.

In future, this library will live on a conan server, making this guide only necessary if changes were made to the code.

## Toggle Conan Build

If this lib is built from this directory, it will automatically generate a Conan ready build folder.

If this lib is built with the examples, toggling ENABLE_CONAN to OFF while using CMake, directly links this library to the example executables. This is particularly useful for quick tests after changes.

## Requirements

Building AMStreamLib requires:

- Cuda (with Display Driver >= 4.18)
- Conan (refer to https://docs.conan.io/en/latest/installation.html)

Conan should take care of the OpenCV dependency

## Building shared library

Conan can use pre-built binaries for their package management.
That means, the first step is to build our source code into a library:

```
cd AMStreamLib
mkdir build && cd build
cmake ..
make
```

## Creating Conan package

In the next step, we want to create a conan package for later upload to a conan server.

While inside the build folder, run the following command to install the package onto your PC:

```
conan export-pkg . AMStreaming/1.0@myuser/stable -f
```
If you have trouble retrieving the conan package from conan you might want to compile the library with your specific configuration. (e.g. -s compiler.version=7)

If you want to upload to a server, please refer to this site:

https://docs.conan.io/en/latest/uploading_packages/uploading_to_remotes.html


## Using the new Package

conanfile.txt (on the same level as your CMakeLists.txt)
```
[requires]
AMStreaming/0.1@myuser/testing

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
