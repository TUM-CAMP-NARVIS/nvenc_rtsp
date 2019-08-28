from conans import ConanFile, tools, CMake

class nvenc_rtsp_Conan(ConanFile):
    name = "nvenc_rtsp"
    version = "0.1"
    generators = "cmake"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    # Not sure which opencv options are really necessary. Atleast one is needed for waitKey to function.
    default_options = {
        "shared": True, 
        "opencv:shared": True, 
        # "opencv:with_imgproc": True, 
        # "opencv:with_imgcodecs": True, 
        # "opencv:with_gtk": True, 
        # "opencv:with_highgui": True
        }
    exports_sources = "include*", "src*", "3rdParty*", "CMakeLists.txt"

    description="NVIDIA-accelerated video compresssion library with built-in rtsp streaming"

    requires = (
        "opencv/[>=3.0]@camposs/stable",
        "nvpipe/[>=0.2]@camposs/stable",
        )


    def build(self):
        cmake = CMake(self)
        cmake.definitions["BUILD_SHARED_LIBS"] = self.options.shared
        cmake.configure()
        cmake.build()

    def package(self):
        self.copy(pattern="*.h", dst="include", src="include")
        self.copy(pattern="*.h", dst="include", src="3rdParty/RTSPClient/include")
        self.copy(pattern="*.h", dst="include", src="3rdParty/RTSPServer/include")
        self.copy(pattern="*.so", dst="lib", keep_path=False)
        self.copy(pattern="*.lib", dst="lib", keep_path=False)
        self.copy(pattern="*.dll", dst="bin", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)

