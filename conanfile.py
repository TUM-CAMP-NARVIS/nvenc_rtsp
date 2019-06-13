from conans import ConanFile, tools, CMake

class nvenc_rtsp_Conan(ConanFile):
    name = "nvenc_rtsp"
    version = "0.1"
    generators = "cmake"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = "shared=True", "opencv:shared=True", "opencv:gtk=2"
    exports_sources = "include*", "src*", "3rdParty*", "CMakeLists.txt"

    description="NVIDIA-accelerated video compresssion library with built-in rtsp streaming"

    def requirements(self):
        self.requires("nvpipe/[>=0.1]@camposs/testing")
        self.requires("nvidia-video-codec-sdk/[>=9.0]@vendor/stable")
        self.requires("opencv/3.4.5@conan/stable")

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

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)

