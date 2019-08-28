#pragma once

#ifdef _WIN32
#   ifdef NVENCRTSP_DLL
#       define NVENCRTSP_EXPORT __declspec( dllexport )
#   else
#       define NVENCRTSP_EXPORT __declspec( dllimport )
#   endif
#else // _WIN32
#   define NVENCRTSP_EXPORT
#endif