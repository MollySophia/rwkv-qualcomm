set(LIB "rwkv-qualcomm")
set(LIB_SOURCES "librwkv-qualcomm-app.cpp"
                "librwkv-qualcomm.cpp"
                "Log/Logger.cpp"
                "Log/LogUtils.cpp"
                "PAL/src/windows/Common.cpp"
                "PAL/src/windows/Directory.cpp"
                "PAL/src/windows/DynamicLoading.cpp"
                "PAL/src/windows/FileOp.cpp"
                "PAL/src/windows/Path.cpp"
                "PAL/src/common/GetOpt.cpp"
                "PAL/src/common/StringOp.cpp"
                "Utils/DataUtil.cpp"
                "Utils/DynamicLoadUtil.cpp"
                "Utils/IOTensor.cpp"
                "Utils/Utils.cpp"
                "WrapperUtils/QnnWrapperUtils.cpp")

add_library(${LIB} STATIC ${LIB_SOURCES})

target_compile_definitions(${LIB} PUBLIC "-DNOMINMAX")
target_link_libraries(${LIB} PRIVATE Shlwapi Shell32)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob3")
target_include_directories(${LIB} PUBLIC CachingUtil
                                         Log
                                         PAL/include
                                         Utils
                                         WrapperUtils
                                         ${CMAKE_BINARY_DIR}
                                         ${QNN_SDK_ROOT}/include/QNN
                                         ./)
