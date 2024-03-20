LOCAL_PATH := $(call my-dir)
SUPPORTED_TARGET_ABI := arm64-v8a x86 x86_64

#============================ Verify Target Info and Application Variables =========================================
ifneq ($(filter $(TARGET_ARCH_ABI),$(SUPPORTED_TARGET_ABI)),)
    ifneq ($(APP_STL), c++_shared)
        $(error Unsupported APP_STL: "$(APP_STL)")
    endif
else
    $(error Unsupported TARGET_ARCH_ABI: '$(TARGET_ARCH_ABI)')
endif

#============================ Define Common Variables ===============================================================
# Include paths
PACKAGE_C_INCLUDES += -I $(QNN_SDK_ROOT)/include/QNN
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/CachingUtil
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/Log
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/PAL/include
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/Utils
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../src/WrapperUtils
PACKAGE_C_INCLUDES += -I $(LOCAL_PATH)/../include/flatbuffers

#========================== Define OpPackage Library Build Variables =============================================
include $(CLEAR_VARS)
LOCAL_C_INCLUDES               := $(PACKAGE_C_INCLUDES)
MY_SRC_FILES                   := $(wildcard $(LOCAL_PATH)/../src/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/Log/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/PAL/src/linux/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/PAL/src/common/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/Utils/*.cpp)
MY_SRC_FILES                   += $(wildcard $(LOCAL_PATH)/../src/WrapperUtils/*.cpp)
LOCAL_MODULE                   := chatrwkv-qualcomm
LOCAL_SRC_FILES                := $(subst make/,,$(MY_SRC_FILES))
LOCAL_LDLIBS                   := -lGLESv2 -lEGL
include $(BUILD_EXECUTABLE)
