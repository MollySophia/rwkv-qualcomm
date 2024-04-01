APP_ABI      := arm64-v8a
APP_STL      := c++_static
APP_PLATFORM := android-21
APP_CPPFLAGS += -std=c++11 -fexceptions -O3 -Wall -Werror -fvisibility=hidden -DQNN_API="__attribute__((visibility(\"default\")))"
APP_LDFLAGS  += -lc -lm -ldl
