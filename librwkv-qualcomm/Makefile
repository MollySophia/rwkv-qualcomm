default: all

# define package name
PACKAGE_NAME := $(notdir $(shell pwd))

# define library prerequisites list
src_folder := src
make_dir := make
EXE_SOURCES = $(src_folder)

# define target_architecture
export TARGET_AARCH_VARS:= -march=x86-64

# define target name
export TARGET = linux-x86_64

# specify compiler
export CXX := clang++-9

.PHONY: all $(EXE_SOURCES) all_android

all: $(EXE_SOURCES) all_android

# Combined Targets
clean: clean_x86 clean_android

all_x86: clean_x86
	$(call build_if_exists,$(src_folder),-$(MAKE) -f $(make_dir)/Makefile.linux-x86_64)

clean_x86:
	@rm -rf bin obj include

# Android Targets

all_android: aarch64-android

aarch64-android: check_ndk clean_aarch64-android
	$(call build_if_exists,$(src_folder),$(ANDROID_NDK_ROOT)/ndk-build APP_ALLOW_MISSING_DEPS=true APP_ABI="arm64-v8a" NDK_PROJECT_PATH=./ NDK_APPLICATION_MK=$(make_dir)/Application.mk APP_BUILD_SCRIPT=$(make_dir)/Android.mk)
	@$(rename_target_dirs)

clean_android: check_ndk clean_aarch64-android

clean_aarch64-android:
	@rm -rf bin/aarch64-android
	@rm -rf obj/local/aarch64-android


# utilities
# Syntax: $(call build_if_exists <dir>,<cmd>)
build_if_exists = $(if $(wildcard $(1)),$(2),$(warning WARNING: $(1) does not exist. Skipping Compilation))
rename_target_dirs = find . -type d -execdir rename 's/arm64-v8a/aarch64-android/' '{}' \+ \
				&& mkdir -p bin/aarch64-android \
				&& mv obj/local/aarch64-android/*.a bin/aarch64-android/

check_ndk:
ifeq ($(ANDROID_NDK_ROOT),)
	$(error ERROR: ANDROID_NDK_ROOT not set, skipping compilation for Android platform(s).)
endif
