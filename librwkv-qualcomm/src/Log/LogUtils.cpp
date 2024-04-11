//==============================================================================
//
//  Copyright (c) 2020, 2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "LogUtils.hpp"
#ifdef ANDROID
#include <android/log.h>
#endif

void qnn::log::utils::logStdoutCallback(const char* fmt,
                                        QnnLog_Level_t level,
                                        uint64_t timestamp,
                                        va_list argp) {
  const char* levelStr = "";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      levelStr = " ERROR ";
      break;
    case QNN_LOG_LEVEL_WARN:
      levelStr = "WARNING";
      break;
    case QNN_LOG_LEVEL_INFO:
      levelStr = "  INFO ";
      break;
    case QNN_LOG_LEVEL_DEBUG:
      levelStr = " DEBUG ";
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      levelStr = "VERBOSE";
      break;
    case QNN_LOG_LEVEL_MAX:
      levelStr = "UNKNOWN";
      break;
  }

  double ms = (double)timestamp / 1000000.0;
  // To avoid interleaved messages
  {
    std::lock_guard<std::mutex> lock(sg_logUtilMutex);
    fprintf(stdout, "%8.1fms [%-7s] ", ms, levelStr);
    vfprintf(stdout, fmt, argp);
    fprintf(stdout, "\n");
  }
}

#ifdef ANDROID
void qnn::log::utils::logAndroidCallback(const char* fmt,
                                        QnnLog_Level_t level,
                                        uint64_t timestamp,
                                        va_list argp){
  int loglevel = ANDROID_LOG_UNKNOWN;
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      loglevel = ANDROID_LOG_ERROR;
      break;
    case QNN_LOG_LEVEL_WARN:
      loglevel = ANDROID_LOG_WARN;
      break;
    case QNN_LOG_LEVEL_INFO:
      loglevel = ANDROID_LOG_INFO;
      break;
    case QNN_LOG_LEVEL_DEBUG:
      loglevel = ANDROID_LOG_DEBUG;
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      loglevel = ANDROID_LOG_VERBOSE;
      break;
    case QNN_LOG_LEVEL_MAX:
      loglevel = ANDROID_LOG_UNKNOWN;
      break;
  }

  __android_log_print(loglevel, "RWKV-QNN", (std::string(fmt) + "\n").c_str(), argp);
}
#endif
