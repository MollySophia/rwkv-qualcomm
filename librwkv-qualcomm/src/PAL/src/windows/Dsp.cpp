//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All rights reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <windows.h>  // To get executable file path

#include <codecvt>  // To use codec for conversion
#include <vector>

#include "PAL/Debug.hpp"
#include "PAL/Dsp.hpp"

std::mutex pal::Dsp::s_mutex;
static thread_local void* sg_libRpcHandle{NULL};

static std::string getServiceBinaryPath(std::wstring const& serviceName) {
  // Get a handle to the SCM database
  SC_HANDLE handleSCManager = OpenSCManagerW(NULL,                   // local computer
                                             NULL,                   // ServicesActive database
                                             STANDARD_RIGHTS_READ);  // standard read access
  if (NULL == handleSCManager) {
    DEBUG_MSG(
        "Failed to open SCManager which is required to access service configuration on Windows. "
        "Error: %d",
        GetLastError());
    return std::string();
  }

  // Get a handle to the service
  SC_HANDLE handleService = OpenServiceW(handleSCManager,        // SCM database
                                         serviceName.c_str(),    // name of service
                                         SERVICE_QUERY_CONFIG);  // need query config access

  if (NULL == handleService) {
    DEBUG_MSG("Failed to open service %s which is required to query service information. Error: %d",
              std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(serviceName).c_str(),
              GetLastError());
    CloseServiceHandle(handleSCManager);
    return std::string();
  }

  // Query the buffer size required by service configuration
  // When first calling it with null pointer and zero buffer size,
  // this function acts as a query function to return how many bytes it requires
  // and set error to ERROR_INSUFFICIENT_BUFFER.

  DWORD bufferSize;  // Store the size of buffer used as an output
  if (!QueryServiceConfigW(handleService, NULL, 0, &bufferSize) &&
      (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
    DEBUG_MSG("Failed to query service configuration to get size of config object. Error: %d",
              GetLastError());
    CloseServiceHandle(handleService);
    CloseServiceHandle(handleSCManager);
    return std::string();
  }
  // Get the configuration of the specified service
  LPQUERY_SERVICE_CONFIGW serviceConfig =
      static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
  if (!QueryServiceConfigW(handleService, serviceConfig, bufferSize, &bufferSize)) {
    DEBUG_MSG("Failed to query service configuration. Error: %d", GetLastError());
    LocalFree(serviceConfig);
    CloseServiceHandle(handleService);
    CloseServiceHandle(handleSCManager);
    return std::string();
  }

  // Read the driver file path
  std::wstring driverPath = std::wstring(serviceConfig->lpBinaryPathName);
  // Get the parent directory of the driver file
  driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

  // Clean up resources
  LocalFree(serviceConfig);
  CloseServiceHandle(handleService);
  CloseServiceHandle(handleSCManager);

  // Driver path would contain invalid path string, like:
  // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
  // "\SystemRoot" should be replace with a correct one (e.g. C:\windows)
  const std::wstring systemRootPlaceholder = L"\\SystemRoot";
  if (0 != driverPath.compare(0, systemRootPlaceholder.length(), systemRootPlaceholder)) {
    DEBUG_MSG(
        "The string pattern does not match. We expect that we can find [%s] "
        "in the beginning of the queried path [%s].",
        std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(systemRootPlaceholder).c_str(),
        std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(driverPath).c_str());
    return std::string();
  }

  // Replace \SystemRoot with an absolute path which is got from system ENV windir
  // ENV name used to get the root path of the system
  const std::wstring systemRootEnv = L"windir";

  // Query the number of wide charactors this variable requires
  DWORD numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), NULL, 0);
  if (numWords == 0) {
    DEBUG_MSG("Failed to query the buffer size when calling GetEnvironmentVariableW().");
    return std::string();
  }

  // Query the actual system root name from environment variable
  std::vector<wchar_t> systemRoot(numWords + 1);
  numWords = GetEnvironmentVariableW(systemRootEnv.c_str(), systemRoot.data(), numWords + 1);
  if (numWords == 0) {
    DEBUG_MSG("Failed to read value from environment variables.");
    return std::string();
  }
  driverPath.replace(0, systemRootPlaceholder.length(), std::wstring(systemRoot.data()));

  // driverPath is wide char string, we need to convert it to std::string
  // Assume to use UTF-8 wide string for conversion
  return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(driverPath);
}

std::string pal::Dsp::getDspDriverPath() { return getServiceBinaryPath(L"qcnspmcdm"); }
