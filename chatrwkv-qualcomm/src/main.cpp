#include <iostream>
#include <memory>
#include <string>
#include <chrono>

#include "BuildId.hpp"
#include "DynamicLoadUtil.hpp"
#include "Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "PAL/GetOpt.hpp"
#include "chatrwkv-qualcomm-app.hpp"
#include "Utils.hpp"

static void* sg_backendHandle{nullptr};
static void* sg_modelHandle{nullptr};

namespace qnn {
namespace tools {
namespace rwkv_app {

void showHelp() {
  std::cout
      << "\nREQUIRED ARGUMENTS:\n"
      << "-------------------\n"
      << "  --model             <FILE>      Path to the model containing a QNN network.\n"
      << "\n"
      << "  --backend           <FILE>      Path to a QNN backend to execute the model.\n"
      << "\n"
      << "  --config_path       <FILE>      Path to the configuration file for the network.\n"
      << "\n"
      << "  --tokenizer_path    <FILE>      Path to the tokenizer file for the network.\n"
      << "\n"
      << "  --retrieve_context  <VAL>       Path to cached binary from which to load a saved\n"
         "                                  context from and execute graphs. --retrieve_context "
         "and\n"
         "                                  --model are mutually exclusive. Only one of the "
         "options\n"
         "                                  can be specified at a time.\n"
      << "\n\n"

      << "OPTIONAL ARGUMENTS:\n"
      << "-------------------\n"
      << "\n"
      << "  --prompt_path       <FILE>      Path to the prompt file for the network.\n"
      << "\n"
      << "  --output_dir        <DIR>       The directory to save output to. Defaults to "
         "./output.\n"
      << "\n"
      << "  --profiling_level   <VAL>       Enable profiling. Valid Values:\n"
         "                                    1. basic:    captures execution and init time.\n"
         "                                    2. detailed: in addition to basic, captures\n"
         "                                                 per Op timing for execution.\n"
      << "\n"
      << "  --save_context      <VAL>       Specifies that the backend context and metadata "
         "related \n"
         "                                  to graphs be saved to a binary file.\n"
         "                                  Value of this parameter is the name of the name\n"
         "                                  required to save the context binary to.\n"
         "                                  Saved in the same path as --output_dir option.\n"
         "                                  Note: --retrieve_context and --save_context are "
         "mutually\n"
         "                                  exclusive. Both options should not be specified at\n"
         "                                  the same time.\n"
      << "\n"
#ifdef QNN_ENABLE_DEBUG
      << "  --log_level                     Specifies max logging level to be set.  Valid "
         "settings: \n"
         "                                 \"error\", \"warn\", \"info\", \"verbose\" and "
         "\"debug\"."
         "\n"
#else
      << "  --log_level                     Specifies max logging level to be set.  Valid "
         "settings: \n"
         "                                 \"error\", \"warn\", \"info\" and \"verbose\"."
         "\n"
#endif
      << "\n"
      << "  --system_library     <FILE>     Path to QNN System library (libQnnSystem.so) needed to "
         "exercise reflection APIs\n"
         "                                  when loading a context from a binary cache.\n"
         "                                  libQnnSystem.so is provided under <target>/lib in the "
         "SDK.\n"
         "\n"
      << "  --version                       Print the QNN SDK version.\n"
      << "\n"
      << "  --help                          Show this help message.\n"
      << std::endl;
}

void showHelpAndExit(std::string&& error) {
  std::cerr << "ERROR: " << error << "\n";
  std::cerr << "Please check help below:\n";
  showHelp();
  std::exit(EXIT_FAILURE);
}

std::unique_ptr<rwkv_app::QnnRwkvApp> processCommandLine(int argc,
                                                             char** argv,
                                                             bool& loadFromCachedBinary) {
  enum OPTIONS {
    OPT_HELP             = 0,
    OPT_MODEL            = 1,
    OPT_BACKEND          = 2,
    OPT_PROMPT_PATH      = 3,
    OPT_CONFIG_PATH      = 4,
    OPT_TOKENIZER_PATH   = 5,
    OPT_LOG_LEVEL        = 6,
    OPT_PROFILING_LEVEL  = 7,
    OPT_RETRIEVE_CONTEXT = 8,
    OPT_SAVE_CONTEXT     = 9,
    OPT_VERSION          = 10,
    OPT_SYSTEM_LIBRARY   = 11
  };

  // Create the command line options
  static struct pal::Option s_longOptions[] = {
      {"help", pal::no_argument, NULL, OPT_HELP},
      {"model", pal::required_argument, NULL, OPT_MODEL},
      {"backend", pal::required_argument, NULL, OPT_BACKEND},
      {"prompt_path", pal::required_argument, NULL, OPT_PROMPT_PATH},
      {"config_path", pal::required_argument, NULL, OPT_CONFIG_PATH},
      {"tokenizer_path", pal::required_argument, NULL, OPT_TOKENIZER_PATH},
      {"profiling_level", pal::required_argument, NULL, OPT_PROFILING_LEVEL},
      {"log_level", pal::required_argument, NULL, OPT_LOG_LEVEL},
      {"retrieve_context", pal::required_argument, NULL, OPT_RETRIEVE_CONTEXT},
      {"save_context", pal::required_argument, NULL, OPT_SAVE_CONTEXT},
      {"system_library", pal::required_argument, NULL, OPT_SYSTEM_LIBRARY},
      {"version", pal::no_argument, NULL, OPT_VERSION},
      {NULL, 0, NULL, 0}};

  // Command line parsing loop
  int longIndex = 0;
  int opt       = 0;
  std::string modelPath;
  std::string backEndPath;
  std::string promptPath;
  std::string configPath;
  std::string tokenizerPath;
  rwkv_app::ProfilingLevel parsedProfilingLevel = ProfilingLevel::OFF;
  std::string cachedBinaryPath;
  std::string saveBinaryName;
  QnnLog_Level_t logLevel{QNN_LOG_LEVEL_ERROR};
  log::setLogLevel(logLevel);
  std::string systemLibraryPath;
  while ((opt = pal::getOptLongOnly(argc, argv, "", s_longOptions, &longIndex)) != -1) {
    switch (opt) {
      case OPT_HELP:
        showHelp();
        std::exit(EXIT_SUCCESS);
        break;

      case OPT_VERSION:
        std::cout << "QNN SDK " << qnn::tools::getBuildId() << "\n";
        std::exit(EXIT_SUCCESS);
        break;

      case OPT_MODEL:
        modelPath = pal::g_optArg;
        break;

      case OPT_BACKEND:
        backEndPath = pal::g_optArg;
        break;

      case OPT_PROMPT_PATH:
        promptPath = pal::g_optArg;
        break;

      case OPT_CONFIG_PATH:
        configPath = pal::g_optArg;
        break;

      case OPT_TOKENIZER_PATH:
        tokenizerPath = pal::g_optArg;
        break;

      case OPT_PROFILING_LEVEL:
        parsedProfilingLevel = rwkv_app::parseProfilingLevel(pal::g_optArg);
        if (parsedProfilingLevel == rwkv_app::ProfilingLevel::INVALID) {
          showHelpAndExit("Invalid profiling level.");
        }
        break;

      case OPT_LOG_LEVEL:
        logLevel = rwkv_app::parseLogLevel(pal::g_optArg);
        if (logLevel != QNN_LOG_LEVEL_MAX) {
          if (!log::setLogLevel(logLevel)) {
            showHelpAndExit("Unable to set log level.");
          }
        }
        break;

      case OPT_RETRIEVE_CONTEXT:
        loadFromCachedBinary = true;
        cachedBinaryPath     = pal::g_optArg;
        if (cachedBinaryPath.empty()) {
          showHelpAndExit("Cached context binary file not specified.");
        }
        break;

      case OPT_SAVE_CONTEXT:
        saveBinaryName = pal::g_optArg;
        if (saveBinaryName.empty()) {
          showHelpAndExit("Save context needs a file name.");
        }
        break;

      case OPT_SYSTEM_LIBRARY:
        systemLibraryPath = pal::g_optArg;
        if (systemLibraryPath.empty()) {
          showHelpAndExit("System library (libQnnSystem.so) path not specified.");
        }
        break;

      default:
        std::cerr << "ERROR: Invalid argument passed: " << argv[pal::g_optInd - 1]
                  << "\nPlease check the Arguments section in the description below.\n";
        showHelp();
        std::exit(EXIT_FAILURE);
    }
  }

  if (!modelPath.empty()) {
    if (!cachedBinaryPath.empty()) {
      showHelpAndExit(
          "Error: both --model and --cached_binary specified. Only one option is valid at a "
          "time.\n");
    }
  } else {
    if (cachedBinaryPath.empty()) {
      showHelpAndExit("Missing option: --model\n");
    }
  }

  if (!cachedBinaryPath.empty() && !saveBinaryName.empty()) {
    showHelpAndExit("Error: both --cached_binary and --save_binary specified");
  }

  if (backEndPath.empty()) {
    showHelpAndExit("Missing option: --backend\n");
  }

  if (configPath.empty()) {
    showHelpAndExit("Missing option: --config_path\n");
  }

  if (tokenizerPath.empty()) {
    showHelpAndExit("Missing option: --tokenizer_path\n");
  }

  if (loadFromCachedBinary && systemLibraryPath.empty()) {
    showHelpAndExit(
        "Missing option: --system_library. QNN System shared library (libQnnSystem.so) is needed "
        "to load from a cached binary\n");
  }

  QNN_INFO("Model: %s", modelPath.c_str());
  QNN_INFO("Backend: %s", backEndPath.c_str());
  QNN_INFO("Tokenizer Path: %s", tokenizerPath.c_str());

  QnnFunctionPointers qnnFunctionPointers;
  // Load backend and model .so and validate all the required function symbols are resolved
  auto statusCode = dynamicloadutil::getQnnFunctionPointers(backEndPath,
                                                            modelPath,
                                                            &qnnFunctionPointers,
                                                            &sg_backendHandle,
                                                            !loadFromCachedBinary,
                                                            &sg_modelHandle);
  if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
    if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
      exitWithMessage(
          "Error initializing QNN Function Pointers: could not load backend: " + backEndPath,
          EXIT_FAILURE);
    } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
      exitWithMessage(
          "Error initializing QNN Function Pointers: could not load model: " + modelPath,
          EXIT_FAILURE);
    } else {
      exitWithMessage("Error initializing QNN Function Pointers", EXIT_FAILURE);
    }
  }

  if (loadFromCachedBinary) {
    statusCode =
        dynamicloadutil::getQnnSystemFunctionPointers(systemLibraryPath, &qnnFunctionPointers);
    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
      exitWithMessage("Error initializing QNN System Function Pointers", EXIT_FAILURE);
    }
  }

  std::unique_ptr<rwkv_app::QnnRwkvApp> app(new rwkv_app::QnnRwkvApp(qnnFunctionPointers,
                                                                             promptPath,
                                                                             configPath,
                                                                             tokenizerPath,
                                                                             sg_backendHandle,
                                                                             parsedProfilingLevel,
                                                                             cachedBinaryPath,
                                                                             saveBinaryName));
  return app;
}

}  // namespace rwkv_app
}  // namespace tools
}  // namespace qnn

std::vector<float> softmax(std::vector<float> in) {
    std::vector<float> out(in);
    int length = in.size();
    float max_in = *std::max_element(in.begin(), in.end()), sum = 0;

    // #pragma omp parallel for
    for (int i = 0; i < length; i++) {
        out[i] = std::exp(in[i] - max_in);
        sum += out[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < length; i++)
        out[i] /= sum;

    return out;
}

int sample_logits(std::vector<half_float::half> logits, float temperature, float top_p, int top_k) {
  std::vector<float> probs(logits.begin(), logits.end());
  temperature = temperature < 0.1f ? 0.1f : temperature;
  temperature = temperature > 5.f ? 5.f : temperature;
  probs = softmax(probs);
  int length = probs.size();
  std::vector<std::pair<int, float>> sorted_probs;
  for (int i = 0; i < length; i++)
      sorted_probs.push_back({i, probs[i]});
  std::sort(sorted_probs.begin(), sorted_probs.end(), [](std::pair<int, float> p1, std::pair<int, float> p2) { return p1.second > p2.second; });

  float cumsum = 0;
  for (int i = 0; i < length; i++) {
      cumsum += sorted_probs[i].second;
      if (cumsum >= top_p) {
          length = i + 1;
          break;
      }
  }

  if (top_k > 0)
      length = std::min(length, top_k);
  
  std::vector<float> top_probs;
  for (int i = 0; i < length; i++)
      top_probs.push_back(std::pow(sorted_probs[i].second, 1.f / temperature));

  float random_value = rand() / float(RAND_MAX) * cumsum;
  cumsum = 0;
  for (int i = 0; i < length; i++) {
      cumsum += top_probs[i];
      if (cumsum >= random_value)
          return sorted_probs[i].first;
  }

  return sorted_probs[0].first;
}

int main(int argc, char** argv) {
  using namespace qnn::tools;

  if (!qnn::log::initializeLogging()) {
    std::cerr << "ERROR: Unable to initialize logging!\n";
    return EXIT_FAILURE;
  }

  {
    bool loadFromCachedBinary{false};
    std::unique_ptr<rwkv_app::QnnRwkvApp> app =
        rwkv_app::processCommandLine(argc, argv, loadFromCachedBinary);

    if (nullptr == app) {
      return EXIT_FAILURE;
    }

    QNN_INFO("chatrwkv-qualcomm build version: %s", qnn::tools::getBuildId().c_str());
    QNN_INFO("Backend        build version: %s", app->getBackendBuildId().c_str());

    if (rwkv_app::StatusCode::SUCCESS != app->initialize()) {
      return app->reportError("Initialization failure");
    }

    if (rwkv_app::StatusCode::SUCCESS != app->initializeBackend()) {
      return app->reportError("Backend Initialization failure");
    }

    auto devicePropertySupportStatus = app->isDevicePropertySupported();
    if (rwkv_app::StatusCode::FAILURE != devicePropertySupportStatus) {
      auto createDeviceStatus = app->createDevice();
      if (rwkv_app::StatusCode::SUCCESS != createDeviceStatus) {
        return app->reportError("Device Creation failure");
      }
    }

    if (rwkv_app::StatusCode::SUCCESS != app->initializeProfiling()) {
      return app->reportError("Profiling Initialization failure");
    }

    if (rwkv_app::StatusCode::SUCCESS != app->createPowerConfigId()) {
      return app->reportError("Power Config Id Creation failure");
    } else {
      if (rwkv_app::StatusCode::SUCCESS != app->setPowerConfig()) {
        return app->reportError("Power Config Set failure");
      }
    }

    // if (rwkv_app::StatusCode::SUCCESS != app->registerOpPackages()) {
    //   return app->reportError("Register Op Packages failure");
    // }

    if (!loadFromCachedBinary) {
      if (rwkv_app::StatusCode::SUCCESS != app->createContext()) {
        return app->reportError("Context Creation failure");
      }
      if (rwkv_app::StatusCode::SUCCESS != app->composeGraphs()) {
        return app->reportError("Graph Prepare failure");
      }
      if (rwkv_app::StatusCode::SUCCESS != app->finalizeGraphs()) {
        return app->reportError("Graph Finalize failure");
      }
    } else {
      if (rwkv_app::StatusCode::SUCCESS != app->createFromBinary()) {
        return app->reportError("Create From Binary failure");
      }
    }

    std::cout.setf(std::ios::unitbuf);
    // if (rwkv_app::StatusCode::SUCCESS != app->execute(0)) {
    //   return app->reportError("Graph Execution failure");
    // }
    // for (int i = 0; i < 10; i++) {
    //   std::cout << app->m_lastOutput[i] << " ";
    // }

    std::chrono::duration<double> duration_invoke;
    int token_num = 0;

    srand((unsigned)time(NULL));
    std::string prompt = "\n我们发现";
    std::vector<int> token_ids = app->m_tokenizer->Encode(prompt);
    for (auto token_id : token_ids) {
      std::chrono::high_resolution_clock::time_point infer_start = std::chrono::high_resolution_clock::now();
      if (rwkv_app::StatusCode::SUCCESS != app->execute(token_id)) {
        return app->reportError("Graph Execution failure");
      }
      std::chrono::high_resolution_clock::time_point infer_end = std::chrono::high_resolution_clock::now();
      duration_invoke += std::chrono::duration_cast<std::chrono::duration<double>>(infer_end - infer_start);
      token_num++;
    }

    int token = sample_logits(app->m_lastOutput, 0.7, 0.9, 0);

    std::cout << prompt;
    for (int i = 0; i < 300; i++) {
      std::cout << app->m_tokenizer->Decode(token);
      std::chrono::high_resolution_clock::time_point infer_start = std::chrono::high_resolution_clock::now();
      if (rwkv_app::StatusCode::SUCCESS != app->execute(token)) {
        return app->reportError("Graph Execution failure");
      }
      std::chrono::high_resolution_clock::time_point infer_end = std::chrono::high_resolution_clock::now();
      duration_invoke += std::chrono::duration_cast<std::chrono::duration<double>>(infer_end - infer_start);
      token_num++;

      token = sample_logits(app->m_lastOutput, 0.7, 0.9, 0);
    }
    std::cout << std::endl;

    std::cout << "Average time per token: " << duration_invoke.count() / token_num << "s" << std::endl;
    std::cout << "Average tokens per second: " << token_num / duration_invoke.count() << std::endl;

    if (rwkv_app::StatusCode::SUCCESS != app->freeGraphs()) {
      return app->reportError("Graph Free failure");
    }

    if (rwkv_app::StatusCode::SUCCESS != app->freeContext()) {
      return app->reportError("Context Free failure");
    }

    if (rwkv_app::StatusCode::SUCCESS != app->destroyPowerConfigId()) {
      return app->reportError("Power Config Id Destroy failure");
    }

    if (rwkv_app::StatusCode::FAILURE != devicePropertySupportStatus) {
      auto freeDeviceStatus = app->freeDevice();
      if (rwkv_app::StatusCode::SUCCESS != freeDeviceStatus) {
        return app->reportError("Device Free failure");
      }
    }
  }

  if (sg_backendHandle) {
    pal::dynamicloading::dlClose(sg_backendHandle);
  }
  if (sg_modelHandle) {
    pal::dynamicloading::dlClose(sg_modelHandle);
  }

  return EXIT_SUCCESS;
}
