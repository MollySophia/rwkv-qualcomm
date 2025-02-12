//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================
#include "QnnCpuOpPackage.h"
#include "CustomOpPackage.hpp"

using namespace qnn::custom;
using namespace qnn::custom::macros;

static Qnn_ErrorHandle_t RwkvWkvOpPackageInitialize(
  QnnOpPackage_GlobalInfrastructure_t globalInfrastructure) {

  QNN_CUSTOM_BE_ENSURE(!(CustomOpPackage::getIsInitialized()),QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED);

  INIT_BE_OP_PACKAGE(RwkvWkvOpPackage)

  REGISTER_PACKAGE_OP(wkv6)
  REGISTER_PACKAGE_OP(wkv7)

  // INIT_BE_PACKAGE_OPTIMIZATIONS();

  CustomOpPackage::setIsInitialized(true);

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t RwkvWkvOpPackageGetInfo(const QnnOpPackage_Info_t** info) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(opPkg->getPackageInfo(info));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t RwkvWkvOpPackageValidateOpConfig(Qnn_OpConfig_t opConfig) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  auto opRegistration = opPkg->getOpRegistration(opConfig.v1.typeName);

  QNN_CUSTOM_BE_ENSURE(opRegistration, QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE)

  QNN_CUSTOM_BE_ENSURE_STATUS(opRegistration->validateOpConfig(opConfig));

return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t RwkvWkvOpPackageCreateOpImpl(
   QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
   QnnOpPackage_Node_t node,
   QnnOpPackage_OpImpl_t* opImpl) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(
    opPkg->createOpImpl(graphInfrastructure, node, opImpl));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t RwkvWkvOpPackageFreeOpImpl(
   QnnCpuOpPackage_OpImpl_t* opImpl) {
  auto opPkg = CustomOpPackage::getInstance();

  QNN_CUSTOM_BE_ENSURE(opPkg, QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED);

  QNN_CUSTOM_BE_ENSURE_STATUS(opPkg->freeOpImpl(opImpl));

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t RwkvWkvOpPackageTerminate() {
  auto opPkg = CustomOpPackage::getInstance();

  CustomOpPackage::destroyInstance();
  opPkg->freeResolver();

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t RwkvWkvOpPackageLogInitialize(
QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) {
// function should be used if at least two backends support it
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY BE

  return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t RwkvWkvOpPackageLogSetLevel(
QnnLog_Level_t maxLogLevel) {
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY CPU BE

return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t RwkvWkvOpPackageLogTerminate() {
// USER SHOULD NOTE THIS FUNCTION IS UNUSED BY CPU BE

  return QNN_SUCCESS;
}


extern "C" QNN_API Qnn_ErrorHandle_t RwkvWkvOpPackageInterfaceProvider(
   QnnOpPackage_Interface_t* interface) {
  interface->interfaceVersion.major = 1;
  interface->interfaceVersion.minor = 4;
  interface->interfaceVersion.patch = 0;
  interface->v1_4.init              = RwkvWkvOpPackageInitialize;
  interface->v1_4.terminate         = RwkvWkvOpPackageTerminate;
  interface->v1_4.getInfo           = RwkvWkvOpPackageGetInfo;
  interface->v1_4.validateOpConfig  = RwkvWkvOpPackageValidateOpConfig;
  interface->v1_4.createOpImpl     =  RwkvWkvOpPackageCreateOpImpl;
  interface->v1_4.freeOpImpl        = RwkvWkvOpPackageFreeOpImpl;
  interface->v1_4.logInitialize     = RwkvWkvOpPackageLogInitialize;
  interface->v1_4.logSetLevel       = RwkvWkvOpPackageLogSetLevel;
  interface->v1_4.logTerminate      = RwkvWkvOpPackageLogTerminate;
  return QNN_SUCCESS;
}

