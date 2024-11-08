//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_wkv);


// op execute function declarations
template<typename TensorType>
GraphStatus wkvImpl(TensorType& out_0,
                     TensorType& out_1,
                    const TensorType& k,
                    const TensorType& v,
                    const TensorType& r,
                    const TensorType& in_3,
                    const TensorType& tf,
                    const TensorType& td);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((wkvImpl<Tensor>), "wkv")
 */
DEF_PACKAGE_OP((wkvImpl<Tensor>), "wkv")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkvImpl<PlainFloatTensor>), "wkv", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((wkvImpl<PlainFloatTensor>),
 * "wkv", wkvCostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax: DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core documentations
 */

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax: DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
     * order of parameters listed determines the order of parameters passed into op execution functions
 * if an op does not have a parameter order definition, parameter order passed into Qnn_addNode
 *   will be passed into op execution functions
 * if an op has a parameter order definition, any parameter passed into Qnn_addNode with unlisted
     *   name will be abandoned
 * if two or more op packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANDATORY refers to whether this parameter is required to be provided at Qnn_addNode
 * DEFAULT is used when MANDATORY is false
 *     if provided as Qnn_Param_t*,
 *       DEFAULT will be used for graph construction when this parameter is not provided at
 *       Qnn_addNode
 *     if provided as nullptr,
 *       graph construction will skip this parameter when this parameter is not provided at
 *       Qnn_addNode
 */


/* execute functions for ops */
#include <hvx_hexagon_protos.h>
#include <hexagon_types.h>

#ifdef USE_HVX
#include <qhmath_hvx_vector.h>
#include <hvx_internal.h>
#endif

template<typename TensorType>
GraphStatus wkvImpl(TensorType& out_0,
                     TensorType& out_1,
                    const TensorType& k,
                    const TensorType& v,
                    const TensorType& r,
                    const TensorType& in_3,
                    const TensorType& tf,
                    const TensorType& td)

{
  /*
   * To have good performance and stability, it is required to avoid heap memory
   * allocation in this function. The heap memory allocation includes but not
   * limited to calling malloc, operator new, constructing STL container objects
   * like std::vector with default allocator, and adding items like calling
   * std::vector::push_back to STL container objects with default allocator.
   *
   * Please check in SDK documentation for more information.
   */

#ifdef USE_HVX
  int num_heads = in_3.dim(1);
  int head_size = in_3.dim(2);
  for (int h = 0; h < num_heads; h++) {
    for (int i = 0; i < head_size; i++) {
      auto v_val = v(0, 0, h, i);
      float tmp = 0;
      for (int j = 0; j < head_size; j++) {
        auto k_val = k(0, 0, h, j);
        auto r_val = r(0, 0, h, j);
        auto kv_val = k_val * v_val;
        auto prev_state_val = in_3(0, h, i, j);
        auto td_val = td(0, h, 0, j);
        auto tf_val = tf(0, h, 0, j);
        tmp += r_val * (kv_val * tf_val + prev_state_val);
        out_1(0, h, i, j) = prev_state_val * td_val + kv_val;
      }
      out_0(0, h, 0, i) = tmp;
    }
  }
#else
  int num_heads = in_3.dim(1);
  int head_size = in_3.dim(2);
  for (int h = 0; h < num_heads; h++) {
    for (int i = 0; i < head_size; i++) {
      auto v_val = v(0, 0, h, i);
      float tmp = 0;
      for (int j = 0; j < head_size; j++) {
        auto k_val = k(0, 0, h, j);
        auto r_val = r(0, 0, h, j);
        auto kv_val = k_val * v_val;
        auto prev_state_val = in_3(0, h, i, j);
        auto td_val = td(0, h, 0, j);
        auto tf_val = tf(0, h, 0, j);
        tmp += r_val * (kv_val * tf_val + prev_state_val);
        out_1(0, h, i, j) = prev_state_val * td_val + kv_val;
      }
      out_0(0, h, 0, i) = tmp;
    }
  }
#endif
  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_wkv);