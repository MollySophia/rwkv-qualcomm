//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_wkv7_output);


// op execute function declarations
template<typename TensorType>
GraphStatus wkv7OutputImpl(TensorType& out_0,
                    const TensorType& r,
                    const TensorType& state_in);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((wkv7Impl<Tensor>), "wkv7")
 */
DEF_PACKAGE_OP((wkv7OutputImpl<Tensor>), "wkv7_output")

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
// #define WKV_SPLIT_SIZE 8
// DEF_PACKAGE_OPTIMIZATION(EARLY + 3,
//   Op("wkv7", "r", "w", "k", "v", "a", "b", "state_in"),
//   OK,
//   AUTOSPLIT(1, "I", WKV_SPLIT_SIZE, 
//     Op("wkv7", 
//       TYPICAL_SLICE("r", "I"), 
//       TYPICAL_SLICE("w", "I"), 
//       TYPICAL_SLICE("k", "I"), 
//       TYPICAL_SLICE("v", "I"), 
//       TYPICAL_SLICE("a", "I"), 
//       TYPICAL_SLICE("b", "I"), 
//       // CHANGEDIM_SLICE("state_in", "I", 3)  // Use CHANGEDIM_SLICE for state since it has a different dimension layout
//       TYPICAL_SLICE("b", "I"), 
//     )
//   )
// )

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
// #include <qhmath_hvx_vector.h>
#include <hvx_internal.h>
// #include <qhblas_hvx.h>

static inline int32_t float_to_int(float scale)
{
    union { float f; int32_t i; } fp32 = { .f = scale };
    return fp32.i;
}

static void wkv7_output_hvx_f(const int seq_length, const int num_heads, const int head_size,
                  float *out_0,
                  const float *r,
                  const float *state_in) {
  float tmp_buf[32];
  float *outptr = out_0;
  HVX_Vector *state_ptr = (HVX_Vector *)(state_in);

  for (int t = 0; t < seq_length; t++) {
    for (int h = 0; h < num_heads; h++) {

      HVX_Vector r_vec_0 = *(HVX_Vector *)r;
      HVX_Vector r_vec_1 = *((HVX_Vector *)r + 1);

      for (int i = 0; i < head_size; i += 4) {
        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector state_vec_0 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_1 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_2 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_3 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_4 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_5 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_6 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_7 = *((HVX_Vector *)state_ptr++);

        HVX_Vector output_vec_0 = Q6_Vqf32_vmpy_VsfVsf(state_vec_0, r_vec_0);
        output_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_0, Q6_Vqf32_vmpy_VsfVsf(state_vec_1, r_vec_1));

        HVX_Vector output_vec_1 = Q6_Vqf32_vmpy_VsfVsf(state_vec_2, r_vec_0);
        output_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_1, Q6_Vqf32_vmpy_VsfVsf(state_vec_3, r_vec_1));

        HVX_Vector output_vec_2 = Q6_Vqf32_vmpy_VsfVsf(state_vec_4, r_vec_0);
        output_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_2, Q6_Vqf32_vmpy_VsfVsf(state_vec_5, r_vec_1));

        HVX_Vector output_vec_3 = Q6_Vqf32_vmpy_VsfVsf(state_vec_6, r_vec_0);
        output_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_3, Q6_Vqf32_vmpy_VsfVsf(state_vec_7, r_vec_1));

        for (int32_t i = 64; i >= 4; i >>= 1)
        {
          output_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_0, Q6_V_vlalign_VVR(output_vec_0, zero, i));
          output_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_1, Q6_V_vlalign_VVR(output_vec_1, zero, i));
          output_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_2, Q6_V_vlalign_VVR(output_vec_2, zero, i));
          output_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_3, Q6_V_vlalign_VVR(output_vec_3, zero, i));
        }

        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(output_vec_0);
        *outptr++ = tmp_buf[31];
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(output_vec_1);
        *outptr++ = tmp_buf[31];
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(output_vec_2);
        *outptr++ = tmp_buf[31];
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(output_vec_3);
        *outptr++ = tmp_buf[31];
      }
      r += head_size;
    }
  }
}

static void wkv7_output_hvx_hf(const int seq_length, const int num_heads, const int head_size,
                  __fp16 *out_0,
                  const __fp16 *r,
                  const __fp16 *state_in) {
  __fp16 *outptr = (__fp16 *)out_0;

  __fp16 __attribute__((aligned(VLEN))) tmp_buf[64];
  HVX_Vector *state_ptr = (HVX_Vector *)(state_in);

  for (int t = 0; t < seq_length; t++) {
    for (int h = 0; h < num_heads; h++) {
      HVX_Vector r_vec = *(HVX_Vector *)r;
      for (int i = 0; i < head_size; i += 8) {
        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector state_vec_0 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_1 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_2 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_3 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_4 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_5 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_6 = *((HVX_Vector *)state_ptr++);
        HVX_Vector state_vec_7 = *((HVX_Vector *)state_ptr++);

        HVX_Vector output_vec_0 = Q6_Vqf16_vmpy_VhfVhf(state_vec_0, r_vec);
        HVX_Vector output_vec_1 = Q6_Vqf16_vmpy_VhfVhf(state_vec_1, r_vec);
        HVX_Vector output_vec_2 = Q6_Vqf16_vmpy_VhfVhf(state_vec_2, r_vec);
        HVX_Vector output_vec_3 = Q6_Vqf16_vmpy_VhfVhf(state_vec_3, r_vec);
        HVX_Vector output_vec_4 = Q6_Vqf16_vmpy_VhfVhf(state_vec_4, r_vec);
        HVX_Vector output_vec_5 = Q6_Vqf16_vmpy_VhfVhf(state_vec_5, r_vec);
        HVX_Vector output_vec_6 = Q6_Vqf16_vmpy_VhfVhf(state_vec_6, r_vec);
        HVX_Vector output_vec_7 = Q6_Vqf16_vmpy_VhfVhf(state_vec_7, r_vec);

        for (int32_t n = 64; n >= 2; n >>= 1) {
          output_vec_0 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_0, Q6_V_vlalign_VVR(output_vec_0, zero, n));
          output_vec_1 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_1, Q6_V_vlalign_VVR(output_vec_1, zero, n));
          output_vec_2 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_2, Q6_V_vlalign_VVR(output_vec_2, zero, n));
          output_vec_3 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_3, Q6_V_vlalign_VVR(output_vec_3, zero, n));
          output_vec_4 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_4, Q6_V_vlalign_VVR(output_vec_4, zero, n));
          output_vec_5 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_5, Q6_V_vlalign_VVR(output_vec_5, zero, n));
          output_vec_6 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_6, Q6_V_vlalign_VVR(output_vec_6, zero, n));
          output_vec_7 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_7, Q6_V_vlalign_VVR(output_vec_7, zero, n));
        }
        output_vec_0 = Q6_Vhf_equals_Vqf16(output_vec_0);
        output_vec_1 = Q6_Vhf_equals_Vqf16(output_vec_1);
        output_vec_2 = Q6_Vhf_equals_Vqf16(output_vec_2);
        output_vec_3 = Q6_Vhf_equals_Vqf16(output_vec_3);
        output_vec_4 = Q6_Vhf_equals_Vqf16(output_vec_4);
        output_vec_5 = Q6_Vhf_equals_Vqf16(output_vec_5);
        output_vec_6 = Q6_Vhf_equals_Vqf16(output_vec_6);
        output_vec_7 = Q6_Vhf_equals_Vqf16(output_vec_7);
        *(HVX_Vector *)tmp_buf = output_vec_0;
        *outptr++ = tmp_buf[63];
        *(HVX_Vector *)tmp_buf = output_vec_1;
        *outptr++ = tmp_buf[63];
        *(HVX_Vector *)tmp_buf = output_vec_2;
        *outptr++ = tmp_buf[63];
        *(HVX_Vector *)tmp_buf = output_vec_3;
        *outptr++ = tmp_buf[63];
        *(HVX_Vector *)tmp_buf = output_vec_4;
        *outptr++ = tmp_buf[63];
        *(HVX_Vector *)tmp_buf = output_vec_5;
        *outptr++ = tmp_buf[63];
        *(HVX_Vector *)tmp_buf = output_vec_6;
        *outptr++ = tmp_buf[63];
        *(HVX_Vector *)tmp_buf = output_vec_7;
        *outptr++ = tmp_buf[63];
      }
      r += head_size;
    }
  }
}

#else

template <typename T>
static void wkv7_output_naive(const int seq_length, const int num_heads, const int head_size,
                  T *out_0,
                  const T *r,
                  const T *state_in) {

  return;
}

#endif

template<typename TensorType>
GraphStatus wkv7OutputImpl(TensorType& out_0,
                    const TensorType& r,
                    const TensorType& state_in)

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

  int num_heads = state_in.dim(1);
  int head_size = state_in.dim(2);
  int seq_length = state_in.dim(0);

#ifdef USE_HVX
// #if 0
  if (state_in.get_dtype() == DType::Float32) {
    auto r_ptr = (float*)r.raw_data_const();
    auto state_in_ptr = (float*)state_in.raw_data_const();
    auto out0_ptr = (float*)out_0.raw_data();
    wkv7_output_hvx_f(seq_length, num_heads, head_size,
                      out0_ptr,
                      r_ptr,
                      state_in_ptr);
  } else if (state_in.get_dtype() == DType::Float16) {
    auto r_ptr = (__fp16*)r.raw_data_const();
    auto state_in_ptr = (__fp16*)state_in.raw_data_const();
    auto out0_ptr = (__fp16*)out_0.raw_data();
    wkv7_output_hvx_hf(seq_length, num_heads, head_size,
                      out0_ptr,
                      r_ptr,
                      state_in_ptr);
  }
#else
  if (state_in.get_dtype() == DType::Float32) {
    auto r_ptr = (float*)r.raw_data_const();
    auto state_in_ptr = (float*)state_in.raw_data_const();
    auto out0_ptr = (float*)out_0.raw_data();
    wkv7_output_naive<float>(seq_length, num_heads, head_size,
                      out0_ptr,
                      r_ptr,
                      state_in_ptr);
  } else if (state_in.get_dtype() == DType::Float16) {
    auto r_ptr = (__fp16*)r.raw_data_const();
    auto state_in_ptr = (__fp16*)state_in.raw_data_const();
    auto out0_ptr = (__fp16*)out_0.raw_data();
    wkv7_output_naive<__fp16>(seq_length, num_heads, head_size,
                      out0_ptr,
                      r_ptr,
                      state_in_ptr);
  }
#endif
  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_wkv7_output);