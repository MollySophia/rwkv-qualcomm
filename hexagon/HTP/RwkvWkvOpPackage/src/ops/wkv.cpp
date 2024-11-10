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
// #include <qhmath_hvx_vector.h>
#include <hvx_internal.h>
// #include <qhblas_hvx.h>

static inline int32_t float_to_int(float scale)
{
    union { float f; int32_t i; } fp32 = { .f = scale };
    return fp32.i;
}

static void wkv_hvx_f(const int num_heads, const int head_size,
                  float *out_0,
                  float *out_1,
                  const float *k,
                  const float *v,
                  const float *r,
                  const float *in_3,
                  const float *tf,
                  const float *td) {
  float __attribute__((aligned(VLEN))) buffer[VLEN_WORD];
  float *outptr = out_0;
  memset(outptr, 0, sizeof(float)*num_heads*head_size);
  HVX_Vector *prev_state_ptr = (HVX_Vector *)(in_3);
  HVX_Vector *out_state_ptr = (HVX_Vector *)(out_1);
  for (int h = 0; h < num_heads; h++) {
    HVX_Vector *r_ptr = (HVX_Vector *)(r + h * head_size);
    HVX_Vector *k_ptr = (HVX_Vector *)(k + h * head_size);
    HVX_Vector *tf_ptr = (HVX_Vector *)(tf + h * head_size);
    HVX_Vector *td_ptr = (HVX_Vector *)(td + h * head_size);
    for (int i = 0; i < head_size; i++) {
      auto v_vec = Q6_V_vsplat_R(float_to_int(v[h * head_size + i]));

      HVX_Vector kv_vec_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*k_ptr, v_vec));
      HVX_Vector kv_vec_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*(k_ptr + 1), v_vec));

      HVX_Vector vtmp_0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_0, *tf_ptr), *prev_state_ptr));
      HVX_Vector vtmp_1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_1, *(tf_ptr+1)), *(prev_state_ptr+1)));
      // dot product r, vtmp
      {
        vtmp_0 = Q6_Vqf32_vmpy_VsfVsf(vtmp_0, *r_ptr);
        vtmp_1 = Q6_Vqf32_vmpy_VsfVsf(vtmp_1, *(r_ptr+1));
        vtmp_0 = Q6_Vqf32_vadd_Vqf32Vqf32(vtmp_0, vtmp_1);
        HVX_Vector zero = Q6_V_vzero();
        for (int32_t i = 64; i >= 4; i >>= 1)
        {
          vtmp_0 = Q6_Vqf32_vadd_Vqf32Vqf32(vtmp_0, Q6_V_vlalign_VVR(vtmp_0, zero, i));
        }
        vtmp_0 = Q6_Vsf_equals_Vqf32(vtmp_0);
        *(HVX_Vector *) buffer = vtmp_0;

        *outptr += buffer[31];
      }

      vstu_variable(out_state_ptr, VLEN,
        Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*prev_state_ptr, *td_ptr), kv_vec_0)));
      vstu_variable(out_state_ptr+1, VLEN,
        Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*(prev_state_ptr+1), *(td_ptr+1)), kv_vec_1)));

      out_state_ptr += 2;
      prev_state_ptr += 2;
      outptr++;
    }
  }
}

static void wkv_hvx_hf(const int num_heads, const int head_size,
                  __fp16 *out_0,
                  __fp16 *out_1,
                  const __fp16 *k,
                  const __fp16 *v,
                  const __fp16 *r,
                  const __fp16 *in_3,
                  const __fp16 *tf,
                  const __fp16 *td) {
  __fp16 __attribute__((aligned(VLEN))) buffer[VLEN_SHORT];
  __fp16 *outptr = out_0;
  memset(outptr, 0, sizeof(__fp16)*num_heads*head_size);
  HVX_Vector *prev_state_ptr = (HVX_Vector *)(in_3);
  HVX_Vector *out_state_ptr = (HVX_Vector *)(out_1);
  for (int h = 0; h < num_heads; h++) {
    HVX_Vector *r_ptr = (HVX_Vector *)(r + h * head_size);
    HVX_Vector *k_ptr = (HVX_Vector *)(k + h * head_size);
    HVX_Vector *tf_ptr = (HVX_Vector *)(tf + h * head_size);
    HVX_Vector *td_ptr = (HVX_Vector *)(td + h * head_size);
    for (int i = 0; i < head_size; i++) {
      auto v_vec = Q6_V_vsplat_R(fp16_to_bits((__fp16*)&v[h * head_size + i]));

      HVX_Vector kv_vec = Q6_Vqf16_vmpy_VhfVhf(*k_ptr, v_vec);

      HVX_Vector vtmp = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec, *tf_ptr), *prev_state_ptr);
      // dot product r, vtmp
      {
        vtmp = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp, *r_ptr);
        HVX_Vector zero = Q6_V_vzero();
        for (int32_t i = 64; i >= 2; i >>= 1)
        {
          vtmp = Q6_Vqf16_vadd_Vqf16Vqf16(vtmp, Q6_V_vlalign_VVR(vtmp, zero, i));
        }
        vtmp = Q6_Vhf_equals_Vqf16(vtmp);
        *(HVX_Vector *) buffer = vtmp;

        *outptr += buffer[63];
      }

      vstu_variable(out_state_ptr, VLEN,
        Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(*prev_state_ptr, *td_ptr), kv_vec)));

      out_state_ptr++;
      prev_state_ptr++;
      outptr++;
    }
  }
}

#else

template <typename T>
static void wkv_naive(const int num_heads, const int head_size,
                  T *out_0,
                  T *out_1,
                  const T *k,
                  const T *v,
                  const T *r,
                  const T *in_3,
                  const T *tf,
                  const T *td) {
  for (int h = 0; h < num_heads; h++) {
    for (int i = 0; i < head_size; i++) {
      auto v_val = v[h * head_size + i];
      T tmp = 0;
      for (int j = 0; j < head_size; j++) {
        auto k_val = k[h * head_size + j];
        auto r_val = r[h * head_size + j];
        auto kv_val = k_val * v_val;
        auto prev_state_val = in_3[h * head_size * head_size + i * head_size + j];
        auto td_val = td[h * head_size + j];
        auto tf_val = tf[h * head_size + j];
        tmp += r_val * (kv_val * tf_val + prev_state_val);
        out_1[h * head_size * head_size + i * head_size + j] = prev_state_val * td_val + kv_val;
      }
      out_0[h * head_size + i] = tmp;
    }
  }
}

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
  if (k.get_dtype() == DType::Float32) {
    auto k_ptr = (float*)k.raw_data_const();
    auto v_ptr = (float*)v.raw_data_const();
    auto r_ptr = (float*)r.raw_data_const();
    auto in_3_ptr = (float*)in_3.raw_data_const();
    auto tf_ptr = (float*)tf.raw_data_const();
    auto td_ptr = (float*)td.raw_data_const();
    auto out0_ptr = (float*)out_0.raw_data();
    auto out1_ptr = (float*)out_1.raw_data();
    wkv_hvx_f(num_heads, head_size,
              out0_ptr,
              out1_ptr,
              k_ptr,
              v_ptr,
              r_ptr,
              in_3_ptr,
              tf_ptr,
              td_ptr);
  } else if (k.get_dtype() == DType::Float16) {
    auto k_ptr = (__fp16*)k.raw_data_const();
    auto v_ptr = (__fp16*)v.raw_data_const();
    auto r_ptr = (__fp16*)r.raw_data_const();
    auto in_3_ptr = (__fp16*)in_3.raw_data_const();
    auto tf_ptr = (__fp16*)tf.raw_data_const();
    auto td_ptr = (__fp16*)td.raw_data_const();
    auto out0_ptr = (__fp16*)out_0.raw_data();
    auto out1_ptr = (__fp16*)out_1.raw_data();
    wkv_hvx_hf(num_heads, head_size,
              out0_ptr,
              out1_ptr,
              k_ptr,
              v_ptr,
              r_ptr,
              in_3_ptr,
              tf_ptr,
              td_ptr);
  }
#else
  int num_heads = in_3.dim(1);
  int head_size = in_3.dim(2);
  if (k.get_dtype() == DType::Float32) {
    auto k_ptr = (float*)k.raw_data_const();
    auto v_ptr = (float*)v.raw_data_const();
    auto r_ptr = (float*)r.raw_data_const();
    auto in_3_ptr = (float*)in_3.raw_data_const();
    auto tf_ptr = (float*)tf.raw_data_const();
    auto td_ptr = (float*)td.raw_data_const();
    auto out0_ptr = (float*)out_0.raw_data();
    auto out1_ptr = (float*)out_1.raw_data();
    wkv_naive<float>(num_heads, head_size,
                      out0_ptr,
                      out1_ptr,
                      k_ptr,
                      v_ptr,
                      r_ptr,
                      in_3_ptr,
                      tf_ptr,
                      td_ptr);
  } else if (k.get_dtype() == DType::Float16) {
    auto k_ptr = (__fp16*)k.raw_data_const();
    auto v_ptr = (__fp16*)v.raw_data_const();
    auto r_ptr = (__fp16*)r.raw_data_const();
    auto in_3_ptr = (__fp16*)in_3.raw_data_const();
    auto tf_ptr = (__fp16*)tf.raw_data_const();
    auto td_ptr = (__fp16*)td.raw_data_const();
    auto out0_ptr = (__fp16*)out_0.raw_data();
    auto out1_ptr = (__fp16*)out_1.raw_data();
    wkv_naive<__fp16>(num_heads, head_size,
                      out0_ptr,
                      out1_ptr,
                      k_ptr,
                      v_ptr,
                      r_ptr,
                      in_3_ptr,
                      tf_ptr,
                      td_ptr);
  }
#endif
  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_wkv);