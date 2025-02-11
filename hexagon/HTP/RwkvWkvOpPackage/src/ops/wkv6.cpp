//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_wkv6);


// op execute function declarations
template<typename TensorType>
GraphStatus wkv6Impl(TensorType& out_0,
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
 * e.g. DEF_PACKAGE_OP((wkv6Impl<Tensor>), "wkv6")
 */
DEF_PACKAGE_OP((wkv6Impl<Tensor>), "wkv6")

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

static void wkv6_hvx_f(const int seq_length, const int num_heads, const int head_size,
                  float *out_0,
                  float *out_1,
                  const float *k,
                  const float *v,
                  const float *r,
                  const float *in_3,
                  const float *tf,
                  const float *td) {
  HVX_Vector *outptr = (HVX_Vector *)out_0;
  const float *v_ptr = v;
  const float *r_ptr = r;
  const float *k_ptr = k;
  const float *td_ptr = td;

  for (int t = 0; t < seq_length; t++) {
    HVX_Vector *prev_state_ptr = t > 0 ? (HVX_Vector *)(out_1) : (HVX_Vector *)(in_3);
    HVX_Vector *out_state_ptr = (HVX_Vector *)(out_1);
    const float *tf_ptr = tf;

    for (int h = 0; h < num_heads; h++) {
      HVX_Vector v_vec_0 = *(HVX_Vector *)v_ptr;
      HVX_Vector v_vec_1 = *((HVX_Vector *)v_ptr + 1);
      HVX_Vector output_vec_0 = Q6_V_vzero();
      HVX_Vector output_vec_1 = Q6_V_vzero();
      for (int i = 0; i < head_size; i += 4) {
        HVX_Vector k_vec_0 = Q6_V_vsplat_R(float_to_int(*k_ptr++));
        HVX_Vector k_vec_1 = Q6_V_vsplat_R(float_to_int(*k_ptr++));
        HVX_Vector k_vec_2 = Q6_V_vsplat_R(float_to_int(*k_ptr++));
        HVX_Vector k_vec_3 = Q6_V_vsplat_R(float_to_int(*k_ptr++));
        HVX_Vector kv_vec_00 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(k_vec_0, v_vec_0));
        HVX_Vector kv_vec_01 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(k_vec_0, v_vec_1));
        HVX_Vector kv_vec_10 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(k_vec_1, v_vec_0));
        HVX_Vector kv_vec_11 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(k_vec_1, v_vec_1));
        HVX_Vector kv_vec_20 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(k_vec_2, v_vec_0));
        HVX_Vector kv_vec_21 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(k_vec_2, v_vec_1));
        HVX_Vector kv_vec_30 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(k_vec_3, v_vec_0));
        HVX_Vector kv_vec_31 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(k_vec_3, v_vec_1));

        HVX_Vector tf_vec_0 = Q6_V_vsplat_R(float_to_int(*tf_ptr++));
        HVX_Vector tf_vec_1 = Q6_V_vsplat_R(float_to_int(*tf_ptr++));
        HVX_Vector tf_vec_2 = Q6_V_vsplat_R(float_to_int(*tf_ptr++));
        HVX_Vector tf_vec_3 = Q6_V_vsplat_R(float_to_int(*tf_ptr++));
        HVX_Vector vtmp_00 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_00, tf_vec_0), *prev_state_ptr));
        HVX_Vector vtmp_01 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_01, tf_vec_0), *(prev_state_ptr + 1)));
        HVX_Vector vtmp_10 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_10, tf_vec_1), *(prev_state_ptr + 2)));
        HVX_Vector vtmp_11 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_11, tf_vec_1), *(prev_state_ptr + 3)));
        HVX_Vector vtmp_20 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_20, tf_vec_2), *(prev_state_ptr + 4)));
        HVX_Vector vtmp_21 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_21, tf_vec_2), *(prev_state_ptr + 5)));
        HVX_Vector vtmp_30 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_30, tf_vec_3), *(prev_state_ptr + 6)));
        HVX_Vector vtmp_31 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(kv_vec_31, tf_vec_3), *(prev_state_ptr + 7)));

        HVX_Vector td_vec_0 = Q6_V_vsplat_R(float_to_int(*td_ptr++));
        HVX_Vector td_vec_1 = Q6_V_vsplat_R(float_to_int(*td_ptr++));
        HVX_Vector td_vec_2 = Q6_V_vsplat_R(float_to_int(*td_ptr++));
        HVX_Vector td_vec_3 = Q6_V_vsplat_R(float_to_int(*td_ptr++));
        *out_state_ptr = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*prev_state_ptr, td_vec_0), kv_vec_00));
        *(out_state_ptr + 1) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*(prev_state_ptr + 1), td_vec_0), kv_vec_01));
        *(out_state_ptr + 2) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*(prev_state_ptr + 2), td_vec_1), kv_vec_10));
        *(out_state_ptr + 3) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*(prev_state_ptr + 3), td_vec_1), kv_vec_11));
        *(out_state_ptr + 4) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*(prev_state_ptr + 4), td_vec_2), kv_vec_20));
        *(out_state_ptr + 5) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*(prev_state_ptr + 5), td_vec_2), kv_vec_21));
        *(out_state_ptr + 6) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*(prev_state_ptr + 6), td_vec_3), kv_vec_30));
        *(out_state_ptr + 7) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(*(prev_state_ptr + 7), td_vec_3), kv_vec_31));

        HVX_Vector r_vec_0 = Q6_V_vsplat_R(float_to_int(*r_ptr++));
        HVX_Vector r_vec_1 = Q6_V_vsplat_R(float_to_int(*r_ptr++));
        HVX_Vector r_vec_2 = Q6_V_vsplat_R(float_to_int(*r_ptr++));
        HVX_Vector r_vec_3 = Q6_V_vsplat_R(float_to_int(*r_ptr++));
        vtmp_00 = Q6_Vqf32_vmpy_VsfVsf(vtmp_00, r_vec_0);
        vtmp_01 = Q6_Vqf32_vmpy_VsfVsf(vtmp_01, r_vec_0);
        vtmp_10 = Q6_Vqf32_vmpy_VsfVsf(vtmp_10, r_vec_1);
        vtmp_11 = Q6_Vqf32_vmpy_VsfVsf(vtmp_11, r_vec_1);
        vtmp_20 = Q6_Vqf32_vmpy_VsfVsf(vtmp_20, r_vec_2);
        vtmp_21 = Q6_Vqf32_vmpy_VsfVsf(vtmp_21, r_vec_2);
        vtmp_30 = Q6_Vqf32_vmpy_VsfVsf(vtmp_30, r_vec_3);
        vtmp_31 = Q6_Vqf32_vmpy_VsfVsf(vtmp_31, r_vec_3);

        output_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_0, vtmp_00);
        output_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_0, vtmp_10);
        output_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_0, vtmp_20);
        output_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_0, vtmp_30);
        output_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_1, vtmp_01);
        output_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_1, vtmp_11);
        output_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_1, vtmp_21);
        output_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(output_vec_1, vtmp_31);

        out_state_ptr += 8;
        prev_state_ptr += 8;
      }
      *outptr++ = Q6_Vsf_equals_Vqf32(output_vec_0);
      *outptr++ = Q6_Vsf_equals_Vqf32(output_vec_1);
      v_ptr += head_size;
    }
  }
}

static void wkv6_hvx_hf(const int seq_length, const int num_heads, const int head_size,
                  __fp16 *out_0,
                  __fp16 *out_1,
                  const __fp16 *k,
                  const __fp16 *v,
                  const __fp16 *r,
                  const __fp16 *in_3,
                  const __fp16 *tf,
                  const __fp16 *td) {
  HVX_Vector *outptr = (HVX_Vector *)out_0;
  __fp16 *k_ptr = (__fp16*)k;
  __fp16 *r_ptr = (__fp16*)r;
  __fp16 *td_ptr = (__fp16*)td;

  for (int t = 0; t < seq_length; t++) {
    HVX_Vector *prev_state_ptr = t > 0 ? (HVX_Vector *)(out_1) : (HVX_Vector *)(in_3);
    HVX_Vector *out_state_ptr = (HVX_Vector *)(out_1);
    __fp16 *tf_ptr = (__fp16*)tf;

    for (int h = 0; h < num_heads; h++) {
      HVX_Vector v_vec = *(HVX_Vector *)v;
      HVX_Vector output_vec = Q6_V_vzero();
      for (int i = 0; i < head_size; i += 8) {
        HVX_Vector kv_vec_0 = Q6_Vqf16_vmpy_VhfVhf(v_vec, Q6_Vh_vsplat_R(fp16_to_bits(k_ptr++)));
        HVX_Vector kv_vec_1 = Q6_Vqf16_vmpy_VhfVhf(v_vec, Q6_Vh_vsplat_R(fp16_to_bits(k_ptr++)));
        HVX_Vector kv_vec_2 = Q6_Vqf16_vmpy_VhfVhf(v_vec, Q6_Vh_vsplat_R(fp16_to_bits(k_ptr++)));
        HVX_Vector kv_vec_3 = Q6_Vqf16_vmpy_VhfVhf(v_vec, Q6_Vh_vsplat_R(fp16_to_bits(k_ptr++)));
        HVX_Vector kv_vec_4 = Q6_Vqf16_vmpy_VhfVhf(v_vec, Q6_Vh_vsplat_R(fp16_to_bits(k_ptr++)));
        HVX_Vector kv_vec_5 = Q6_Vqf16_vmpy_VhfVhf(v_vec, Q6_Vh_vsplat_R(fp16_to_bits(k_ptr++)));
        HVX_Vector kv_vec_6 = Q6_Vqf16_vmpy_VhfVhf(v_vec, Q6_Vh_vsplat_R(fp16_to_bits(k_ptr++)));
        HVX_Vector kv_vec_7 = Q6_Vqf16_vmpy_VhfVhf(v_vec, Q6_Vh_vsplat_R(fp16_to_bits(k_ptr++)));

        HVX_Vector vtmp_0 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec_0, Q6_Vh_vsplat_R(fp16_to_bits(tf_ptr++))), *prev_state_ptr);
        HVX_Vector vtmp_1 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec_1, Q6_Vh_vsplat_R(fp16_to_bits(tf_ptr++))), *(prev_state_ptr + 1));
        HVX_Vector vtmp_2 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec_2, Q6_Vh_vsplat_R(fp16_to_bits(tf_ptr++))), *(prev_state_ptr + 2));
        HVX_Vector vtmp_3 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec_3, Q6_Vh_vsplat_R(fp16_to_bits(tf_ptr++))), *(prev_state_ptr + 3));
        HVX_Vector vtmp_4 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec_4, Q6_Vh_vsplat_R(fp16_to_bits(tf_ptr++))), *(prev_state_ptr + 4));
        HVX_Vector vtmp_5 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec_5, Q6_Vh_vsplat_R(fp16_to_bits(tf_ptr++))), *(prev_state_ptr + 5));
        HVX_Vector vtmp_6 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec_6, Q6_Vh_vsplat_R(fp16_to_bits(tf_ptr++))), *(prev_state_ptr + 6));
        HVX_Vector vtmp_7 = Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_Vqf16Vhf(kv_vec_7, Q6_Vh_vsplat_R(fp16_to_bits(tf_ptr++))), *(prev_state_ptr + 7));

        *(out_state_ptr) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(*prev_state_ptr, Q6_Vh_vsplat_R(fp16_to_bits(td_ptr++))), kv_vec_0));
        *(out_state_ptr + 1) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(*(prev_state_ptr + 1), Q6_Vh_vsplat_R(fp16_to_bits(td_ptr++))), kv_vec_1));
        *(out_state_ptr + 2) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(*(prev_state_ptr + 2), Q6_Vh_vsplat_R(fp16_to_bits(td_ptr++))), kv_vec_2));
        *(out_state_ptr + 3) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(*(prev_state_ptr + 3), Q6_Vh_vsplat_R(fp16_to_bits(td_ptr++))), kv_vec_3));
        *(out_state_ptr + 4) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(*(prev_state_ptr + 4), Q6_Vh_vsplat_R(fp16_to_bits(td_ptr++))), kv_vec_4));
        *(out_state_ptr + 5) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(*(prev_state_ptr + 5), Q6_Vh_vsplat_R(fp16_to_bits(td_ptr++))), kv_vec_5));
        *(out_state_ptr + 6) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(*(prev_state_ptr + 6), Q6_Vh_vsplat_R(fp16_to_bits(td_ptr++))), kv_vec_6));
        *(out_state_ptr + 7) = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vqf16(
          Q6_Vqf16_vmpy_VhfVhf(*(prev_state_ptr + 7), Q6_Vh_vsplat_R(fp16_to_bits(td_ptr++))), kv_vec_7));

        vtmp_0 = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp_0, Q6_Vh_vsplat_R(fp16_to_bits(r_ptr++)));
        vtmp_1 = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp_1, Q6_Vh_vsplat_R(fp16_to_bits(r_ptr++)));
        vtmp_2 = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp_2, Q6_Vh_vsplat_R(fp16_to_bits(r_ptr++)));
        vtmp_3 = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp_3, Q6_Vh_vsplat_R(fp16_to_bits(r_ptr++)));
        vtmp_4 = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp_4, Q6_Vh_vsplat_R(fp16_to_bits(r_ptr++)));
        vtmp_5 = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp_5, Q6_Vh_vsplat_R(fp16_to_bits(r_ptr++)));
        vtmp_6 = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp_6, Q6_Vh_vsplat_R(fp16_to_bits(r_ptr++)));
        vtmp_7 = Q6_Vqf16_vmpy_Vqf16Vhf(vtmp_7, Q6_Vh_vsplat_R(fp16_to_bits(r_ptr++)));
        
        vtmp_0 = Q6_Vqf16_vadd_Vqf16Vqf16(vtmp_0, vtmp_1);
        vtmp_1 = Q6_Vqf16_vadd_Vqf16Vqf16(vtmp_2, vtmp_3);
        vtmp_2 = Q6_Vqf16_vadd_Vqf16Vqf16(vtmp_4, vtmp_5);
        vtmp_3 = Q6_Vqf16_vadd_Vqf16Vqf16(vtmp_6, vtmp_7);

        vtmp_0 = Q6_Vqf16_vadd_Vqf16Vqf16(vtmp_0, vtmp_1);
        vtmp_1 = Q6_Vqf16_vadd_Vqf16Vqf16(vtmp_2, vtmp_3);

        vtmp_0 = Q6_Vqf16_vadd_Vqf16Vqf16(vtmp_0, vtmp_1);
        output_vec = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec, vtmp_0);

        out_state_ptr += 8;
        prev_state_ptr += 8;
      }
      *outptr++ = Q6_Vhf_equals_Vqf16(output_vec);
      v += head_size;
    }
  }
}

#else

template <typename T>
static void wkv6_naive(const int seq_length, const int num_heads, const int head_size,
                  T *out_0,
                  T *out_1,
                  const T *k,
                  const T *v,
                  const T *r,
                  const T *in_3,
                  const T *tf,
                  const T *td) {
  memset(out_0, 0, sizeof(T) * seq_length * num_heads * head_size);
  for (int t = 0; t < seq_length; t++) {
    T * state_in = t > 0 ? out_1 : (T*)in_3;
    for (int h = 0; h < num_heads; h++) {
      for (int i = 0; i < head_size; i++) {
        auto k_val = k[t * num_heads * head_size + h * head_size + i];
        auto r_val = r[t * num_heads * head_size + h * head_size + i];
        auto td_val = td[t * num_heads * head_size + h * head_size + i];
        auto tf_val = tf[h * head_size + i];
        for (int j = 0; j < head_size; j++) {
          auto v_val = v[t * num_heads * head_size + h * head_size + j];
          auto kv_val = k_val * v_val;
          auto prev_state_val = state_in[h * head_size * head_size + i * head_size + j];
          out_0[t * num_heads * head_size + h * head_size + j] += r_val * (kv_val * tf_val + prev_state_val);
          out_1[h * head_size * head_size + i * head_size + j] = prev_state_val * td_val + kv_val;
        }
      }
    }
  }
}

#endif

template<typename TensorType>
GraphStatus wkv6Impl(TensorType& out_0,
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

  int seq_length = k.dim(1);
  int num_heads = in_3.dim(1);
  int head_size = in_3.dim(2);

#ifdef USE_HVX
  if (k.get_dtype() == DType::Float32) {
    auto k_ptr = (float*)k.raw_data_const();
    auto v_ptr = (float*)v.raw_data_const();
    auto r_ptr = (float*)r.raw_data_const();
    auto in_3_ptr = (float*)in_3.raw_data_const();
    auto tf_ptr = (float*)tf.raw_data_const();
    auto td_ptr = (float*)td.raw_data_const();
    auto out0_ptr = (float*)out_0.raw_data();
    auto out1_ptr = (float*)out_1.raw_data();
    wkv6_hvx_f(seq_length, num_heads, head_size,
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
    wkv6_hvx_hf(seq_length, num_heads, head_size,
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
  if (k.get_dtype() == DType::Float32) {
    auto k_ptr = (float*)k.raw_data_const();
    auto v_ptr = (float*)v.raw_data_const();
    auto r_ptr = (float*)r.raw_data_const();
    auto in_3_ptr = (float*)in_3.raw_data_const();
    auto tf_ptr = (float*)tf.raw_data_const();
    auto td_ptr = (float*)td.raw_data_const();
    auto out0_ptr = (float*)out_0.raw_data();
    auto out1_ptr = (float*)out_1.raw_data();
    wkv6_naive<float>(seq_length, num_heads, head_size,
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
    wkv6_naive<__fp16>(seq_length, num_heads, head_size,
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
END_PKG_OP_DEFINITION(PKG_wkv6);