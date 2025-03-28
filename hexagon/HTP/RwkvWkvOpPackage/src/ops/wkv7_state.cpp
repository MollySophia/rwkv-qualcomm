//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_wkv7_state);


// op execute function declarations
template<typename TensorType>
GraphStatus wkv7StateImpl(TensorType& out_0,
                    const TensorType& w,
                    const TensorType& k,
                    const TensorType& v,
                    const TensorType& a,
                    const TensorType& b,
                    const TensorType& state);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((wkv7Impl<Tensor>), "wkv7")
 */
DEF_PACKAGE_OP((wkv7StateImpl<Tensor>), "wkv7_state")

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
#endif

static inline int32_t float_to_int(float scale)
{
    union { float f; int32_t i; } fp32 = { .f = scale };
    return fp32.i;
}

static void wkv7_hvx_f(const int seq_length, const int num_heads, const int head_size,
                  float *out_0,
                  const float *w,
                  const float *k,
                  const float *v,
                  const float *a,
                  const float *b,
                  const float *state_in) {
#ifdef USE_HVX
  const float *v_ptr = v;
  float tmp_buf[32];
  HVX_Vector *out_state_ptr = (HVX_Vector *)(out_0);

  for (int t = 0; t < seq_length; t++) {
    HVX_Vector *prev_state_ptr = t > 0 ? (HVX_Vector *)(out_0 + t * num_heads * head_size * head_size) : (HVX_Vector *)(state_in);

    for (int h = 0; h < num_heads; h++) {
      HVX_Vector w_vec_0 = *(HVX_Vector *)w;
      HVX_Vector w_vec_1 = *((HVX_Vector *)w + 1);
      HVX_Vector k_vec_0 = *(HVX_Vector *)k;
      HVX_Vector k_vec_1 = *((HVX_Vector *)k + 1);
      HVX_Vector a_vec_0 = *(HVX_Vector *)a;
      HVX_Vector a_vec_1 = *((HVX_Vector *)a + 1);
      HVX_Vector b_vec_0 = *(HVX_Vector *)b;
      HVX_Vector b_vec_1 = *((HVX_Vector *)b + 1);

      for (int i = 0; i < head_size; i += 4) {
        HVX_Vector v_vec_0 = Q6_V_vsplat_R(float_to_int(*v_ptr++));
        HVX_Vector v_vec_1 = Q6_V_vsplat_R(float_to_int(*v_ptr++));
        HVX_Vector v_vec_2 = Q6_V_vsplat_R(float_to_int(*v_ptr++));
        HVX_Vector v_vec_3 = Q6_V_vsplat_R(float_to_int(*v_ptr++));
        HVX_Vector kv_vec_00 = Q6_Vqf32_vmpy_VsfVsf(v_vec_0, k_vec_0);
        HVX_Vector kv_vec_01 = Q6_Vqf32_vmpy_VsfVsf(v_vec_0, k_vec_1);
        HVX_Vector kv_vec_10 = Q6_Vqf32_vmpy_VsfVsf(v_vec_1, k_vec_0);
        HVX_Vector kv_vec_11 = Q6_Vqf32_vmpy_VsfVsf(v_vec_1, k_vec_1);
        HVX_Vector kv_vec_20 = Q6_Vqf32_vmpy_VsfVsf(v_vec_2, k_vec_0);
        HVX_Vector kv_vec_21 = Q6_Vqf32_vmpy_VsfVsf(v_vec_2, k_vec_1);
        HVX_Vector kv_vec_30 = Q6_Vqf32_vmpy_VsfVsf(v_vec_3, k_vec_0);
        HVX_Vector kv_vec_31 = Q6_Vqf32_vmpy_VsfVsf(v_vec_3, k_vec_1);

        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector state_vec_0 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_1 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_2 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_3 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_4 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_5 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_6 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_7 = *((HVX_Vector *)prev_state_ptr++);

        // dot product
        HVX_Vector sa_vec_0 = Q6_Vqf32_vmpy_VsfVsf(state_vec_0, a_vec_0);
        sa_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_0, Q6_Vqf32_vmpy_VsfVsf(state_vec_1, a_vec_1));

        HVX_Vector sa_vec_1 = Q6_Vqf32_vmpy_VsfVsf(state_vec_2, a_vec_0);
        sa_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_1, Q6_Vqf32_vmpy_VsfVsf(state_vec_3, a_vec_1));

        HVX_Vector sa_vec_2 = Q6_Vqf32_vmpy_VsfVsf(state_vec_4, a_vec_0);
        sa_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_2, Q6_Vqf32_vmpy_VsfVsf(state_vec_5, a_vec_1));

        HVX_Vector sa_vec_3 = Q6_Vqf32_vmpy_VsfVsf(state_vec_6, a_vec_0);
        sa_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_3, Q6_Vqf32_vmpy_VsfVsf(state_vec_7, a_vec_1));

        for (int32_t i = 64; i >= 4; i >>= 1)
        {
          sa_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_0, Q6_V_vlalign_VVR(sa_vec_0, zero, i));
          sa_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_1, Q6_V_vlalign_VVR(sa_vec_1, zero, i));
          sa_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_2, Q6_V_vlalign_VVR(sa_vec_2, zero, i));
          sa_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_3, Q6_V_vlalign_VVR(sa_vec_3, zero, i));
        }

        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(sa_vec_0);
        sa_vec_0 = Q6_V_vsplat_R(float_to_int(tmp_buf[31]));
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(sa_vec_1);
        sa_vec_1 = Q6_V_vsplat_R(float_to_int(tmp_buf[31]));
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(sa_vec_2);
        sa_vec_2 = Q6_V_vsplat_R(float_to_int(tmp_buf[31]));
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(sa_vec_3);
        sa_vec_3 = Q6_V_vsplat_R(float_to_int(tmp_buf[31]));

        state_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(state_vec_0, w_vec_0), kv_vec_00);
        state_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_0, Q6_Vqf32_vmpy_VsfVsf(sa_vec_0, b_vec_0));
        state_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(state_vec_1, w_vec_1), kv_vec_01);
        state_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_1, Q6_Vqf32_vmpy_VsfVsf(sa_vec_0, b_vec_1));

        state_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(state_vec_2, w_vec_0), kv_vec_10);
        state_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_2, Q6_Vqf32_vmpy_VsfVsf(sa_vec_1, b_vec_0));
        state_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(state_vec_3, w_vec_1), kv_vec_11);
        state_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_3, Q6_Vqf32_vmpy_VsfVsf(sa_vec_1, b_vec_1));

        state_vec_4 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(state_vec_4, w_vec_0), kv_vec_20);
        state_vec_4 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_4, Q6_Vqf32_vmpy_VsfVsf(sa_vec_2, b_vec_0));
        state_vec_5 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(state_vec_5, w_vec_1), kv_vec_21);
        state_vec_5 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_5, Q6_Vqf32_vmpy_VsfVsf(sa_vec_2, b_vec_1));

        state_vec_6 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(state_vec_6, w_vec_0), kv_vec_30);
        state_vec_6 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_6, Q6_Vqf32_vmpy_VsfVsf(sa_vec_3, b_vec_0));
        state_vec_7 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_VsfVsf(state_vec_7, w_vec_1), kv_vec_31);
        state_vec_7 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_7, Q6_Vqf32_vmpy_VsfVsf(sa_vec_3, b_vec_1));

        *out_state_ptr++ = Q6_Vsf_equals_Vqf32(state_vec_0);
        *out_state_ptr++ = Q6_Vsf_equals_Vqf32(state_vec_1);
        *out_state_ptr++ = Q6_Vsf_equals_Vqf32(state_vec_2);
        *out_state_ptr++ = Q6_Vsf_equals_Vqf32(state_vec_3);
        *out_state_ptr++ = Q6_Vsf_equals_Vqf32(state_vec_4);
        *out_state_ptr++ = Q6_Vsf_equals_Vqf32(state_vec_5);
        *out_state_ptr++ = Q6_Vsf_equals_Vqf32(state_vec_6);
        *out_state_ptr++ = Q6_Vsf_equals_Vqf32(state_vec_7);
      }
      w += head_size;
      k += head_size;
      a += head_size;
      b += head_size;
    }
  }
#endif
}

static void wkv7_hvx_hf(const int seq_length, const int num_heads, const int head_size,
                  __fp16 *out_0,
                  const __fp16 *w,
                  const __fp16 *k,
                  const __fp16 *v,
                  const __fp16 *a,
                  const __fp16 *b,
                  const __fp16 *state_in) {
#ifdef USE_HVX
  __fp16 *v_ptr = (__fp16 *)v;

  __fp16 __attribute__((aligned(VLEN))) tmp_buf[64];
  HVX_Vector *out_state_ptr = (HVX_Vector *)(out_0);

  for (int t = 0; t < seq_length; t++) {
    HVX_Vector *prev_state_ptr = t > 0 ? (HVX_Vector *)(out_0 + t * num_heads * head_size * head_size) : (HVX_Vector *)(state_in);

    for (int h = 0; h < num_heads; h++) {
      HVX_Vector w_vec = *(HVX_Vector *)w;
      HVX_Vector k_vec = *(HVX_Vector *)k;
      HVX_Vector a_vec = *(HVX_Vector *)a;
      HVX_Vector b_vec = *(HVX_Vector *)b;
      for (int i = 0; i < head_size; i += 8) {
        HVX_Vector kv_vec_0 = Q6_Vqf16_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_1 = Q6_Vqf16_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_2 = Q6_Vqf16_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_3 = Q6_Vqf16_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_4 = Q6_Vqf16_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_5 = Q6_Vqf16_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_6 = Q6_Vqf16_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_7 = Q6_Vqf16_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));

        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector state_vec_0 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_1 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_2 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_3 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_4 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_5 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_6 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_7 = *((HVX_Vector *)prev_state_ptr++);

        // dot product
        HVX_Vector sa_vec_0 = Q6_Vqf16_vmpy_VhfVhf(state_vec_0, a_vec);
        HVX_Vector sa_vec_1 = Q6_Vqf16_vmpy_VhfVhf(state_vec_1, a_vec);
        HVX_Vector sa_vec_2 = Q6_Vqf16_vmpy_VhfVhf(state_vec_2, a_vec);
        HVX_Vector sa_vec_3 = Q6_Vqf16_vmpy_VhfVhf(state_vec_3, a_vec);
        HVX_Vector sa_vec_4 = Q6_Vqf16_vmpy_VhfVhf(state_vec_4, a_vec);
        HVX_Vector sa_vec_5 = Q6_Vqf16_vmpy_VhfVhf(state_vec_5, a_vec);
        HVX_Vector sa_vec_6 = Q6_Vqf16_vmpy_VhfVhf(state_vec_6, a_vec);
        HVX_Vector sa_vec_7 = Q6_Vqf16_vmpy_VhfVhf(state_vec_7, a_vec);
        for (int32_t n = 64; n >= 2; n >>= 1) {
          sa_vec_0 = Q6_Vqf16_vadd_Vqf16Vqf16(sa_vec_0, Q6_V_vlalign_VVR(sa_vec_0, zero, n));
          sa_vec_1 = Q6_Vqf16_vadd_Vqf16Vqf16(sa_vec_1, Q6_V_vlalign_VVR(sa_vec_1, zero, n));
          sa_vec_2 = Q6_Vqf16_vadd_Vqf16Vqf16(sa_vec_2, Q6_V_vlalign_VVR(sa_vec_2, zero, n));
          sa_vec_3 = Q6_Vqf16_vadd_Vqf16Vqf16(sa_vec_3, Q6_V_vlalign_VVR(sa_vec_3, zero, n));
          sa_vec_4 = Q6_Vqf16_vadd_Vqf16Vqf16(sa_vec_4, Q6_V_vlalign_VVR(sa_vec_4, zero, n));
          sa_vec_5 = Q6_Vqf16_vadd_Vqf16Vqf16(sa_vec_5, Q6_V_vlalign_VVR(sa_vec_5, zero, n));
          sa_vec_6 = Q6_Vqf16_vadd_Vqf16Vqf16(sa_vec_6, Q6_V_vlalign_VVR(sa_vec_6, zero, n));
          sa_vec_7 = Q6_Vqf16_vadd_Vqf16Vqf16(sa_vec_7, Q6_V_vlalign_VVR(sa_vec_7, zero, n));
        }
        *(HVX_Vector *)tmp_buf = Q6_Vhf_equals_Vqf16(sa_vec_0);
        sa_vec_0 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_buf[63]));
        *(HVX_Vector *)tmp_buf = Q6_Vhf_equals_Vqf16(sa_vec_1);
        sa_vec_1 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_buf[63]));
        *(HVX_Vector *)tmp_buf = Q6_Vhf_equals_Vqf16(sa_vec_2);
        sa_vec_2 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_buf[63]));
        *(HVX_Vector *)tmp_buf = Q6_Vhf_equals_Vqf16(sa_vec_3);
        sa_vec_3 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_buf[63]));
        *(HVX_Vector *)tmp_buf = Q6_Vhf_equals_Vqf16(sa_vec_4);
        sa_vec_4 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_buf[63]));
        *(HVX_Vector *)tmp_buf = Q6_Vhf_equals_Vqf16(sa_vec_5);
        sa_vec_5 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_buf[63]));
        *(HVX_Vector *)tmp_buf = Q6_Vhf_equals_Vqf16(sa_vec_6);
        sa_vec_6 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_buf[63]));
        *(HVX_Vector *)tmp_buf = Q6_Vhf_equals_Vqf16(sa_vec_7);
        sa_vec_7 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_buf[63]));

        state_vec_0 = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(state_vec_0, w_vec), kv_vec_0);
        state_vec_0 = Q6_Vqf16_vadd_Vqf16Vqf16(state_vec_0, Q6_Vqf16_vmpy_VhfVhf(sa_vec_0, b_vec));

        state_vec_1 = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(state_vec_1, w_vec), kv_vec_1);
        state_vec_1 = Q6_Vqf16_vadd_Vqf16Vqf16(state_vec_1, Q6_Vqf16_vmpy_VhfVhf(sa_vec_1, b_vec));

        state_vec_2 = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(state_vec_2, w_vec), kv_vec_2);
        state_vec_2 = Q6_Vqf16_vadd_Vqf16Vqf16(state_vec_2, Q6_Vqf16_vmpy_VhfVhf(sa_vec_2, b_vec));

        state_vec_3 = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(state_vec_3, w_vec), kv_vec_3);
        state_vec_3 = Q6_Vqf16_vadd_Vqf16Vqf16(state_vec_3, Q6_Vqf16_vmpy_VhfVhf(sa_vec_3, b_vec));

        state_vec_4 = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(state_vec_4, w_vec), kv_vec_4);
        state_vec_4 = Q6_Vqf16_vadd_Vqf16Vqf16(state_vec_4, Q6_Vqf16_vmpy_VhfVhf(sa_vec_4, b_vec));

        state_vec_5 = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(state_vec_5, w_vec), kv_vec_5);
        state_vec_5 = Q6_Vqf16_vadd_Vqf16Vqf16(state_vec_5, Q6_Vqf16_vmpy_VhfVhf(sa_vec_5, b_vec));

        state_vec_6 = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(state_vec_6, w_vec), kv_vec_6);
        state_vec_6 = Q6_Vqf16_vadd_Vqf16Vqf16(state_vec_6, Q6_Vqf16_vmpy_VhfVhf(sa_vec_6, b_vec));

        state_vec_7 = Q6_Vqf16_vadd_Vqf16Vqf16(Q6_Vqf16_vmpy_VhfVhf(state_vec_7, w_vec), kv_vec_7);
        state_vec_7 = Q6_Vqf16_vadd_Vqf16Vqf16(state_vec_7, Q6_Vqf16_vmpy_VhfVhf(sa_vec_7, b_vec));

        *out_state_ptr++ = Q6_Vhf_equals_Vqf16(state_vec_0);
        *out_state_ptr++ = Q6_Vhf_equals_Vqf16(state_vec_1);
        *out_state_ptr++ = Q6_Vhf_equals_Vqf16(state_vec_2);
        *out_state_ptr++ = Q6_Vhf_equals_Vqf16(state_vec_3);
        *out_state_ptr++ = Q6_Vhf_equals_Vqf16(state_vec_4);
        *out_state_ptr++ = Q6_Vhf_equals_Vqf16(state_vec_5);
        *out_state_ptr++ = Q6_Vhf_equals_Vqf16(state_vec_6);
        *out_state_ptr++ = Q6_Vhf_equals_Vqf16(state_vec_7);
      }
      w += head_size;
      k += head_size;
      a += head_size;
      b += head_size;
    }
  }
#endif
}


template <typename T>
static void wkv7_naive(const int seq_length, const int num_heads, const int head_size,
                  T *out_0,
                  const T *w,
                  const T *k,
                  const T *v,
                  const T *a,
                  const T *b,
                  const T *state) {

  // for (int t = 0; t < seq_length; t++) {
  //   T * state_in = t > 0 ? out_1 : (T*)state;
  //   for (int h = 0; h < num_heads; h++) {
  //     for (int i = 0; i < head_size; i++) {
  //       auto v_val = v[t * num_heads * head_size + h * head_size + i];

  //       T sa = 0, result = 0;
  //       for (int j = 0; j < head_size; j++) {
  //         sa += a[t * num_heads * head_size + h * head_size + j] * state_in[h * head_size * head_size + i * head_size + j];
  //       }

  //       for (int j = 0; j < head_size; j++) {
  //         auto r_val = r[t * num_heads * head_size + h * head_size + j];
  //         auto w_val = w[t * num_heads * head_size + h * head_size + j];
  //         auto k_val = k[t * num_heads * head_size + h * head_size + j];
  //         auto b_val = b[t * num_heads * head_size + h * head_size + j];
  //         auto kv_val = k_val * v_val;

  //         auto state_val = state_in[h * head_size * head_size + i * head_size + j];
  //         state_val = state_val * w_val + kv_val + sa * b_val;
  //         out_1[h * head_size * head_size + i * head_size + j] = state_val;
  //         result += r_val * state_val;
  //       }
  //       out_0[t * num_heads * head_size + h * head_size + i] = result;
  //     }
  //   }
  // }
}

#endif

template<typename TensorType>
GraphStatus wkv7StateImpl(TensorType& out_0,
                    const TensorType& w,
                    const TensorType& k,
                    const TensorType& v,
                    const TensorType& a,
                    const TensorType& b,
                    const TensorType& state)

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

  int num_heads = state.dim(1);
  int head_size = state.dim(2);
  int seq_length = k.dim(0);

#ifdef USE_HVX
// #if 0
  if (state.get_dtype() == DType::Float32) {
    auto w_ptr = (float*)w.raw_data_const();
    auto k_ptr = (float*)k.raw_data_const();
    auto v_ptr = (float*)v.raw_data_const();
    auto a_ptr = (float*)a.raw_data_const();
    auto b_ptr = (float*)b.raw_data_const();
    auto state_ptr = (float*)state.raw_data_const();
    auto out0_ptr = (float*)out_0.raw_data();
    wkv7_hvx_f(seq_length, num_heads, head_size,
                      out0_ptr,
                      w_ptr,
                      k_ptr,
                      v_ptr,
                      a_ptr,
                      b_ptr,
                      state_ptr);
  } else if (state.get_dtype() == DType::Float16) {
    auto w_ptr = (__fp16*)w.raw_data_const();
    auto k_ptr = (__fp16*)k.raw_data_const();
    auto v_ptr = (__fp16*)v.raw_data_const();
    auto a_ptr = (__fp16*)a.raw_data_const();
    auto b_ptr = (__fp16*)b.raw_data_const();
    auto state_ptr = (__fp16*)state.raw_data_const();
    auto out0_ptr = (__fp16*)out_0.raw_data();
    wkv7_hvx_hf(seq_length, num_heads, head_size,
                      out0_ptr,
                      w_ptr,
                      k_ptr,
                      v_ptr,
                      a_ptr,
                      b_ptr,
                      state_ptr);
  }
#else
  if (state.get_dtype() == DType::Float32) {
    auto w_ptr = (float*)w.raw_data_const();
    auto k_ptr = (float*)k.raw_data_const();
    auto v_ptr = (float*)v.raw_data_const();
    auto a_ptr = (float*)a.raw_data_const();
    auto b_ptr = (float*)b.raw_data_const();
    auto state_ptr = (float*)state.raw_data_const();
    auto out0_ptr = (float*)out_0.raw_data();
    wkv7_naive<float>(seq_length, num_heads, head_size,
                      out0_ptr,
                      w_ptr,
                      k_ptr,
                      v_ptr,
                      a_ptr,
                      b_ptr,
                      state_ptr);
  } else if (state.get_dtype() == DType::Float16) {
    auto w_ptr = (__fp16*)w.raw_data_const();
    auto k_ptr = (__fp16*)k.raw_data_const();
    auto v_ptr = (__fp16*)v.raw_data_const();
    auto a_ptr = (__fp16*)a.raw_data_const();
    auto b_ptr = (__fp16*)b.raw_data_const();
    auto state_ptr = (__fp16*)state.raw_data_const();
    auto out0_ptr = (__fp16*)out_0.raw_data();
    wkv7_naive<__fp16>(seq_length, num_heads, head_size,
                      out0_ptr,
                      w_ptr,
                      k_ptr,
                      v_ptr,
                      a_ptr,
                      b_ptr,
                      state_ptr);
  }
#endif
  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_wkv7_state);