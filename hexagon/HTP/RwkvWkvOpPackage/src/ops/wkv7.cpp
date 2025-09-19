//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_wkv7);


// op execute function declarations
template<typename TensorType>
GraphStatus wkv7FloatImpl(TensorType& out_0,
                    const TensorType& r,
                    const TensorType& w,
                    const TensorType& k,
                    const TensorType& v,
                    const TensorType& a,
                    const TensorType& b,
                    const TensorType& state);

template<typename TensorType, typename StateType>
GraphStatus wkv7Float16Impl(StateType& out_0,
                    const TensorType& r,
                    const TensorType& w,
                    const TensorType& k,
                    const TensorType& v,
                    const TensorType& a,
                    const TensorType& b,
                    const StateType& state);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((wkv7Impl<Tensor>), "wkv7")
 */
// DEF_PACKAGE_OP((wkv7StateImpl<Tensor>), "wkv7_state")

DEF_PACKAGE_OPTIMIZATION(
  CLEANUP_GRAPH + 50,
  Op("wkv7", "r", "w", "k", "v", "a", "b", "state_in"),
  IS_FLOAT16("state_in"),
  Op("wkv7.fp16", "r", "w", "k", "v", "a", "b", "state_in")
)

DEF_PACKAGE_OPTIMIZATION(
  CLEANUP_GRAPH + 50,
  Op("wkv7", "r", "w", "k", "v", "a", "b", "state_in"),
  IS_QUINT16("w"),
  Op("wkv7.uint16", "r", "w", "k", "v", "a", "b", "state_in")
)

DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
  CLEANUP_GRAPH + 51, relaxed_precision_flag,
  Op("wkv7", "r", "w", "k", "v", "a", "b", "state_in"),
  OK,
  MAKE_OP_FP16_AND_INSERT_CAST(Op("wkv7.fp16", CAST_TO_FP16("r"), CAST_TO_FP16("w"), CAST_TO_FP16("k"), CAST_TO_FP16("v"), CAST_TO_FP16("a"), CAST_TO_FP16("b"), CAST_TO_FP16("state_in")))
)

// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7.fp16", "r", "w", "k", "v", "a", "b", "state_in"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7.fp16",
//       TYPICAL_SLICE("r", "I"),
//       TYPICAL_SLICE("w", "I"),
//       TYPICAL_SLICE("k", "I"),
//       TYPICAL_SLICE("v", "I"),
//       TYPICAL_SLICE("a", "I"),
//       TYPICAL_SLICE("b", "I"),
//       TYPICAL_SLICE("state_in", "I")
//     )
//   )
// )

// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7", "r", "w", "k", "v", "a", "b", "state_in"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7",
//       TYPICAL_SLICE("r", "I"),
//       TYPICAL_SLICE("w", "I"),
//       TYPICAL_SLICE("k", "I"),
//       TYPICAL_SLICE("v", "I"),
//       TYPICAL_SLICE("a", "I"),
//       TYPICAL_SLICE("b", "I"),
//       TYPICAL_SLICE("state_in", "I")
//     )
//   )
// )

// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7.uint16", "r", "w", "k", "v", "a", "b", "state_in"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7.uint16",
//       TYPICAL_SLICE("r", "I"),
//       TYPICAL_SLICE("w", "I"),
//       TYPICAL_SLICE("k", "I"),
//       TYPICAL_SLICE("v", "I"),
//       TYPICAL_SLICE("a", "I"),
//       TYPICAL_SLICE("b", "I"),
//       TYPICAL_SLICE("state_in", "I")
//     )
//   )
// )

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7", "r", "w", "k", "v", "a", "b", "state_in"),
  OK,
  Op("wkv7.flat",
    WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "r")),
    WITH_SAME_OUTPUT("w", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "w")),
    WITH_SAME_OUTPUT("k", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "k")),
    WITH_SAME_OUTPUT("v", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "v")),
    WITH_SAME_OUTPUT("a", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "a")),
    WITH_SAME_OUTPUT("b", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "b")),
    WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "state_in"))
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7.fp16", "r", "w", "k", "v", "a", "b", "state_in"),
  OK,
  Op("wkv7.fp16.flat",
    WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "r")),
    WITH_SAME_OUTPUT("w", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "w")),
    WITH_SAME_OUTPUT("k", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "k")),
    WITH_SAME_OUTPUT("v", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "v")),
    WITH_SAME_OUTPUT("a", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "a")),
    WITH_SAME_OUTPUT("b", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "b")),
    WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "state_in"))
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7.uint16", "r", "w", "k", "v", "a", "b", "state_in"),
  OK,
  WITH_OUTPUT_TYPE(DType::QUInt16, ZERO_OFFSET_OF("*"), STEPSIZE_OF("*"),
    Op("wkv7.uint16.flat.dequant",
      WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("r"), STEPSIZE_OF("r"),
        WITH_SIZE("r", Op(FROM_DEFAULT_PACKAGE("Dequantize"), WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "r"))))
      ),
      WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("w"), STEPSIZE_OF("w"),
        WITH_SIZE("w", Op(FROM_DEFAULT_PACKAGE("Dequantize"), WITH_SAME_OUTPUT("w", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "w"))))
      ),
      WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("k"), STEPSIZE_OF("k"),
        WITH_SIZE("k", Op(FROM_DEFAULT_PACKAGE("Dequantize"), WITH_SAME_OUTPUT("k", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "k"))))
      ),
      WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("v"), STEPSIZE_OF("v"),
        WITH_SIZE("v", Op(FROM_DEFAULT_PACKAGE("Dequantize"), WITH_SAME_OUTPUT("v", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "v"))))
      ),
      WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("a"), STEPSIZE_OF("a"),
        WITH_SIZE("a", Op(FROM_DEFAULT_PACKAGE("Dequantize"), WITH_SAME_OUTPUT("a", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "a"))))
      ),
      WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("b"), STEPSIZE_OF("b"),
        WITH_SIZE("b", Op(FROM_DEFAULT_PACKAGE("Dequantize"), WITH_SAME_OUTPUT("b", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "b"))))
      ),
      WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "state_in"))
    )
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7.fp16.flat", "r", "w", "k", "v", "a", "b", "state_in"),
  OK,
  // Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7.fp16.flat.tcm",
      WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "r")),
      WITH_SAME_OUTPUT("w", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "w")),
      WITH_SAME_OUTPUT("k", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "k")),
      WITH_SAME_OUTPUT("v", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "v")),
      WITH_SAME_OUTPUT("a", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "a")),
      WITH_SAME_OUTPUT("b", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "b")),
      WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "state_in"))
    )
  // )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7.flat", "r", "w", "k", "v", "a", "b", "state_in"),
  OK,
  // Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7.flat.tcm",
      WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "r")),
      WITH_SAME_OUTPUT("w", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "w")),
      WITH_SAME_OUTPUT("k", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "k")),
      WITH_SAME_OUTPUT("v", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "v")),
      WITH_SAME_OUTPUT("a", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "a")),
      WITH_SAME_OUTPUT("b", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "b")),
      WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "state_in"))
    )
  // )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7.uint16.flat.dequant", "r", "w", "k", "v", "a", "b", "state_in"),
  OK,
  // Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7.uint16.flat.dequant.tcm",
      WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "r")),
      WITH_SAME_OUTPUT("w", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "w")),
      WITH_SAME_OUTPUT("k", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "k")),
      WITH_SAME_OUTPUT("v", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "v")),
      WITH_SAME_OUTPUT("a", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "a")),
      WITH_SAME_OUTPUT("b", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "b")),
      WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "state_in"))
    )
  // )
)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7Float16Impl<PlainFloat16Tensor, PlainFloat16Tensor>), "wkv7.fp16.flat", FAST, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7Float16Impl<PlainFloat16Tensor_TCM, PlainFloat16Tensor_TCM>), "wkv7.fp16.flat.tcm", FAST, Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7FloatImpl<PlainFloatTensor>), "wkv7.flat", FAST, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7FloatImpl<PlainFloatTensor_TCM>), "wkv7.flat.tcm", FAST, Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7Float16Impl<PlainFloat16Tensor_TCM, QuantUint16Tensor_TCM>), "wkv7.uint16.flat.dequant.tcm", FAST, Flags::RESOURCE_HVX)

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

template<typename TensorType>
GraphStatus wkv7FloatImpl(TensorType& out_0,
                    const TensorType& r,
                    const TensorType& w,
                    const TensorType& k,
                    const TensorType& v,
                    const TensorType& a,
                    const TensorType& b,
                    const TensorType& state) {
// TODO: fix this (if it is needed in the future)
#ifdef USE_HVX
  int num_heads = state.dim(1);
  int head_size = state.dim(2);
  int seq_length = k.dim(0);
  // auto r_ptr = (float*)r.raw_data_const();
  auto w_ptr = (float*)w.raw_data_const();
  auto k_ptr = (float*)k.raw_data_const();
  auto v_ptr = (float*)v.raw_data_const();
  auto a_ptr = (float*)a.raw_data_const();
  auto b_ptr = (float*)b.raw_data_const();
  auto state_ptr = (float*)state.raw_data_const();
  auto out0_ptr = (float*)out_0.raw_data();
  float tmp_buf[32];
  HVX_Vector *out_state_ptr = (HVX_Vector *)(out0_ptr);

  for (int t = 0; t < seq_length; t++) {
    HVX_Vector *prev_state_ptr = t > 0 ? (HVX_Vector *)(out0_ptr + (t - 1) * num_heads * head_size * head_size) : (HVX_Vector *)(state_ptr);

    for (int h = 0; h < num_heads; h++) {
      HVX_Vector w_vec_0 = *(HVX_Vector *)w_ptr;
      HVX_Vector w_vec_1 = *((HVX_Vector *)w_ptr + 1);
      HVX_Vector k_vec_0 = *(HVX_Vector *)k_ptr;
      HVX_Vector k_vec_1 = *((HVX_Vector *)k_ptr + 1);
      HVX_Vector a_vec_0 = *(HVX_Vector *)a_ptr;
      HVX_Vector a_vec_1 = *((HVX_Vector *)a_ptr + 1);
      HVX_Vector b_vec_0 = *(HVX_Vector *)b_ptr;
      HVX_Vector b_vec_1 = *((HVX_Vector *)b_ptr + 1);

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
      w_ptr += head_size;
      k_ptr += head_size;
      a_ptr += head_size;
      b_ptr += head_size;
    }
  }
#endif
  return GraphStatus::Success;
}

template<typename TensorType, typename StateType>
GraphStatus wkv7Float16Impl(StateType& out_0,
                    const TensorType& r,
                    const TensorType& w,
                    const TensorType& k,
                    const TensorType& v,
                    const TensorType& a,
                    const TensorType& b,
                    const StateType& state) {
#ifdef USE_HVX
  int num_heads = state.dim(1);
  int head_size = state.dim(2);
  int seq_length = k.dim(1);
  __fp16* r_ptr = (__fp16*)r.raw_data_const();
  __fp16* w_ptr = (__fp16*)w.raw_data_const();
  __fp16* k_ptr = (__fp16*)k.raw_data_const();
  __fp16* v_ptr = (__fp16*)v.raw_data_const();
  __fp16* a_ptr = (__fp16*)a.raw_data_const();
  __fp16* b_ptr = (__fp16*)b.raw_data_const();
  __fp16* state_ptr = (__fp16*)state.raw_data_const();
  __fp16* out0_ptr = (__fp16*)out_0.raw_data();

  __fp16 __attribute__((aligned(VLEN))) tmp_buf_fp16[64];
  // HVX_Vector *out_state_ptr = (HVX_Vector *)(out0_ptr);
  float tmp_buf[32];

  for (int t = 0; t < seq_length; t++) {
    // HVX_Vector *prev_state_ptr = t > 0 ? (HVX_Vector *)(out0_ptr + (t - 1) * num_heads * head_size * head_size) : (HVX_Vector *)(state_ptr);

    for (int h = 0; h < num_heads; h++) {
      HVX_Vector *prev_state_ptr = t > 0 ? (HVX_Vector *)(out0_ptr + h * (seq_length + head_size) * head_size + seq_length * head_size)
        : (HVX_Vector *)(state_ptr + h * head_size * head_size);
      HVX_Vector *out_state_ptr = (HVX_Vector *)(out0_ptr + h * (seq_length + head_size) * head_size + seq_length * head_size);
      __fp16 *out_x_ptr = out0_ptr + h * (seq_length + head_size) * head_size + t * head_size;

      HVX_Vector r_vec = *(HVX_Vector *)r_ptr;
      HVX_Vector w_vec = *(HVX_Vector *)w_ptr;
      HVX_Vector k_vec = *(HVX_Vector *)k_ptr;
      HVX_Vector a_vec = *(HVX_Vector *)a_ptr;
      HVX_Vector b_vec = *(HVX_Vector *)b_ptr;
      for (int i = 0; i < head_size; i += 4) {
        // Wqf32 kv
        HVX_VectorPair kv_vecpair_0 = Q6_Wqf32_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_00 = Q6_V_hi_W(kv_vecpair_0);
        HVX_Vector kv_vec_01 = Q6_V_lo_W(kv_vecpair_0);
        HVX_VectorPair kv_vecpair_1 = Q6_Wqf32_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_10 = Q6_V_hi_W(kv_vecpair_1);
        HVX_Vector kv_vec_11 = Q6_V_lo_W(kv_vecpair_1);
        HVX_VectorPair kv_vecpair_2 = Q6_Wqf32_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_20 = Q6_V_hi_W(kv_vecpair_2);
        HVX_Vector kv_vec_21 = Q6_V_lo_W(kv_vecpair_2);
        HVX_VectorPair kv_vecpair_3 = Q6_Wqf32_vmpy_VhfVhf(k_vec, Q6_Vh_vsplat_R(fp16_to_bits(v_ptr++)));
        HVX_Vector kv_vec_30 = Q6_V_hi_W(kv_vecpair_3);
        HVX_Vector kv_vec_31 = Q6_V_lo_W(kv_vecpair_3);
        // Vhf state in
        HVX_Vector zero = Q6_V_vzero();
        HVX_Vector state_vec_0 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_1 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_2 = *((HVX_Vector *)prev_state_ptr++);
        HVX_Vector state_vec_3 = *((HVX_Vector *)prev_state_ptr++);

        // dot product
        // Wqf32 sa
        HVX_VectorPair sa_vecpair_0 = Q6_Wqf32_vmpy_VhfVhf(state_vec_0, a_vec);
        HVX_Vector sa_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(sa_vecpair_0), Q6_V_lo_W(sa_vecpair_0));
        HVX_VectorPair sa_vecpair_1 = Q6_Wqf32_vmpy_VhfVhf(state_vec_1, a_vec);
        HVX_Vector sa_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(sa_vecpair_1), Q6_V_lo_W(sa_vecpair_1));
        HVX_VectorPair sa_vecpair_2 = Q6_Wqf32_vmpy_VhfVhf(state_vec_2, a_vec);
        HVX_Vector sa_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(sa_vecpair_2), Q6_V_lo_W(sa_vecpair_2));
        HVX_VectorPair sa_vecpair_3 = Q6_Wqf32_vmpy_VhfVhf(state_vec_3, a_vec);
        HVX_Vector sa_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(sa_vecpair_3), Q6_V_lo_W(sa_vecpair_3));

        for (int32_t i = 64; i >= 4; i >>= 1)
        {
          sa_vec_0 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_0, Q6_V_vlalign_VVR(sa_vec_0, zero, i));
          sa_vec_1 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_1, Q6_V_vlalign_VVR(sa_vec_1, zero, i));
          sa_vec_2 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_2, Q6_V_vlalign_VVR(sa_vec_2, zero, i));
          sa_vec_3 = Q6_Vqf32_vadd_Vqf32Vqf32(sa_vec_3, Q6_V_vlalign_VVR(sa_vec_3, zero, i));
        }

        // Vhf sa
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(sa_vec_0);
        __fp16 tmp_val_fp16 = (__fp16)tmp_buf[31];
        sa_vec_0 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_val_fp16));
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(sa_vec_1);
        tmp_val_fp16 = (__fp16)tmp_buf[31];
        sa_vec_1 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_val_fp16));
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(sa_vec_2);
        tmp_val_fp16 = (__fp16)tmp_buf[31];
        sa_vec_2 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_val_fp16));
        *(HVX_Vector *)tmp_buf = Q6_Vsf_equals_Vqf32(sa_vec_3);
        tmp_val_fp16 = (__fp16)tmp_buf[31];
        sa_vec_3 = Q6_Vh_vsplat_R(fp16_to_bits(&tmp_val_fp16));

        sa_vecpair_0 = Q6_Wqf32_vmpy_VhfVhf(sa_vec_0, b_vec);
        sa_vecpair_1 = Q6_Wqf32_vmpy_VhfVhf(sa_vec_1, b_vec);
        sa_vecpair_2 = Q6_Wqf32_vmpy_VhfVhf(sa_vec_2, b_vec);
        sa_vecpair_3 = Q6_Wqf32_vmpy_VhfVhf(sa_vec_3, b_vec);

        HVX_VectorPair state_vecpair_0 = Q6_Wqf32_vmpy_VhfVhf(state_vec_0, w_vec);
        HVX_Vector state_vec_00 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(state_vecpair_0), kv_vec_00);
        state_vec_00 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_00, Q6_V_hi_W(sa_vecpair_0));
        HVX_Vector state_vec_01 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(state_vecpair_0), kv_vec_01);
        state_vec_01 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_01, Q6_V_lo_W(sa_vecpair_0));

        HVX_VectorPair state_vecpair_1 = Q6_Wqf32_vmpy_VhfVhf(state_vec_1, w_vec);
        HVX_Vector state_vec_10 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(state_vecpair_1), kv_vec_10);
        state_vec_10 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_10, Q6_V_hi_W(sa_vecpair_1));
        HVX_Vector state_vec_11 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(state_vecpair_1), kv_vec_11);
        state_vec_11 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_11, Q6_V_lo_W(sa_vecpair_1));

        HVX_VectorPair state_vecpair_2 = Q6_Wqf32_vmpy_VhfVhf(state_vec_2, w_vec);
        HVX_Vector state_vec_20 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(state_vecpair_2), kv_vec_20);
        state_vec_20 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_20, Q6_V_hi_W(sa_vecpair_2));
        HVX_Vector state_vec_21 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(state_vecpair_2), kv_vec_21);
        state_vec_21 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_21, Q6_V_lo_W(sa_vecpair_2));

        HVX_VectorPair state_vecpair_3 = Q6_Wqf32_vmpy_VhfVhf(state_vec_3, w_vec);
        HVX_Vector state_vec_30 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_hi_W(state_vecpair_3), kv_vec_30);
        state_vec_30 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_30, Q6_V_hi_W(sa_vecpair_3));
        HVX_Vector state_vec_31 = Q6_Vqf32_vadd_Vqf32Vqf32(Q6_V_lo_W(state_vecpair_3), kv_vec_31);
        state_vec_31 = Q6_Vqf32_vadd_Vqf32Vqf32(state_vec_31, Q6_V_lo_W(sa_vecpair_3));

        state_vec_0 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(state_vec_00, state_vec_01));
        state_vec_1 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(state_vec_10, state_vec_11));
        state_vec_2 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(state_vec_20, state_vec_21));
        state_vec_3 = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(state_vec_30, state_vec_31));

        *out_state_ptr++ = state_vec_0;
        *out_state_ptr++ = state_vec_1;
        *out_state_ptr++ = state_vec_2;
        *out_state_ptr++ = state_vec_3;
        // *out_state_ptr++ = state_vec_4;
        // *out_state_ptr++ = state_vec_5;
        // *out_state_ptr++ = state_vec_6;
        // *out_state_ptr++ = state_vec_7;

        // r @ state
        HVX_Vector output_vec_0 = Q6_Vqf16_vmpy_VhfVhf(state_vec_0, r_vec);
        HVX_Vector output_vec_1 = Q6_Vqf16_vmpy_VhfVhf(state_vec_1, r_vec);
        HVX_Vector output_vec_2 = Q6_Vqf16_vmpy_VhfVhf(state_vec_2, r_vec);
        HVX_Vector output_vec_3 = Q6_Vqf16_vmpy_VhfVhf(state_vec_3, r_vec);
        // HVX_Vector output_vec_4 = Q6_Vqf16_vmpy_VhfVhf(state_vec_4, r_vec);
        // HVX_Vector output_vec_5 = Q6_Vqf16_vmpy_VhfVhf(state_vec_5, r_vec);
        // HVX_Vector output_vec_6 = Q6_Vqf16_vmpy_VhfVhf(state_vec_6, r_vec);
        // HVX_Vector output_vec_7 = Q6_Vqf16_vmpy_VhfVhf(state_vec_7, r_vec);

        zero = Q6_V_vzero();
        for (int32_t n = 64; n >= 2; n >>= 1) {
          output_vec_0 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_0, Q6_V_vlalign_VVR(output_vec_0, zero, n));
          output_vec_1 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_1, Q6_V_vlalign_VVR(output_vec_1, zero, n));
          output_vec_2 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_2, Q6_V_vlalign_VVR(output_vec_2, zero, n));
          output_vec_3 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_3, Q6_V_vlalign_VVR(output_vec_3, zero, n));
          // output_vec_4 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_4, Q6_V_vlalign_VVR(output_vec_4, zero, n));
          // output_vec_5 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_5, Q6_V_vlalign_VVR(output_vec_5, zero, n));
          // output_vec_6 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_6, Q6_V_vlalign_VVR(output_vec_6, zero, n));
          // output_vec_7 = Q6_Vqf16_vadd_Vqf16Vqf16(output_vec_7, Q6_V_vlalign_VVR(output_vec_7, zero, n));
        }
        output_vec_0 = Q6_Vhf_equals_Vqf16(output_vec_0);
        output_vec_1 = Q6_Vhf_equals_Vqf16(output_vec_1);
        output_vec_2 = Q6_Vhf_equals_Vqf16(output_vec_2);
        output_vec_3 = Q6_Vhf_equals_Vqf16(output_vec_3);
        // output_vec_4 = Q6_Vhf_equals_Vqf16(output_vec_4);
        // output_vec_5 = Q6_Vhf_equals_Vqf16(output_vec_5);
        // output_vec_6 = Q6_Vhf_equals_Vqf16(output_vec_6);
        // output_vec_7 = Q6_Vhf_equals_Vqf16(output_vec_7);

        *(HVX_Vector *)tmp_buf_fp16 = output_vec_0;
        *out_x_ptr++ = tmp_buf_fp16[63];
        *(HVX_Vector *)tmp_buf_fp16 = output_vec_1;
        *out_x_ptr++ = tmp_buf_fp16[63];
        *(HVX_Vector *)tmp_buf_fp16 = output_vec_2;
        *out_x_ptr++ = tmp_buf_fp16[63];
        *(HVX_Vector *)tmp_buf_fp16 = output_vec_3;
        *out_x_ptr++ = tmp_buf_fp16[63];
        // *(HVX_Vector *)tmp_buf_fp16 = output_vec_4;
        // *out_x_ptr++ = tmp_buf_fp16[63];
        // *(HVX_Vector *)tmp_buf_fp16 = output_vec_5;
        // *out_x_ptr++ = tmp_buf_fp16[63];
        // *(HVX_Vector *)tmp_buf_fp16 = output_vec_6;
        // *out_x_ptr++ = tmp_buf_fp16[63];
        // *(HVX_Vector *)tmp_buf_fp16 = output_vec_7;
        // *out_x_ptr++ = tmp_buf_fp16[63];
      }
      r_ptr += head_size;
      w_ptr += head_size;
      k_ptr += head_size;
      a_ptr += head_size;
      b_ptr += head_size;
    }
  }
#endif
  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_wkv7);