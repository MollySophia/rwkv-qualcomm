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
template<typename TensorType, typename StateType>
GraphStatus wkv7OutputFloat16Impl(TensorType& out_0,
                    const TensorType& r,
                    const StateType& state_in);

template<typename TensorType>
GraphStatus wkv7OutputFloatImpl(TensorType& out_0,
                    const TensorType& r,
                    const TensorType& state_in);

DEF_PACKAGE_OPTIMIZATION(
  CLEANUP_GRAPH + 50,
  Op("wkv7_output", "r", "state_in"),
  IS_FLOAT16("state_in"),
  Op("wkv7_output.fp16", "r", "state_in")
)

DEF_PACKAGE_OPTIMIZATION(
  CLEANUP_GRAPH + 50,
  Op("wkv7_output", "r", "state_in"),
  IS_QUINT16("r"),
  Op("wkv7_output.uint16", "r", "state_in")
)

// cast
DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
  CLEANUP_GRAPH + 51, relaxed_precision_flag,
  Op("wkv7_output", "r", "state_in"),
  OK,
  MAKE_OP_FP16_AND_INSERT_CAST(Op("wkv7_output.fp16", CAST_TO_FP16("r"), CAST_TO_FP16("state_in")))
)

// split
DEF_PACKAGE_OPTIMIZATION(
  TILING + 100,
  Op("wkv7_output.fp16", "r", "state_in"),
  GT(DIM_HEIGHT("*"), 2),
  AUTOSPLIT(1, "I", 2,
    Op("wkv7_output.fp16",
      TYPICAL_SLICE("r", "I"),
      TYPICAL_SLICE("state_in", "I")
    )
  )
)

DEF_PACKAGE_OPTIMIZATION(
  TILING + 100,
  Op("wkv7_output", "r", "state_in"),
  GT(DIM_HEIGHT("*"), 2),
  AUTOSPLIT(1, "I", 2,
    Op("wkv7_output",
      TYPICAL_SLICE("r", "I"),
      TYPICAL_SLICE("state_in", "I")
    )
  )
)

DEF_PACKAGE_OPTIMIZATION(
  TILING + 100,
  Op("wkv7_output.uint16", "r", "state_in"),
  GT(DIM_HEIGHT("*"), 2),
  AUTOSPLIT(1, "I", 2,
    Op("wkv7_output.uint16",
      TYPICAL_SLICE("r", "I"),
      TYPICAL_SLICE("state_in", "I")
    )
  )
)

// forceformat
DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7_output.fp16", "r", "state_in"),
  OK,
  Op("wkv7_output.fp16.flat",
    WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "r")),
    WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "state_in"))
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7_output", "r", "state_in"),
  OK,
  Op("wkv7_output.flat",
    WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "r")),
    WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "state_in"))
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7_output.uint16", "r", "state_in"),
  OK,
  WITH_SAME_OUTPUT("*",
    WITH_OUTPUT_TYPE(DType::QUInt16, ZERO_OFFSET_OF("*"), STEPSIZE_OF("*"),
      Op(FROM_DEFAULT_PACKAGE("Quantize"),
        WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("*"), STEPSIZE_OF("*"),
          Op("wkv7_output.uint16.flat.dequant",
            WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("r"), STEPSIZE_OF("r"),
              WITH_SIZE("r", Op(FROM_DEFAULT_PACKAGE("Dequantize"), WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "r"))))
            ),
            WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "state_in"))
          )
        )
      )
    )
  )
)

// vtcm
DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7_output.fp16.flat", "r", "state_in"),
  OK,
  Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7_output.fp16.flat.tcm",
      WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "r")),
      WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "state_in"))
    )
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7_output.flat", "r", "state_in"),
  OK,
  Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7_output.flat.tcm",
      WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "r")),
      WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "state_in"))
    )
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7_output.uint16.flat.dequant", "r", "state_in"),
  OK,
  Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7_output.uint16.flat.dequant.tcm",
      WITH_SAME_OUTPUT("r", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "r")),
      WITH_SAME_OUTPUT("state_in", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "state_in"))
    )
  )
)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputFloat16Impl<PlainFloat16Tensor, PlainFloat16Tensor>), "wkv7_output.fp16.flat", FAST, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputFloat16Impl<PlainFloat16Tensor_TCM, PlainFloat16Tensor_TCM>), "wkv7_output.fp16.flat.tcm", FAST, Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputFloatImpl<PlainFloatTensor>), "wkv7_output.flat", FAST, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputFloatImpl<PlainFloatTensor_TCM>), "wkv7_output.flat.tcm", FAST, Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputFloat16Impl<PlainFloat16Tensor, QuantUint16Tensor>), "wkv7_output.uint16.flat.dequant", FAST, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputFloat16Impl<PlainFloat16Tensor_TCM, QuantUint16Tensor_TCM>), "wkv7_output.uint16.flat.dequant.tcm", FAST, Flags::RESOURCE_HVX)

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
GraphStatus wkv7OutputFloatImpl(TensorType& out_0,
                    const TensorType& r,
                    const TensorType& state_in) {
#ifdef USE_HVX
  int num_heads = state_in.dim(1);
  int head_size = state_in.dim(2);
  int seq_length = state_in.dim(0);

  auto r_ptr = (float*)r.raw_data_const();
  auto outptr = (float*)out_0.raw_data();

  float tmp_buf[32];
  HVX_Vector *state_ptr = (HVX_Vector *)(state_in.raw_data_const());

  for (int t = 0; t < seq_length; t++) {
    for (int h = 0; h < num_heads; h++) {

      HVX_Vector r_vec_0 = *(HVX_Vector *)r_ptr;
      HVX_Vector r_vec_1 = *((HVX_Vector *)r_ptr + 1);

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
      r_ptr += head_size;
    }
  }
#endif
  return GraphStatus::Success;
}

template<typename TensorType, typename StateType>
GraphStatus wkv7OutputFloat16Impl(TensorType& out_0,
                    const TensorType& r,
                    const StateType& state_in) {
#ifdef USE_HVX
  int num_heads = state_in.dim(1);
  int head_size = state_in.dim(2);
  int seq_length = r.dim(0);

  auto r_ptr = (__fp16*)r.raw_data_const();
  auto outptr = (__fp16*)out_0.raw_data();

  __fp16 __attribute__((aligned(VLEN))) tmp_buf[64];
  HVX_Vector *state_ptr = (HVX_Vector *)(state_in.raw_data_const());

  for (int t = 0; t < seq_length; t++) {
    for (int h = 0; h < num_heads; h++) {
      HVX_Vector r_vec = *(HVX_Vector *)r_ptr;
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
      r_ptr += head_size;
    }
  }
#endif
  return GraphStatus::Success;
}

template<typename TensorType>
GraphStatus wkv7OutputImplNaive(TensorType& out_0,
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


  return GraphStatus::Success;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_wkv7_output);