//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_wkv7_output_state);

// op execute function declarations
template<typename TensorType, typename StateType>
GraphStatus wkv7OutputStateFloat16Impl(TensorType& out_0,
                    const StateType& input);

template<typename TensorType>
GraphStatus wkv7OutputStateFloatImpl(TensorType& out_0,
                    const TensorType& input);

DEF_PACKAGE_OPTIMIZATION(
  CLEANUP_GRAPH + 50,
  Op("wkv7_output_state", "input"),
  IS_FLOAT16("input"),
  Op("wkv7_output_state.fp16", "input")
)

DEF_PACKAGE_OPTIMIZATION(
  CLEANUP_GRAPH + 50,
  Op("wkv7_output_state", "input"),
  IS_QUINT16("input"),
  Op("wkv7_output_state.uint16", "input")
)

// cast
DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
  CLEANUP_GRAPH + 51, relaxed_precision_flag,
  Op("wkv7_output_state", "input"),
  OK,
  MAKE_OP_FP16_AND_INSERT_CAST(Op("wkv7_output_state.fp16", CAST_TO_FP16("input")))
)

// split
// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7_output_state.fp16", "input"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7_output_state.fp16",
//       TYPICAL_SLICE("input", "I")
//     )
//   )
// )

// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7_output_state", "input"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7_output_state",
//       TYPICAL_SLICE("input", "I")
//     )
//   )
// )

// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7_output_state.uint16", "input"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7_output_state.uint16",
//       TYPICAL_SLICE("input", "I")
//     )
//   )
// )

// vtcm
// DEF_PACKAGE_OPTIMIZATION(
//   HARD_OPS + 130,
//   // Op("wkv7_output_state.fp16.flat", "input"),
//   Op("wkv7_output_state.fp16", "input"),
//   OK,
//   Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
//     Op("wkv7_output_state.fp16.flat.tcm",
//       // WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "input"))
//       "input"
//     )
//   )
// )

// DEF_PACKAGE_OPTIMIZATION(
//   HARD_OPS + 130,
//   // Op("wkv7_output_state.flat", "input"),
//   Op("wkv7_output_state", "input"),
//   OK,
//   Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
//     Op("wkv7_output_state.flat.tcm",
//       // WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "input"))
//       "input"
//     )
//   )
// )

// DEF_PACKAGE_OPTIMIZATION(
//   HARD_OPS + 130,
//   // Op("wkv7_output_state.uint16.flat.dequant", "input"),
//   Op("wkv7_output_state.uint16", "input"),
//   OK,
//   Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
//     Op("wkv7_output_state.uint16.flat.dequant.tcm",
//       // WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "input"))
//       "input"
//     )
//   )
// )

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputStateFloat16Impl<PlainFloat16Tensor, PlainFloat16Tensor>), "wkv7_output_state.fp16", SNAIL, Flags::RESOURCE_HVX)
// DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputStateFloat16Impl<PlainFloat16Tensor_TCM, PlainFloat16Tensor_TCM>), "wkv7_output_state.fp16.flat.tcm", SNAIL, Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputStateFloatImpl<PlainFloatTensor>), "wkv7_output_state", SNAIL, Flags::RESOURCE_HVX)
// DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputStateFloatImpl<PlainFloatTensor_TCM>), "wkv7_output_state.flat.tcm", SNAIL, Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputStateFloat16Impl<QuantUint16Tensor, QuantUint16Tensor>), "wkv7_output_state.uint16", SNAIL, Flags::RESOURCE_HVX)
// DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputStateFloat16Impl<QuantUint16Tensor_TCM, QuantUint16Tensor_TCM>), "wkv7_output_state.uint16.flat.dequant.tcm", SNAIL, Flags::RESOURCE_HVX)

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
GraphStatus wkv7OutputStateFloatImpl(TensorType& out_0,
                    const TensorType& input) {
#ifdef USE_HVX
  const int num_heads = input.dim(1);
  const int head_size = input.dim(3);
  const int seq_length = input.dim(2) - head_size;

  auto out_vec_ptr = (HVX_Vector*)out_0.raw_data();
  auto inptr = (float*)input.raw_data_const();

  for (int i = 0; i < num_heads; i++) {
    for (int j = 0; j < head_size; j += 8) {
      auto in_vec_ptr = (HVX_Vector*)(inptr + i * (head_size + seq_length) * head_size + (j + seq_length) * head_size);
      HVX_Vector state_vec_00 = *in_vec_ptr++;
      HVX_Vector state_vec_01 = *in_vec_ptr++;
      HVX_Vector state_vec_10 = *in_vec_ptr++;
      HVX_Vector state_vec_11 = *in_vec_ptr++;
      HVX_Vector state_vec_20 = *in_vec_ptr++;
      HVX_Vector state_vec_21 = *in_vec_ptr++;
      HVX_Vector state_vec_30 = *in_vec_ptr++;
      HVX_Vector state_vec_31 = *in_vec_ptr++;
      HVX_Vector state_vec_40 = *in_vec_ptr++;
      HVX_Vector state_vec_41 = *in_vec_ptr++;
      HVX_Vector state_vec_50 = *in_vec_ptr++;
      HVX_Vector state_vec_51 = *in_vec_ptr++;
      HVX_Vector state_vec_60 = *in_vec_ptr++;
      HVX_Vector state_vec_61 = *in_vec_ptr++;
      HVX_Vector state_vec_70 = *in_vec_ptr++;
      HVX_Vector state_vec_71 = *in_vec_ptr++;

      *out_vec_ptr++ = state_vec_00;
      *out_vec_ptr++ = state_vec_01;
      *out_vec_ptr++ = state_vec_10;
      *out_vec_ptr++ = state_vec_11;
      *out_vec_ptr++ = state_vec_20;
      *out_vec_ptr++ = state_vec_21;
      *out_vec_ptr++ = state_vec_30;
      *out_vec_ptr++ = state_vec_31;
      *out_vec_ptr++ = state_vec_40;
      *out_vec_ptr++ = state_vec_41;
      *out_vec_ptr++ = state_vec_50;
      *out_vec_ptr++ = state_vec_51;
      *out_vec_ptr++ = state_vec_60;
      *out_vec_ptr++ = state_vec_61;
      *out_vec_ptr++ = state_vec_70;
      *out_vec_ptr++ = state_vec_71;
    }
  }
#endif
  return GraphStatus::Success;
}

template<typename TensorType, typename StateType>
GraphStatus wkv7OutputStateFloat16Impl(TensorType& out_0,
                    const StateType& input) {
#ifdef USE_HVX
  const int batch_size = input.dim(0);
  const int num_heads = input.dim(1);
  const int head_size = input.dim(3);
  const int seq_length = input.dim(2) - head_size;

  // auto out_vec_ptr = (HVX_Vector*)out_0.raw_data();
  // auto inptr = (__fp16*)input.raw_data_const();

  for (int b = 0; b < batch_size; b++) {
    long indarr[] = {b, 0, 0, 0};
    auto out_vec_ptr = (HVX_Vector*)out_0.element_ptr(4, indarr);
    auto inptr = (__fp16*)input.element_ptr(4, indarr);
    for (int i = 0; i < num_heads; i++) {
      auto in_vec_ptr = (HVX_Vector*)(inptr + i * (head_size + seq_length) * head_size + seq_length * head_size);
      for (int j = 0; j < head_size; j += 8) {
        HVX_Vector state_vec_0 = *in_vec_ptr++;
        HVX_Vector state_vec_1 = *in_vec_ptr++;
        HVX_Vector state_vec_2 = *in_vec_ptr++;
        HVX_Vector state_vec_3 = *in_vec_ptr++;
        HVX_Vector state_vec_4 = *in_vec_ptr++;
        HVX_Vector state_vec_5 = *in_vec_ptr++;
        HVX_Vector state_vec_6 = *in_vec_ptr++;
        HVX_Vector state_vec_7 = *in_vec_ptr++;
        // HVX_Vector state_vec_8 = *in_vec_ptr++;
        // HVX_Vector state_vec_9 = *in_vec_ptr++;
        // HVX_Vector state_vec_10 = *in_vec_ptr++;
        // HVX_Vector state_vec_11 = *in_vec_ptr++;
        // HVX_Vector state_vec_12 = *in_vec_ptr++;
        // HVX_Vector state_vec_13 = *in_vec_ptr++;
        // HVX_Vector state_vec_14 = *in_vec_ptr++;
        // HVX_Vector state_vec_15 = *in_vec_ptr++;

        *out_vec_ptr++ = state_vec_0;
        *out_vec_ptr++ = state_vec_1;
        *out_vec_ptr++ = state_vec_2;
        *out_vec_ptr++ = state_vec_3;
        *out_vec_ptr++ = state_vec_4;
        *out_vec_ptr++ = state_vec_5;
        *out_vec_ptr++ = state_vec_6;
        *out_vec_ptr++ = state_vec_7;
        // *out_vec_ptr++ = state_vec_8;
        // *out_vec_ptr++ = state_vec_9;
        // *out_vec_ptr++ = state_vec_10;
        // *out_vec_ptr++ = state_vec_11;
        // *out_vec_ptr++ = state_vec_12;
        // *out_vec_ptr++ = state_vec_13;
        // *out_vec_ptr++ = state_vec_14;
        // *out_vec_ptr++ = state_vec_15;
      }
    }
  }
#endif
  return GraphStatus::Success;
}

template<typename TensorType>
GraphStatus wkv7OutputStateImplNaive(TensorType& out_0,
                    const TensorType& input)

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
END_PKG_OP_DEFINITION(PKG_wkv7_output_state);