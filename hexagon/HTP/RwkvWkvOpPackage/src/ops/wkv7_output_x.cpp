//==============================================================================
// Auto Generated Code for RwkvWkvOpPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_wkv7_output_x);

// op execute function declarations
template<typename TensorType, typename StateType>
GraphStatus wkv7OutputXFloat16Impl(TensorType& out_0,
                    const StateType& input);

template<typename TensorType>
GraphStatus wkv7OutputXFloatImpl(TensorType& out_0,
                    const TensorType& input);

DEF_PACKAGE_OPTIMIZATION(
  CLEANUP_GRAPH + 50,
  Op("wkv7_output_x", "input"),
  IS_FLOAT16("input"),
  Op("wkv7_output_x.fp16", "input")
)

DEF_PACKAGE_OPTIMIZATION(
  CLEANUP_GRAPH + 50,
  Op("wkv7_output_x", "input"),
  IS_QUINT16("input"),
  Op("wkv7_output_x.uint16", "input")
)

// cast
DEF_PACKAGE_OPTIMIZATION_WITH_FLAGS(
  CLEANUP_GRAPH + 51, relaxed_precision_flag,
  Op("wkv7_output_x", "input"),
  OK,
  MAKE_OP_FP16_AND_INSERT_CAST(Op("wkv7_output_x.fp16", CAST_TO_FP16("input")))
)

// split
// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7_output_x.fp16", "input"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7_output_x.fp16",
//       TYPICAL_SLICE("input", "I")
//     )
//   )
// )

// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7_output_x", "input"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7_output_x",
//       TYPICAL_SLICE("input", "I")
//     )
//   )
// )

// DEF_PACKAGE_OPTIMIZATION(
//   TILING + 100,
//   Op("wkv7_output_x.uint16", "input"),
//   GT(DIM_HEIGHT("*"), 2),
//   AUTOSPLIT(1, "I", 2,
//     Op("wkv7_output_x.uint16",
//       TYPICAL_SLICE("input", "I")
//     )
//   )
// )

// forceformat
DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7_output_x.fp16", "input"),
  OK,
  Op("wkv7_output_x.fp16.flat",
    WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "input"))
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7_output_x", "input"),
  OK,
  Op("wkv7_output_x.flat",
    WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "input"))
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 100,
  Op("wkv7_output_x.uint16", "input"),
  OK,
  WITH_SAME_OUTPUT("*",
    WITH_OUTPUT_TYPE(DType::QUInt16, ZERO_OFFSET_OF("*"), STEPSIZE_OF("*"),
      Op(FROM_DEFAULT_PACKAGE("Quantize"),
        WITH_OUTPUT_TYPE(DType::Float16, ZERO_OFFSET_OF("*"), STEPSIZE_OF("*"),
          Op("wkv7_output_x.uint16.flat.dequant",
            WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("ForceFormat_Flat"), "input"))
          )
        )
      )
    )
  )
)

// vtcm
DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7_output_x.fp16.flat", "input"),
  OK,
  Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7_output_x.fp16.flat.tcm",
      WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "input"))
    )
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7_output_x.flat", "input"),
  OK,
  Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7_output_x.flat.tcm",
      WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "input"))
    )
  )
)

DEF_PACKAGE_OPTIMIZATION(
  HARD_OPS + 130,
  Op("wkv7_output_x.uint16.flat.dequant", "input"),
  OK,
  Op(FROM_DEFAULT_PACKAGE("flat_from_vtcm"),
    Op("wkv7_output_x.uint16.flat.dequant.tcm",
      WITH_SAME_OUTPUT("input", Op(FROM_DEFAULT_PACKAGE("flat_to_vtcm"), "input"))
    )
  )
)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputXFloat16Impl<PlainFloat16Tensor, PlainFloat16Tensor>), "wkv7_output_x.fp16.flat", FAST, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputXFloat16Impl<PlainFloat16Tensor_TCM, PlainFloat16Tensor_TCM>), "wkv7_output_x.fp16.flat.tcm", FAST, Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputXFloatImpl<PlainFloatTensor>), "wkv7_output_x.flat", FAST, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputXFloatImpl<PlainFloatTensor_TCM>), "wkv7_output_x.flat.tcm", FAST, Flags::RESOURCE_HVX)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputXFloat16Impl<PlainFloat16Tensor, QuantUint16Tensor>), "wkv7_output_x.uint16.flat.dequant", FAST, Flags::RESOURCE_HVX)
DEF_PACKAGE_OP_AND_COST_AND_FLAGS((wkv7OutputXFloat16Impl<PlainFloat16Tensor_TCM, QuantUint16Tensor_TCM>), "wkv7_output_x.uint16.flat.dequant.tcm", FAST, Flags::RESOURCE_HVX)

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
GraphStatus wkv7OutputXFloatImpl(TensorType& out_0,
                    const TensorType& input) {
#ifdef USE_HVX
  const int num_heads = input.dim(1);
  const int head_size = input.dim(3);
  const int seq_length = input.dim(2) - head_size;

  // auto r_ptr = (float*)r.raw_data_const();
  auto out_vec_ptr = (HVX_Vector*)out_0.raw_data();
  // auto inptr = (HVX_Vector*)input.raw_data_const();
  // auto outptr = (float*)out_0.raw_data();
  auto inptr = (float*)input.raw_data_const();
  for (int i = 0; i < seq_length; i++) {
    // auto in_vec_ptr = (HVX_Vector*)(inptr + i * head_size);
    for (int h = 0; h < num_heads; h++) {
      auto in_vec_ptr = (HVX_Vector*)(inptr + h * (seq_length + head_size) * head_size + i * head_size);
      HVX_Vector x_vec_0 = *in_vec_ptr;
      HVX_Vector x_vec_1 = *(in_vec_ptr + 1);
      // HVX_Vector x_vec_0 = *in_vec_ptr;
      // HVX_Vector x_vec_1 = *(in_vec_ptr + 1);
      // in_vec_ptr += 
      // HVX_Vector x_vec_2 = *in_vec_ptr++;
      // HVX_Vector x_vec_3 = *in_vec_ptr++;
      // HVX_Vector x_vec_4 = *in_vec_ptr++;
      // HVX_Vector x_vec_5 = *in_vec_ptr++;
      // HVX_Vector x_vec_6 = *in_vec_ptr++;
      // HVX_Vector x_vec_7 = *in_vec_ptr++;

      *out_vec_ptr++ = x_vec_0;
      *out_vec_ptr++ = x_vec_1;
      // *out_vec_ptr++ = x_vec_2;
      // *out_vec_ptr++ = x_vec_3;
      // *out_vec_ptr++ = x_vec_4;
      // *out_vec_ptr++ = x_vec_5;
      // *out_vec_ptr++ = x_vec_6;
      // *out_vec_ptr++ = x_vec_7;
    }
  }
#endif
  return GraphStatus::Success;
}

template<typename TensorType, typename StateType>
GraphStatus wkv7OutputXFloat16Impl(TensorType& out_0,
                    const StateType& input) {
#ifdef USE_HVX
  const int num_heads = input.dim(1);
  const int head_size = input.dim(3);
  const int seq_length = input.dim(2) - head_size;

  // auto outptr = (__fp16*)out_0.raw_data();
  auto out_vec_ptr = (HVX_Vector*)out_0.raw_data();
  auto inptr = (__fp16*)input.raw_data_const();

  for (int i = 0; i < seq_length; i++) {
    for (int h = 0; h < num_heads; h++) {
      auto in_vec_ptr = (HVX_Vector*)(inptr + h * (seq_length + head_size) * head_size + i * head_size);
      HVX_Vector x_vec_0 = *in_vec_ptr;
      // in_vec_ptr += (seq_length + head_size);
      // HVX_Vector x_vec_1 = *in_vec_ptr;
      // in_vec_ptr += (seq_length + head_size);
      // HVX_Vector x_vec_2 = *in_vec_ptr;
      // in_vec_ptr += (seq_length + head_size);
      // HVX_Vector x_vec_3 = *in_vec_ptr;

      *out_vec_ptr++ = x_vec_0;
      // *out_vec_ptr++ = x_vec_1;
      // *out_vec_ptr++ = x_vec_2;
      // *out_vec_ptr++ = x_vec_3;
    }
  }
#endif
  return GraphStatus::Success;
}

template<typename TensorType>
GraphStatus wkv7OutputXImplNaive(TensorType& out_0,
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
END_PKG_OP_DEFINITION(PKG_wkv7_output_x);