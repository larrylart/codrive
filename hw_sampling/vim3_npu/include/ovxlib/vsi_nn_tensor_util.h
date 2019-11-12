/****************************************************************************
*
*    Copyright (c) 2018 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
#ifndef _VSI_NN_TENSOR_UTIL_H
#define _VSI_NN_TENSOR_UTIL_H

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "vsi_nn_platform.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*-------------------------------------------
                Types
-------------------------------------------*/

typedef enum
{
    VSI_NN_TENSOR_ATTR_DIM_NUM = 0x1,
    VSI_NN_TENSOR_ATTR_DTYPE = 0x2,
    VSI_NN_TENSOR_ATTR_SIZE = 0x4,
    VSI_NN_TENSOR_ATTR_FIXED_POINT_POS = 0x8,
    VSI_NN_TENSOR_ATTR_CONST = 0x10,
    VSI_NN_TENSOR_ATTR_HIGH_PRECISION = 0x20,
    VSI_NN_TENSOR_ATTR_ALL =  0xFF
}vsi_nn_vxtensor_attr_t;

/*-------------------------------------------
        Macros and Variables
-------------------------------------------*/

#define vsi_nn_hasattr( mask, attr )    (( mask & attr ) != 0)

/*-------------------------------------------
                  Functions
-------------------------------------------*/

OVXLIB_API vsi_nn_tensor_t * vsi_nn_CreateTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr
    );

vsi_bool vsi_nn_TensorReinit
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor
    );

OVXLIB_API void vsi_nn_ReleaseTensor
    (
    vsi_nn_tensor_t ** tensor
    );

OVXLIB_API vsi_status vsi_nn_SetTensorAttr
    (
    vsi_nn_tensor_t * tensor,
    const vsi_nn_vxtensor_attr_t attrs
    );

OVXLIB_API vsi_status vsi_nn_QueryTensorAttr
    (
    vsi_nn_tensor_t * tensor,
    const vsi_nn_vxtensor_attr_t attrs
    );

OVXLIB_API uint8_t * vsi_nn_ConvertTensorToData
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor
    );

OVXLIB_API float * vsi_nn_ConvertTensorToFloat32Data
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor
    );

/*
 * Deprecated: Use vsi_nn_ConvertRawTensorToData2() instead
 */
OVXLIB_API uint8_t * vsi_nn_ConvertRawTensorToData
    (
    vx_context context,
    vx_tensor tensor,
    uint32_t * dim,
    vx_enum  * data_format,
    uint32_t * size,
    uint32_t * stride_size,
    vx_tensor_addressing * addr,
    vx_enum accessor
    );

OVXLIB_API uint8_t * vsi_nn_ConvertRawTensorToData2
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t * attr,
    uint32_t * stride_size,
    vx_tensor_addressing * addr,
    vx_enum accessor
    );

OVXLIB_API void vsi_nn_SaveTensorToText
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename,
    char             * seperator
    );

OVXLIB_API void vsi_nn_SaveTensorToTextByFp32
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename,
    char             * seperator
    );

OVXLIB_API void vsi_nn_SaveDataToText
    (
    const char  * filename,
    uint8_t    * data,
    uint32_t     data_size,
    vsi_nn_type_e data_format,
    char        * seperator
    );

OVXLIB_API void vsi_nn_SaveTensorToBinary
    (
    vsi_nn_graph_t   * graph,
    vsi_nn_tensor_t  * tensor,
    const char       * filename
    );

OVXLIB_API vsi_nn_tensor_t * vsi_nn_CreateTensorFromData
    (
    vsi_nn_graph_t       * graph,
    uint8_t             * data,
    vsi_nn_tensor_attr_t * attr
    );

OVXLIB_API vsi_status vsi_nn_CopyDataToTensor
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_t      * tensor,
    uint8_t             * data
    );

OVXLIB_API vsi_status vsi_nn_CopyRawDataToTensor
    (
    vsi_nn_graph_t*         graph,
    uint8_t*                src_data,
    const vsi_nn_dtype_t*   src_dtype,
    vsi_nn_tensor_t*        tensor
    );

OVXLIB_API uint32_t vsi_nn_CopyTensorToBuffer
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    uint8_t        * buffer
    );

OVXLIB_API void vsi_nn_PrintNodeIO
    (
    vsi_nn_graph_t *graph,
    vsi_nn_node_t *node
    );

OVXLIB_API void vsi_nn_PrintTensor
    (
    vsi_nn_tensor_t * tensor,
    vsi_nn_tensor_id_t id
    );

OVXLIB_API void vsi_nn_TransposeTensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * tensor,
    uint32_t       * perm,
    uint32_t         dim_num,
    uint32_t       * as_shape
    );

OVXLIB_API vsi_bool vsi_nn_CalcReshapeTensor
    (
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    uint32_t       * shape,
    uint32_t         dim_num
    );

OVXLIB_API vsi_bool vsi_nn_ReshapeTensor
    (
    vsi_nn_graph_t  * graph,
    vsi_nn_tensor_t * input,
    vsi_nn_tensor_t * output,
    uint32_t       * shape,
    uint32_t         dim_num
    );

OVXLIB_API vsi_nn_size_t vsi_nn_GetElementNum
    (
    vsi_nn_tensor_t * tensor
    );

OVXLIB_API uint32_t vsi_nn_GetTensorSize
    (
    uint32_t   * shape,
    uint32_t     dim_num,
    vsi_nn_type_e dtype
    );

OVXLIB_API vsi_nn_tensor_t * vsi_nn_VariableToTensor
    (
    vsi_nn_node_t * self,
    uint8_t * data,
    vsi_nn_type_e type
    );

OVXLIB_API void vsi_nn_Free
    (
    void * data
    );

OVXLIB_API vx_tensor vsi_nn_CreateViewTensor
    (
    vsi_nn_graph_t *graph,
    uint32_t *start,
    uint32_t *end,
    vsi_nn_tensor_t *tensor
    );

OVXLIB_API void vsi_nn_ReleaseTensorRelevance
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_rel_t *tensor_ref
    );

OVXLIB_API vsi_nn_tensor_rel_t *vsi_nn_CreateTensorRelevance
    (
    vsi_nn_graph_t *graph
    );

OVXLIB_API vsi_nn_tensor_t * vsi_nn_CreateTensorFromHandle
    (
    vsi_nn_graph_t       * graph,
    uint8_t              * data,
    vsi_nn_tensor_attr_t * attr
    );

OVXLIB_API vsi_status vsi_nn_SwapTensorHandle
    (
    vsi_nn_tensor_t * tensor0,
    vsi_nn_tensor_t * tensor1
    );

OVXLIB_API vsi_nn_size_t vsi_nn_vxGetTensorElementNum
    (
    vsi_nn_tensor_attr_t *attr
    );

OVXLIB_API vsi_status vsi_nn_vxGetTensorAttr
    (
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr
    );

OVXLIB_API uint8_t *vsi_nn_vxCopyTensorToData
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr
    );

OVXLIB_API vsi_status vsi_nn_vxCopyDataToTensor
    (
    vx_context context,
    vx_tensor tensor,
    vsi_nn_tensor_attr_t *attr,
    uint8_t *data
    );

OVXLIB_API vsi_nn_tensor_t * vsi_nn_CreateTensorWithDefault
    (
    vsi_nn_graph_t       * graph,
    vsi_nn_tensor_attr_t * attr,
    float                  defualt_value
    );

#ifdef __cplusplus
}
#endif

#endif
