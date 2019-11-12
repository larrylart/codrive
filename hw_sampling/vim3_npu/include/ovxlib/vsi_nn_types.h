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
#ifndef _VSI_NN_TYPES_H_
#define _VSI_NN_TYPES_H_

#include <stdint.h>
#include "vsi_nn_platform.h"

#if defined(__cplusplus)
extern "C"{
#endif

#ifdef _WIN32
#define inline __inline
#endif

/* Basic data type definition */
typedef int32_t  vsi_enum;
typedef int32_t  vsi_status;
typedef int32_t   vsi_bool;

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

/* Status enum */
typedef enum _vsi_nn_status_e
{
    VSI_FAILURE = VX_FAILURE,
    VSI_SUCCESS = VX_SUCCESS,
}vsi_nn_status_e;

/* Pad enum */
typedef enum _vsi_nn_pad_e
{
    VSI_NN_PAD_AUTO,
    VSI_NN_PAD_VALID,
    VSI_NN_PAD_SAME
} vsi_nn_pad_e;

/* Platform enum */
typedef enum _vsi_nn_platform_e
{
    VSI_NN_PLATFORM_CAFFE,
    VSI_NN_PLATFORM_TENSORFLOW
}vsi_nn_platform_e;

/* Round type enum */
typedef enum _vsi_nn_round_type_e
{
    VSI_NN_ROUND_CEIL,
    VSI_NN_ROUND_FLOOR
}vsi_nn_round_type_e;

/* Optimize driction */
typedef enum _vsi_nn_opt_direction_e
{
    VSI_NN_OPTIMIZE_FORWARD,
    VSI_NN_OPTIMIZE_BACKWARD
} vsi_nn_opt_direction_e;

/* Type enum */
typedef enum _vsi_nn_type_e
{
    VSI_NN_TYPE_INT8 = VX_TYPE_INT8,
    VSI_NN_TYPE_INT16 = VX_TYPE_INT16,
    VSI_NN_TYPE_INT32 = VX_TYPE_INT32,
    VSI_NN_TYPE_INT64 = VX_TYPE_INT64,
    VSI_NN_TYPE_UINT8 = VX_TYPE_UINT8,
    VSI_NN_TYPE_UINT16 = VX_TYPE_UINT16,
    VSI_NN_TYPE_UINT32 = VX_TYPE_UINT32,
    VSI_NN_TYPE_UINT64 = VX_TYPE_UINT64,
    VSI_NN_TYPE_FLOAT16 = VX_TYPE_FLOAT16,
    VSI_NN_TYPE_FLOAT32 = VX_TYPE_FLOAT32,
    VSI_NN_TYPE_FLOAT64 = VX_TYPE_FLOAT64,
    VSI_NN_TYPE_VDATA = VX_TYPE_USER_STRUCT_START + 0x1,
}vsi_nn_type_e;

typedef uint32_t vsi_nn_size_t;

typedef uint32_t vsi_nn_tensor_id_t;

typedef uint32_t vsi_nn_node_id_t;

typedef struct _vsi_nn_graph vsi_nn_graph_t;

typedef struct _vsi_nn_node vsi_nn_node_t;

typedef struct _vsi_nn_tensor vsi_nn_tensor_t;

#if defined(__cplusplus)
}
#endif

#endif
