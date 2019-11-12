/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#ifndef _VSI_NN_OP_LSTMUNIT_OVXLIB_H
#define _VSI_NN_OP_LSTMUNIT_OVXLIB_H

#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"
#include "vsi_nn_op_lstmunit.h"
#include "vsi_nn_internal_node.h"

typedef int32_t vsi_nnlstmunit_ovxlib_internal_node_index_t; enum
{
    LSTMUNIT_OVXLIB_INT_INVALID = -1,
    /* Add internal node def */

    LSTMUNIT_OVXLIB_FC_I2I,
    LSTMUNIT_OVXLIB_FC_I2F,
    LSTMUNIT_OVXLIB_FC_I2C,
    LSTMUNIT_OVXLIB_FC_I2O,

    LSTMUNIT_OVXLIB_TEST_NODE,

    LSTMUNIT_OVXLIB_FC_R2I,
    LSTMUNIT_OVXLIB_FC_R2F,
    LSTMUNIT_OVXLIB_FC_R2C,
    LSTMUNIT_OVXLIB_FC_R2O,

    LSTMUNIT_OVXLIB_INPUT_FC_OUTPUTS_CONCAT,
    LSTMUNIT_OVXLIB_RECURRENT_FC_OUTPUTS_CONCAT,
    LSTMUNIT_OVXLIB_LAYER_NORM,
    LSTMUNIT_OVXLIB_LAYER_NORM_SPLIT,

    LSTMUNIT_OVXLIB_LAYER_NORM_I,
    LSTMUNIT_OVXLIB_LAYER_NORM_F,
    LSTMUNIT_OVXLIB_LAYER_NORM_C,
    LSTMUNIT_OVXLIB_LAYER_NORM_O,

    LSTMUNIT_OVXLIB_ACTIVATIONS, /* Activations */

    LSTMUNIT_OVXLIB_FC_PROJ,
    LSTMUNIT_OVXLIB_ADD_PROJ,

    LSTMUNIT_OVXLIB_INT_NODE_CNT
};

enum
{
    LSTMUNIT_TENSOR_ZERO_BIAS_I2I,
    LSTMUNIT_TENSOR_ZERO_BIAS_I2F,
    LSTMUNIT_TENSOR_ZERO_BIAS_I2C,
    LSTMUNIT_TENSOR_ZERO_BIAS_I2O,

    LSTMUNIT_TENSOR_ZERO_BIAS_R2I,
    LSTMUNIT_TENSOR_ZERO_BIAS_R2F,
    LSTMUNIT_TENSOR_ZERO_BIAS_R2C,
    LSTMUNIT_TENSOR_ZERO_BIAS_R2O,

    LSTMUNIT_TENSOR_CONCATED_BIAS,
    LSTMUNIT_TENSOR_CONCATED_LN_W,

    LSTMUNIT_TENSOR_OUTPUT_I2I,
    LSTMUNIT_TENSOR_OUTPUT_I2F,
    LSTMUNIT_TENSOR_OUTPUT_I2C,
    LSTMUNIT_TENSOR_OUTPUT_I2O,
    LSTMUNIT_TENSOR_OUTPUT_R2I,
    LSTMUNIT_TENSOR_OUTPUT_R2F,
    LSTMUNIT_TENSOR_OUTPUT_R2C,
    LSTMUNIT_TENSOR_OUTPUT_R2O,

    LSTMUNIT_TENSOR_INPUT_FC_OUTPUTS,
    LSTMUNIT_TENSOR_RECURRENT_FC_OUTPUTS,
    LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT,

    LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_I,
    LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_F,
    LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_C,
    LSTMUNIT_TENSOR_LAYER_NORM_OUTPUT_O,

    LSTMUNIT_TENSOR_ACTIVATION_OUTPUT,
    LSTMUNIT_TENSOR_ZERO_BIAS_PROJECTION,
    LSTMUNIT_TENSOR_PROJECTION_FC_OUTPUT,

    LSTMUNIT_TENSOR_CNT
};

enum
{
    LSTMUNIT_IFC_I,
    LSTMUNIT_IFC_F,
    LSTMUNIT_IFC_C,
    LSTMUNIT_IFC_O,
    LSTMUNIT_HFC_I,
    LSTMUNIT_HFC_F,
    LSTMUNIT_HFC_C,
    LSTMUNIT_HFC_O,
    LSTMUNIT_FC_CNT
};

typedef struct _vsi_nn_lstmunit_ovxlib_lcl_data_t
{
    vsi_nn_internal_node_t* nodes[LSTMUNIT_OVXLIB_INT_NODE_CNT];
    vsi_nn_tensor_t* tensors[LSTMUNIT_TENSOR_CNT];
    vsi_bool use_cifg;
    vsi_bool use_layer_norm;
    vsi_bool use_projection;
    vsi_bool use_projection_bias;
} vsi_nn_lstmunit_ovxlib_lcl_data_t;

typedef struct _vsi_nn_lstmunit_ovxlib_param
{
    vsi_nn_lstmunit_ovxlib_lcl_data_t local;

    float cell_clip;
    float proj_clip;
    vsi_nn_lstmunit_activation_e activation;
    float forget_bias;
    vsi_nn_dtype_t internal_dtype[LSTMUNIT_FC_CNT];
} vsi_nn_lstmunit_ovxlib_param;

#endif

