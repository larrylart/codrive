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
#ifndef _VSI_NN_INTRNAL_NODE_H
#define _VSI_NN_INTRNAL_NODE_H

#include "vsi_nn_platform.h"
#include "vsi_nn_context.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_node.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_types.h"
#include "vsi_nn_rnn.h"
#include "utils/vsi_nn_map.h"

typedef struct _vsi_nn_internal_node_t
{
    vsi_nn_node_t* node;
    vsi_nn_tensor_t** inputs;
    vsi_nn_tensor_t** outputs;
} vsi_nn_internal_node_t;

vsi_nn_internal_node_t* vsi_nn_internal_create_node
    (
    vsi_nn_graph_t* graph,
    vsi_nn_op_t op,
    uint32_t input_num,
    uint32_t output_num
    );

vsi_status vsi_nn_internal_release_node
    (
    vsi_nn_internal_node_t** node
    );

#endif /* _VSI_NN_INTRNAL_NODE_H */
