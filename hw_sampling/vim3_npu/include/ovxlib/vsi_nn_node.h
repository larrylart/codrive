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
#ifndef _VSI_NN_NODE_H
#define _VSI_NN_NODE_H

/*------------------------------------
               Includes
  -----------------------------------*/
#include "vsi_nn_platform.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_types.h"
#include "vsi_nn_node_type.h"

#if defined(__cplusplus)
extern "C"{
#endif

/*------------------------------------
                Macros
  -----------------------------------*/

#define VSI_NN_NODE_ID_NA           ((uint32_t)-1)
#define VSI_NN_NODE_UID_NA          ((uint32_t)-1)

/*------------------------------------
                Types
  -----------------------------------*/
struct _vsi_nn_node
{
    /* VSI NN graph */
    vsi_nn_graph_t * graph;
    /* OpenVX node */
    vx_node          n;
    /* VSI NN operation type */
    vsi_nn_op_t      op;
    struct
    {
        vsi_nn_tensor_id_t * tensors;
        uint32_t            num;
    } input;
    struct
    {
        vsi_nn_tensor_id_t * tensors;
        uint32_t            num;
    } output;
    /* Parameters */
    vsi_nn_nn_param_t nn_param;
    vsi_nn_vx_param_t vx_param;
    /* uid - User specific ID, debug only*/
    uint32_t uid;
};

/*------------------------------------
              Functions
  -----------------------------------*/
OVXLIB_API vsi_nn_node_t * vsi_nn_NewNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_op_t      op,
    uint32_t         input_num,
    uint32_t         output_num
    );

OVXLIB_API vsi_nn_node_t * vsi_nn_CreateNode
    (
    vsi_nn_graph_t * graph,
    vsi_nn_op_t      op
    );

OVXLIB_API void vsi_nn_ReleaseNode
    (
    vsi_nn_node_t ** node
    );

OVXLIB_API void vsi_nn_PrintNode
    (
    vsi_nn_node_t * node,
    vsi_nn_node_id_t id
    );

#if defined(__cplusplus)
}
#endif

#endif
