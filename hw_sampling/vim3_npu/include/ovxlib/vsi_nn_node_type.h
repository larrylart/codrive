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
#ifndef _VSI_NN_NODE_TYPES_H_
#define _VSI_NN_NODE_TYPES_H_

#include "vsi_nn_types.h"
#include "ops/vsi_nn_op_activations.h"
#include "ops/vsi_nn_op_batch_norm.h"
#include "ops/vsi_nn_op_multiply.h"
#include "ops/vsi_nn_op_concat.h"
#include "ops/vsi_nn_op_split.h"
#include "ops/vsi_nn_op_conv2d.h"
#include "ops/vsi_nn_op_deconvolution.h"
#include "ops/vsi_nn_op_fullconnect.h"
#include "ops/vsi_nn_op_lrn.h"
#include "ops/vsi_nn_op_permute.h"
#include "ops/vsi_nn_op_pool.h"
#include "ops/vsi_nn_op_proposal.h"
#include "ops/vsi_nn_op_reshape.h"
#include "ops/vsi_nn_op_roi_pool.h"
#include "ops/vsi_nn_op_upsample.h"
#include "ops/vsi_nn_op_resize.h"
#include "ops/vsi_nn_op_lstm.h"
#include "ops/vsi_nn_op_reorg.h"
#include "ops/vsi_nn_op_l2normalizescale.h"
#include "ops/vsi_nn_op_crop.h"
#include "ops/vsi_nn_op_relun.h"
#include "ops/vsi_nn_op_divide.h"
#include "ops/vsi_nn_op_tanh.h"
#include "ops/vsi_nn_op_dropout.h"
#include "ops/vsi_nn_op_shufflechannel.h"
#include "ops/vsi_nn_op_prelu.h"
#include "ops/vsi_nn_op_elu.h"
#include "ops/vsi_nn_op_reverse.h"
#include "ops/vsi_nn_op_space2depth.h"
#include "ops/vsi_nn_op_depth2space.h"
#include "ops/vsi_nn_op_eltwisemax.h"
#include "ops/vsi_nn_op_scale.h"
#include "ops/vsi_nn_op_slice.h"
#include "ops/vsi_nn_op_space2batch.h"
#include "ops/vsi_nn_op_batch2space.h"
#include "ops/vsi_nn_op_pad.h"
#include "ops/vsi_nn_op_imageprocess.h"
#include "ops/vsi_nn_op_matrixmul.h"
#include "ops/vsi_nn_op_lstmunit.h"
#include "ops/vsi_nn_op_layernormalize.h"
#include "ops/vsi_nn_op_reduce.h"
#include "ops/vsi_nn_op_softmax.h"
#include "ops/vsi_nn_op_instancenormalize.h"
#include "ops/vsi_nn_op_tensorstackconcat.h"
#include "ops/vsi_nn_op_strided_slice.h"
#include "ops/vsi_nn_op_signalframe.h"
#include "ops/vsi_nn_op_argmax.h"
#include "ops/vsi_nn_op_svdf.h"
#include "ops/vsi_nn_op_conv1d.h"
#include "ops/vsi_nn_op_nbg.h"
#include "ops/vsi_nn_op_minimum.h"
#include "ops/vsi_nn_op_spatial_transformer.h"
#include "ops/vsi_nn_op_logical_ops.h"
#include "ops/vsi_nn_op_select.h"
#include "ops/vsi_nn_op_concatshift.h"
#include "ops/vsi_nn_op_relational_ops.h"
#include "ops/vsi_nn_op_pow.h"
#include "ops/vsi_nn_op_floordiv.h"
#include "ops/vsi_nn_op_lstmunit_activation.h"
#include "ops/vsi_nn_op_lstmunit_ovxlib.h"
#include "ops/vsi_nn_op_tensor_add_mean_stddev_norm.h"
#include "ops/vsi_nn_op_stack.h"

/* custom node head define define */
#include "custom/vsi_nn_custom_node_type.h"

#if defined(__cplusplus)
extern "C"{
#endif

typedef union _vsi_nn_nn_param
{
    struct
    {
        vsi_nn_conv2d_param         conv2d;
        vsi_nn_pool_param           pool;
    };
    vsi_nn_fcl_param                fcl;
    vsi_nn_activation_param         activation;
    vsi_nn_lrn_param                lrn;
    vsi_nn_concat_param             concat;
    vsi_nn_split_param              split;
    vsi_nn_roi_pool_param           roi_pool;
    vsi_nn_batch_norm_param         batch_norm;
    vsi_nn_multiply_param           multiply;
    vsi_nn_proposal_param           proposal;
    vsi_nn_deconv_param             deconv;
    vsi_nn_reshape_param            reshape;
    vsi_nn_permute_param            permute;
    vsi_nn_upsample_param           upsample;
    vsi_nn_resize_param             resize;
    vsi_nn_lstm_param               lstm;
    vsi_nn_reorg_param              reorg;
    vsi_nn_l2normalizescale_param   l2normalizescale;
    vsi_nn_crop_param               crop;
    vsi_nn_relun_param              relun;
    vsi_nn_divide_param             divide;
    vsi_nn_tanh_param               tanh;
    vsi_nn_dropout_param            dropout;
    vsi_nn_shufflechannel_param     shufflechannel;
    vsi_nn_prelu_param              prelu;
    vsi_nn_elu_param                elu;
    vsi_nn_reverse_param            reverse;
    vsi_nn_space2depth_param        space2depth;
    vsi_nn_depth2space_param        depth2space;
    vsi_nn_eltwisemax_param         eltwisemax;
    vsi_nn_scale_param              scale;
    vsi_nn_slice_param              slice;
    vsi_nn_space2batch_param        space2batch;
    vsi_nn_batch2space_param        batch2space;
    vsi_nn_pad_param                pad;
    vsi_nn_imageprocess_param       imageprocess;
    vsi_nn_matrixmul_param          matrixmul;
    vsi_nn_lstmunit_param           lstmunit;
    vsi_nn_layernormalize_param     layernorm;
    vsi_nn_reduce_param             reduce;
    vsi_nn_instancenormalize_param  instancenorm;
    vsi_nn_tensorstackconcat_param  tensorstackconcat;
    vsi_nn_softmax_param            softmax;
    vsi_nn_strided_slice_param      strided_slice;
    vsi_nn_signalframe_param        signalframe;
    vsi_nn_svdf_param               svdf;
    vsi_nn_conv1d_param             conv1d;
    vsi_nn_nbg_param                nbg;
    vsi_nn_concatshift_param        concatshift;
    vsi_nn_relational_ops_param     relational_ops;
    vsi_nn_pow_param                pow;
    vsi_nn_floordiv_param           floordiv;
    vsi_nn_minimum_param            minimum;
    vsi_nn_spatial_transformer_param spatial_transformer;
    vsi_nn_logical_ops_param        logical_ops;
    vsi_nn_select_param             select;
    vsi_nn_lstmunit_activation_param lstmunit_activation;
    vsi_nn_lstmunit_ovxlib_param    lstmunit_ovxlib;
    vsi_nn_tensor_add_mean_stddev_norm_param tensor_add_mean_stddev_norm;
    vsi_nn_stack_param              stack;
    uint8_t                         client_param[128];

    /* custom node data struct define */
#define DEF_NODE_TYPE( NAME ) vsi_nn_##NAME##_param NAME;
    #include "custom/custom_node_type.def"
#undef DEF_NODE_TYPE
} vsi_nn_nn_param_t;

typedef struct _vsi_nn_vx_param
{
    vsi_enum   overflow_policy;
    vsi_enum   rounding_policy;
    vsi_enum   down_scale_size_rounding;
    vsi_bool   has_relu;
    uint32_t accumulator_bits;
    vsi_nn_platform_e platform;
} vsi_nn_vx_param_t;

#if defined(__cplusplus)
}
#endif

#endif
