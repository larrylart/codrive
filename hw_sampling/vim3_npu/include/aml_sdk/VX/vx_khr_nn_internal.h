/****************************************************************************
*
*    Copyright (c) 2005 - 2019 by Vivante Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Vivante Corporation. This is proprietary information owned by
*    Vivante Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Vivante Corporation.
*
*****************************************************************************/


/*

 * Copyright (c) 2012-2017 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _VX_KHR_NN_INTERNAL_H_
#define _VX_KHR_NN_INTERNAL_H_

/*!
 * \file
 * \brief The Khronos Extension for Deep Convolutional Networks Functions.
 *
 * \defgroup group_cnn Extension: Deep Convolutional Networks API
 * \brief Convolutional Network Nodes.
 */

#define OPENVX_KHR_NN_INTERNAL   "vx_khr_nn_internal"

#include <VX/vx.h>


#ifdef __cplusplus
extern "C" {
#endif

/*TODO: check it for OpenVX 1.2*/
//#if defined(OPENVX_CNN_1_0)
//#undef OPENVX_CNN_1_1
//#endif

/*! \brief [Graph] Creates a Convolutional Network Convolution and Activation(Relu) and pooling Layer Node.
* \details This function implement Convolutional Network Convolution and Activation(Relu) and pooling layer.
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
* The dimension order is [width, height, #IFM, #batches]. \n
* \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference.\n
* \param [in] pad_x [static] Number of elements added at each side in the x dimension of the input.
* \param [in] pad_y [static] Number of elements added at each side in the y dimension of the input. In fully connected layers this input is ignored.
* \param [in] accumulator_bits [static] Is the total number of bits used during intermediate accumulation.
* \param [in] overflow_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration.
* \param [in] rounding_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
* \param [in] down_scale_size_rounding [static] Rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt>
* \param [in] enable_relu [static] If true, enable vxActivationLayer's relu function
* \param [in] pool_type [static] if neither max pooling nor average pooling, disable pooling function. (see <tt>\ref vx_convolutional_network_pooling_type_e</tt>).
* \param [in] pool_size_x [static] Size of the pooling region in the x dimension
* \param [in] pool_size_y [static] Size of the pooling region in the y dimension.
* \param [out] outputs The output tensor data. Output will have the same number and structure of dimensions as input.
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_cnn
*/
VX_API_ENTRY vx_node VX_API_CALL vxConvolutionReluPoolingLayer(
    vx_graph                    graph,
    vx_tensor                   inputs,
    vx_weights_biases_parameter weights_biases,
    vx_uint32                   pad_x,
    vx_uint32                   pad_y,
    vx_uint8                    accumulator_bits,
    vx_enum                     overflow_policy,
    vx_enum                     rounding_policy,
    vx_enum                     down_scale_size_rounding,
    vx_bool                     enable_relu,
    vx_enum                     pool_type,
    vx_uint32                   pool_size_x,
    vx_uint32                   pool_size_y,
    vx_tensor                   outputs
    );

/*! \brief [Graph] Creates a Convolutional Network Convolution and Activation(Relu) Layer Node.
* \details This function implement Convolutional Network Convolution and Activation(Relu) layer.
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * The dimension order is [width, height, #IFM, #batches]. \n
* \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference.
* \param [in] pad_x [static] Number of elements added at each side in the x dimension of the input.
* \param [in] pad_y [static] Number of elements added at each side in the y dimension of the input. In fully connected layers this input is ignored.
* \param [in] accumulator_bits [static] Is the total number of bits used during intermediate accumulation.
* \param [in] overflow_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration.
* \param [in] rounding_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
* \param [in] down_scale_size_rounding [static] Rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt>
* \param [in] enable_relu [static] If true, enable vxActivationLayer's relu function.
* \param [out] outputs The output tensor data. Output will have the same number and structure of dimensions as input.
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_cnn
*/

VX_API_ENTRY vx_node VX_API_CALL vxConvolutionReluLayer(
    vx_graph                    graph,
    vx_tensor                   inputs,
    vx_weights_biases_parameter weights_biases,
    vx_uint32                   pad_x,
    vx_uint32                   pad_y,
    vx_uint8                    accumulator_bits,
    vx_enum                     overflow_policy,
    vx_enum                     rounding_policy,
    vx_enum                     down_scale_size_rounding,
    vx_bool                     enable_relu,
    vx_tensor                   outputs
    );

/*! \brief [Graph] Creates a Fully connected and Activation(Relu) Convolutional Network Layer Node.
* \details This function implement Fully connected and Activation(Relu) Convolutional Network layers.
* \param [in] graph The handle to the graph.
* \param [in] inputs The input tensor data. There two possible input layouts:
* 1. [#IFM, #batches]. See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>.
* 2. [width, height, #IFM, #batches]. See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>\n
* In both cases number of batches are optional and may be multidimensional.
* The second option is a special case to deal with convolution layer followed by fully connected.
* The dimension order is [#IFM, #batches]. See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>. Note that batch may be multidimensional.
* \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference.\n
* \param [in] pad [static] Number of elements added at each side in the input.
* \param [in] accumulator_bits [static] Is the total number of bits used during intermediate accumulation.
* \param [in] overflow_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_convert_policy_e</tt> enumeration.
* \param [in] rounding_policy [static] A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
* \param [in] down_scale_size_rounding [static] Rounding method for calculating output dimensions. See <tt>\ref vx_convolutional_network_rounding_type_e</tt>
* \param [in] enable_relu [static] If true, enable vxActivationLayer's relu function.
* \param [out] outputs The output tensor data. Output dimension layout is [#OFM,#batches]. See <tt>\ref vxCreateTensor</tt> and <tt>\ref vxCreateVirtualTensor</tt>, where #batches may be multidimensional.
* \return <tt> vx_node</tt>.
* \retval 0 Node could not be created.
* \retval * Node handle.
* \ingroup group_cnn
*/
VX_API_ENTRY vx_node VX_API_CALL vxFullyConnectedReluLayer(
    vx_graph                    graph,
    vx_tensor                   inputs,
    vx_weights_biases_parameter weights_biases,
    vx_uint32                   pad,
    vx_uint8                    accumulator_bits,
    vx_enum                     overflow_policy,
    vx_enum                     rounding_policy,
    vx_enum                     down_scale_size_rounding,
    vx_bool                     enable_relu,
    vx_tensor                   outputs
    );

/*! \brief Input parameter for convolutionReluPooling2
 * \ingroup group_cnn
 */
typedef struct _vx_nn_convolution_relu_pooling_params_t
{
    vx_size   dilation_x;                /*!< \brief  "inflate" the kernel by inserting zeros between the kernel elements in the x direction.
                                              The value is the number of zeros to insert. */
    vx_size   dilation_y;                /*!< \brief  "inflate" the kernel by inserting zeros between the kernel elements in the y direction.
                                              The value is the number of zeros to insert. */
    vx_uint32  pad_x_left;                /*!< \brief  Number of elements added at each side in the left of x dimension of the input. */
    vx_uint32  pad_x_right;               /*!< \brief  Number of elements added at each side in the right of x dimension of the input. */
    vx_uint32  pad_y_top;                 /*!< \brief  Number of elements added at each side in the top of y dimension of the input. */
    vx_uint32  pad_y_bottom;              /*!< \brief  Number of elements added at each side in the bottom of y dimension of the input. */
    vx_uint8   accumulator_bits;          /*!< \brief  Is the total number of bits used during intermediate accumulation. */
    vx_enum    overflow_policy;           /*!< \brief  A VX_TYPE_ENUM of the vx_convert_policy_e enumeration. */
    vx_enum    rounding_policy;           /*!< \brief  A VX_TYPE_ENUM of the vx_round_policy_e enumeration. */
    vx_enum    down_scale_size_rounding;  /*!< \brief  Rounding method for calculating output dimensions. See vx_convolutional_network_rounding_type_e */
    vx_bool    enable_relu;               /*!< \brief  Enable Relu layer function or not. */
    vx_enum    pool_type;                 /*!< \brief  neither max pooling nor average pooling, disable pooling function (see vx_convolutional_network_pooling_type_e). */
    vx_uint32  pool_size_x;               /*!< \brief  Size of the pooling region in the x dimension */
    vx_uint32  pool_size_y;               /*!< \brief  Size of the pooling region in the y dimension. */
    vx_enum    pad_mode;                  /*!< \brief  A VX_TYPE_ENUM of the <tt> \ref vx_pad_mode_e </tt> enumeration. */
    vx_scalar  pad_const;                 /*!< \brief  The order const value if setting pad mode to const, the const value is base value, not quantized value. */
} vx_nn_convolution_relu_pooling_params_t, * vx_nn_convolution_relu_pooling_params;

/*! \brief Extended input parameter for a convolutionReluPooling2 operation.
 * \ingroup group_cnn
 *\version 0.3
 */
typedef struct _vx_nn_convolution_relu_pooling_params_ext_t
{
    vx_nn_convolution_relu_pooling_params_t base;  /*!< \brief convolution relu pooling params <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt> */
    vx_uint32       stride_x;       /*!< \brief  skip x jump for down scale.  */
    vx_uint32       stride_y;       /*!< \brief  skip y jump for down scale.  */
} vx_nn_convolution_relu_pooling_params_ext_t, * vx_nn_convolution_relu_pooling_params_ext;

/*! \brief The 2nd version of extended input parameter for a convolutionReluPooling2 operation.
 *\ingroup group_cnn
 *\version 0.4
 */
typedef struct _vx_nn_convolution_relu_pooling_params_ext2_t
{
    vx_nn_convolution_relu_pooling_params_ext_t ext;  /*!< \brief convolution relu pooling params <tt>\ref vx_nn_convolution_relu_pooling_params__ext_t</tt> */
    vx_int32        depth_multiplier; /*!< \brief  specifying the depthwise multiplier for depthwise convolution.  */
    vx_enum         src_rank_mode; /*!< \brief source rank mode A VX_TYPE_ENUM of the <tt> \ref vx_tensor_rank_type_e </tt> enumeration. */
    vx_enum         convert_dst_format;    /*!< \brief The convert target format. */
} vx_nn_convolution_relu_pooling_params_ext2_t, * vx_nn_convolution_relu_pooling_params_ext2;

/*! \brief [Graph] Creates a Convolutional Network Convolution and Activation(Relu) and Pooling Layer Node, this fucntion match kronos NN Extension 1.2 verion.
 * \details This function implement Convolutional Network Convolution and Activation(Relu) and Pooling layer.
 *  For fixed-point data types, a fixed point calculation is performed with round and saturate according to the number of accumulator bits. The number of the accumulator bits are implementation defined,
 * and should be at least 16.\n
 * round: rounding according the <tt>vx_round_policy_e</tt> enumeration. \n
 * saturate: A saturation according the <tt>vx_convert_policy_e</tt> enumeration.
 * The following equation is implemented: \n
 * \f$ outputs[j,k,i] = saturate(round(\sum_{l} (\sum_{m,n} inputs[j-m,k-n,l] \times weights[m,n,l,i])+biasses[j,k,i])) \f$\n
 * Where \f$m,n\f$ are indexes on the convolution matrices. \f$ l\f$ is an index on all the convolutions per input.\f$ i\f$ is an index per output.
 * \f$ j,k \f$ are the inputs/outputs spatial indexes.
 * Convolution is done on the width and height dimensions of the <tt>\ref vx_tensor</tt>. Therefore, we use here the term x for index along the width dimension and y for index along the height dimension.\n
 * before the Convolution is done, a padding with zeros of the width and height input dimensions is performed.
 * Then down scale is done by picking the results according to a skip jump. The skip in the x and y is determined by the output size dimensions.
 * The relation between input to output is as follows: \n
 * \f$ width_{output} = round(\frac{(width_{input} + paddingleft_x + paddingright_x - kernel_x - (kernel_x -1) * dilation_x)}{skip_x} + 1) \f$\n
 * and \n
 * \f$ height_{output} = round(\frac{(height + paddingtop_y + paddingbottom_y - kernel_y - (kernel_y -1) * dilation_y)}{skip_y} + 1) \f$\n
 * where \f$width\f$ is the size of the input width dimension. \f$height\f$ is the size of the input height dimension.
 * \f$width_{output}\f$ is the size of the output width dimension. \f$height_{output}\f$ is the size of the output height dimension.
 * \f$kernel_x\f$ and \f$kernel_y\f$ are the convolution sizes in width and height dimensions.
 * skip is calculated by the relation between input and output.
 * rounding is done according to <tt>\ref vx_convolutional_network_rounding_type_e</tt>.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data. 3 lower dimensions represent a single input, all following dimensions represent number of batches, possibly nested.
 * The dimension order is [width, height, #IFM, #batches]. \n
 * \param [in] weights_biases [static] Point to WeightBiasesParameter data, vx_weights_biases_parameter is an opaque reference.
 * \param [in] convolution_relu_pooling_params [static] Pointer to parameters of type <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt>
 * \param [in] size_of_convolution_relu_pooling_params [static] Size in bytes of convolution_relu_pooling_params.
 * \param [out] outputs The output tensor data. Output will have the same number and structure of dimensions as input.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_node VX_API_CALL vxConvolutionReluPoolingLayer2(
    vx_graph                    graph,
    vx_tensor                   inputs,
    vx_weights_biases_parameter weights_biases,
    const vx_nn_convolution_relu_pooling_params_t * convolution_relu_pooling_params,
    vx_size                     size_of_convolution_relu_pooling_params,
    vx_tensor                   outputs);

/*! \brief The optimization direvative for weights_biases_parameter create.
 * \ingroup group_cnn
 */
typedef struct _vx_weights_biases_parameter_optimizations_t {
    vx_int8  zrl;             /*!< \brief The zero run length. Set negtive value to disable*/
    vx_enum  outputFormat;    /*!< \brief The output format. */
    vx_int32 inputZeroPoint;  /*!< \brief  zero point of input. A 32 bit integer, in range [0, 255], Set zero value to disable */
} vx_weights_biases_parameter_optimizations_t;

typedef struct _vx_weights_biases_parameter_optimizations_ext_t {
    vx_int8  zrl;             /*!< \brief The zero run length. Set negtive value to disable*/
    vx_enum  outputFormat;    /*!< \brief The output format. */
    vx_int32 inputZeroPoint;  /*!< \brief  zero point of input. A 32 bit integer, in range [0, 255], Set zero value to disable */
    vx_uint32 num_of_input_dims; /*< \brief The input dimesion number*/
    vx_uint32 num_of_output_dims; /*!< \brief The output dimesion number*/
} vx_weights_biases_parameter_optimizations_ext_t;

/*!
 * \brief Creates a reference to a vx_weights_biases_parameter opaque object.
 *
 * \param [in] layer_type                The network type of objects to hold. Types allowed are:
 *                                           \arg VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER for convolution layer.
 *                                           \arg VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER for fullyconnected layer.
 * \param [in] num_of_dims               The dimention number of input & output image tensor.
 * \param [in] inputs_dims               The input tensor's dimension size.
 * \param [in] pad_x                     The number of elements subtracted at each side in the x dimension of the input.
 * \param [in] pad_y                     The number of elements subtracted at each side in the y dimension of the input.
 * \param [in] pooling_size_x            The size of the pooling region in the x dimension, 0 means no pooling operation.
 * \param [in] pooling_size_y            The size of the pooling region in the y dimension, 0 means no pooling operation.
 * \param [in] down_scale_size_rounding  A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
 * \param [in] convolution_outputs_dims  The output's dimension size after covolution operation.
 * \param [in] pool_outputs_dims         The output's dimension size after pooling operation.
 * \param [in] optimizations             A optional param for <tt>\ref vx_weights_biases_parameter_optimizations_t</tt>.
 * \param [in] weights                   The weights tensor which need be compressed.
 * \param [in] biases                    The biases tensor which need be compressed.
 *
 * \returns An opaque vx_weights_biases_parameter reference with compressed kernel data. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL
vxCreateWeightsBiasesParameterFromTensors(
    vx_enum layer_type,
    vx_uint32 num_of_dims,
    vx_uint32 * inputs_dims,
    vx_uint32 pad_x,
    vx_uint32 pad_y,
    vx_uint32 pooling_size_x,
    vx_uint32 pooling_size_y,
    vx_enum down_scale_size_rounding,
    vx_uint32 * convolution_outputs_dims,
    vx_uint32 * pool_outputs_dims,
    vx_weights_biases_parameter_optimizations_t *optimizations,
    vx_tensor weights,
    vx_tensor biases);

/*!
 * \brief Creates a reference to an opaque vx_weights_biases_parameter object.
 *
 * \param [in] layer_type                              The network type of objects to hold. Types allowed are:
 *                                                         \arg VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER for convolution layer.
 *                                                         \arg VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER for fullyconnected layer.
 * \param [in] num_of_dims                             The dimention number of input & output image tensor.
 * \param [in] inputs_dims                             The input tensor's dimension size.
 * \param [in] convolution_outputs_dims                The output's dimension size after covolution operation.
 * \param [in] pool_outputs_dims                       The output's dimension size after pooling operation.
 * \param [in] output_format                           The output tensor element type.
 * \param [in] convolution_relu_pooling_params         The convolution_relu_pooling_params Pointer to parameters of type <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt>
 * \param [in] size_of_convolution_relu_pooling_params The size in bytes of convolution_relu_pooling_params.
 * \param [in] optimizations                           A optional param for <tt>\ref vx_weights_biases_parameter_optimizations_t</tt>.
 * \param [in] weights                                 The weights tensor which need be compressed.
 * \param [in] biases                                  The biases tensor which need be compressed.
 *
 * \returns An opaque vx_weights_biases_parameter reference with compressed kernel data. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL vxCreateWeightsBiasesParameterFromTensors2(
    vx_enum     layer_type,
    vx_uint32   num_of_dims,
    vx_uint32 * inputs_dims,
    vx_uint32 * convolution_outputs_dims,
    vx_uint32 * pool_outputs_dims,
    vx_enum     output_format,
    const vx_nn_convolution_relu_pooling_params convolution_relu_pooling_params,
    vx_size size_of_convolution_relu_pooling_params,
    vx_weights_biases_parameter_optimizations_t *optimizations,
    vx_tensor   weights,
    vx_tensor   biases);

/*!
 * \brief Creates a reference to an opaque vx_weights_biases_parameter object.
 *
 * \param [in] layer_type                              The network type of objects to hold. Types allowed are:
 *                                                         \arg VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER for convolution layer.
 *                                                         \arg VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER for fullyconnected layer.
 * \param [in] inputs_dims                             The input tensor's dimension size.
 * \param [in] convolution_outputs_dims                The output's dimension size after covolution operation.
 * \param [in] pool_outputs_dims                       The output's dimension size after pooling operation.
 * \param [in] convolution_relu_pooling_params         The convolution_relu_pooling_params Pointer to parameters of type <tt>\ref vx_nn_convolution_relu_pooling_params_t</tt>
 * \param [in] size_of_convolution_relu_pooling_params The size in bytes of convolution_relu_pooling_params.
 * \param [in] optimizations                           A optional param for <tt>\ref vx_weights_biases_parameter_optimizations_t</tt>.
 * \param [in] size_of_optimizations                   The size in bytes of optimizations.
 * \param [in] weights                                 The weights tensor which need be compressed.
 * \param [in] biases                                  The biases tensor which need be compressed.
 *
 * \returns An opaque vx_weights_biases_parameter reference with compressed kernel data. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL vxCreateWeightsBiasesParameterFromTensors3(
    vx_enum     layer_type,
    vx_uint32 * inputs_dims,
    vx_uint32 * convolution_outputs_dims,
    vx_uint32 * pool_outputs_dims,
    const vx_nn_convolution_relu_pooling_params convolution_relu_pooling_params,
    vx_size size_of_convolution_relu_pooling_params,
    vx_weights_biases_parameter_optimizations_t *optimizations,
    vx_size size_of_optimizations,
    vx_tensor   weights,
    vx_tensor   biases);

/*! \brief Releases the OpenVX object vx_weights_biases_parameter.
 * \param [in] weights_bias The pointer to the reference to the vx_weights_biases_parameter.
 * \post After returning from this function the reference is zeroed.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE If weights_bias is not a <tt> vx_weights_biases_parameter</tt>.
 * \pre <tt>\ref vxCreateWeightsBiasesParameterFromTensors / vxCreateWeightsBiasesParameterFromTensors2/ vxCreateWeightsBiasesParameter / vxCreateWeightsBiasesParameterFromStream</tt>
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseWeightsBiasesParameter(vx_weights_biases_parameter *weights_bias);

/*!
 * \brief Creates a reference to an vx_weights_biases_parameter object.
 * \param [in] context                   The OpenVX context object.
 * \param [in] layer_type                The network type of objects to hold. Types allowed are:
 *                                           \arg VX_CONVOLUTIONAL_NETWORK_CONVOLUTION_LAYER for convolution layer.
 *                                           \arg VX_CONVOLUTIONAL_NETWORK_FULLYCONNECTED_LAYER for fullyconnected layer.
 * \param [in] num_of_dims               The dimention number of input & output image tensor.
 * \param [in] inputs_dims               The input tensor's dimension size.
 * \param [in] pad_x                     The number of elements subtracted at each side in the x dimension of the input.
 * \param [in] pad_y                     The number of elements subtracted at each side in the y dimension of the input.
 * \param [in] pooling_size_x            The size of the pooling region in the x dimension, 0 means no pooling operation.
 * \param [in] pooling_size_y            The size of the pooling region in the y dimension, 0 means no pooling operation.
 * \param [in] down_scale_size_rounding  A <tt> VX_TYPE_ENUM</tt> of the <tt> vx_round_policy_e</tt> enumeration.
 * \param [in] convolution_outputs_dims  The output's dimension size after covolution operation.
 * \param [in] pool_outputs_dims         The output's dimension size after pooling operation.
 * \param [in] weights_num_of_dims       The dimention number of weights tensor.
 * \param [in] weights_dims              The dimention size of weights tensor.
 * \param [in] weights_data_format       The format of weights tensor.
 * \param [in] weights_fixed_point_pos   The fixed point position when the weights element type is int16/int8, if 0 calculations are performed in integer math.
 * \param [in] biases_num_of_dims        The dimention number of biases tensor.
 * \param [in] biases_dims               The dimention size of biases tensor.
 * \param [in] biases_data_format        The format of biases tensor.
 * \param [in] biases_fixed_point_pos    The fixed point position when the biases element type is int16/int8, if 0 calculations are performed in integer math.
 * \param [in] raw_data_size             The data size of compressed data.
 *
 * \returns A weightsbiases reference without compressed kernel data <tt>vx_weights_biases_parameter</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL
vxCreateWeightsBiasesParameter(
    vx_context context,
    vx_enum layer_type,
    vx_uint32 num_of_dims,
    vx_uint32 * inputs_dims,
    vx_uint32 pad_x,
    vx_uint32 pad_y,
    vx_uint32 pooling_size_x,
    vx_uint32 pooling_size_y,
    vx_enum down_scale_size_rounding,
    vx_uint32 * convolution_outputs_dims,
    vx_uint32 * pool_outputs_dims,
    vx_uint32 weights_num_of_dims,
    vx_uint32 * weights_dims,
    vx_enum weights_data_format,
    vx_int8 weights_fixed_point_pos,
    vx_uint32 biases_num_of_dims,
    vx_uint32 * biases_dims,
    vx_enum biases_data_format,
    vx_int8 biases_fixed_point_pos,
    vx_uint32 raw_data_size
    );

/*!
 * \brief Creates a stream buffer that contain an opaque vx_weights_biases_parameter object info
 *
 * \param [in] context                     The reference to the overall Context.
 * \param [in] weights_biases_parameter    The stream buffer which generated by vxWeightsBiasesParameterToStream.
 * \param [out] weights_biases_stream_size The size of the stream buffer.
 * \param [in] onlyHeaderStream            If only header stream, will not save compressed data to stream buffer.
 *
 * \returns A stream buffer.
 * returns VX_NULL if any errors.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_uint32* VX_API_CALL
vxWeightsBiasesParameterToStream(
    vx_context context,
    vx_weights_biases_parameter weights_biases_parameter,
    vx_uint32 * weights_biases_stream_size,
    vx_bool onlyHeaderStream
);

/*!
 * \brief Create a reference to an vx_weights_biases_parameter object from a buffer
 *
 * \param [in] context               The reference to the overall Context.
 * \param [in] weights_biases_stream The stream buffer which generated by vxWeightsBiasesParameterToStream.
 *
 * \returns A weightsbiases reference with compressed kernel data <tt>vx_weights_biases_parameter</tt>.
 * Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 *
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_weights_biases_parameter VX_API_CALL
vxCreateWeightsBiasesParameterFromStream (
    vx_context context,
    vx_uint32 * weights_biases_stream
);

/*! \brief Releases the stream buffer which generated by vxWeightsBiasesParameterToStream.
 * \param [in] weights_biases_stream The pointer to the buffer.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \pre <tt>\ref vxWeightsBiasesParameterToStream</tt>
 * \ingroup group_cnn
 */
VX_API_ENTRY vx_status VX_API_CALL vxFreeWeightsBiasesParameterStream(
    vx_uint32 *weights_biases_stream
);

#ifdef __cplusplus
}
#endif


#endif
