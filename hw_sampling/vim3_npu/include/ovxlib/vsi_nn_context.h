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
#ifndef _VSI_NN_CONTEXT_H
#define _VSI_NN_CONTEXT_H

#include "vsi_nn_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VSI_NN_MAX_TARGET_NAME 32
typedef enum _vsi_nn_hw_evis_version_e
{
    VSI_NN_HW_EVIS_NONE,
    VSI_NN_HW_EVIS_1,
    VSI_NN_HW_EVIS_2
}vsi_nn_hw_evis_version_e;

typedef struct _vsi_nn_hw_evis_t
{
    vsi_nn_hw_evis_version_e ver;
}vsi_nn_hw_evis_t;

typedef struct _vsi_nn_hw_config_t
{
    char target_name[VSI_NN_MAX_TARGET_NAME];
    vsi_nn_hw_evis_t evis;
}vsi_nn_hw_config_t;

typedef struct _vsi_nn_context_t
{
    vx_context c;
    vsi_nn_hw_config_t config;
} *vsi_nn_context_t;

OVXLIB_API vsi_nn_context_t vsi_nn_CreateContext
    ( void );

OVXLIB_API void vsi_nn_ReleaseContext
    ( vsi_nn_context_t * ctx );

#ifdef __cplusplus
}
#endif

#endif
