#ifndef _VSI_NN_LOG_H
#define _VSI_NN_LOG_H
#include <stdio.h>

#if defined(__cplusplus)
extern "C"{
#endif

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

typedef enum _vsi_nn_log_level_e
{
    VSI_NN_LOG_UNINIT = -1,
    VSI_NN_LOG_CLOSE,
    VSI_NN_LOG_ERROR,
    VSI_NN_LOG_WARN,
    VSI_NN_LOG_INFO,
    VSI_NN_LOG_DEBUG
}vsi_nn_log_level_e;

#define VSI_NN_MAX_DEBUG_BUFFER_LEN 1024
#define VSILOGE( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_ERROR, "E [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define VSILOGW( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_WARN,  "W [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define VSILOGI( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_INFO,  "I [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define VSILOGD( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_DEBUG, "D [%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define _LOG_( fmt, ... ) \
    vsi_nn_LogMsg(VSI_NN_LOG_DEBUG, "[%s:%d]" fmt, __FUNCTION__, __LINE__, ##__VA_ARGS__)

OVXLIB_API void vsi_nn_LogMsg
    (
    vsi_nn_log_level_e level,
    const char *fmt,
    ...
    );

#if defined(__cplusplus)
}
#endif

#endif

