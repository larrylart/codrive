////////////////////////////////////////////////////////////////////
// base class interface for object detection with Khadas's AML NPU
// Created by: Larry Lart
////////////////////////////////////////////////////////////////////
#ifndef _AMLWORKER_H
#define _AMLWORKER_H

// base
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core_c.h>

// other locals
// aml includes
#include "vsi_nn_pub.h"
//#include "vnn_global.h"
//#include "vnn_pre_process.h"
//#include "vnn_post_process.h"

// load acuity generate net 
#ifdef __cplusplus
extern "C"{
#endif

// load acuity generated net code
#include "vnn_mobilenetssdv2a.h"

#ifdef __cplusplus
}
#endif

// debug
#define _DEBUG_FPS	0

// defines
#define NN_IMAGE_MEAN 128.0f
#define NN_IMAGE_STD 128.0f
// tpu image resolution
#define NN_INPUT_IMAGE_WIDTH 300
#define NN_INPUT_IMAGE_HEIGHT 300
#define NN_INPUT_IMAGE_CHANNELS 3

// some image format defines?
/// pixel format definition
typedef enum {
  ///< Y 1 8bpp(Single channel 8bit gray pixels )
  PIX_FMT_GRAY8,
  ///< YUV  4:2:0 12bpp ( 3 channels, one brightness channel, the othe
  /// two for the U component and V component channel, all channels are continuous)
  PIX_FMT_YUV420P,
  ///< YUV  4:2:0 12bpp ( 2 channels, one channel is a continuous
  /// luminance
  /// channel, and the other channel is interleaved as a UV component )
  PIX_FMT_NV12,
  ///< YUV  4:2:0   	12bpp ( 2 channels, one channel is a continuous
  /// luminance
  /// channel, and the other channel is interleaved as a UV component )
  PIX_FMT_NV21,
  ///< BGRA 8:8:8:8 	32bpp ( 4-channel 32bit BGRA pixels )
  PIX_FMT_BGRA8888,
  ///< BGR  8:8:8   	24bpp ( 3-channel 24bit BGR pixels )
  PIX_FMT_BGR888,
  ///< RGBA 8:8:8：8	32bpp ( 4-channel 32bit RGBA pixels )
  PIX_FMT_RGBA8888,
  ///< RGB  8:8:8		24bpp ( 3-channel 24bit RGB pixels )
  PIX_FMT_RGB888
} det_pixel_format;


// nn output scales
#define NN_OUT_X_SCALE	10.0f
#define NN_OUT_Y_SCALE	10.0f
#define NN_OUT_H_SCALE	5.0f  
#define NN_OUT_W_SCALE	5.0f		
	
// network output structure
typedef struct networkResult 
{
	int left;
	int top;
	int right;
	int bottom;
	cv::Rect roi;
	int objectClass;
	float confidence;
	char objectName[64];
	
} networkResult;

// box structure for output
typedef struct{
    float x, y, w, h, prob_obj;
	int classId;
} box;


// namespaces in use
using namespace std;
using namespace cv;

/////////////////////////////////////////////////
class CAMLWorker 
{
// methods
public:
	CAMLWorker( const char* model_path, const char* model_labels );
	~CAMLWorker();

	bool FormatImage( const cv::Mat& src_image_mat, cv::Mat& preprocessed_image_mat );
	
	bool PushImage( cv::Mat* inputMat );
	bool RunInferenceOnImage();
	int SetInferenceResult();
	bool GetInference( std::vector<networkResult> &vectNetworkResult );

	bool IsProcessing();
	
	std::string GetClassName( int nObjId );
	
	// new
	bool LoadClassLabels(const char* file_name);
	
	// mfunc
	int i_width() { return(m_pInputTensor->attr.size[0]); }
	int i_height() { return(m_pInputTensor->attr.size[1]); }
	int i_channels() { return(m_pInputTensor->attr.size[3]); }
			
/////////
// data
public:
	// flags	
	bool	m_hasImage;
	bool	m_hasProc;
	
	// results vector
	std::vector<networkResult>	m_vectNetworkResult;
		
	// globals for video resize
	int m_s_size;
	int m_x_start;
	int m_y_start;
	// image processing vars
	cv::Mat* m_pInputFrame;
	cv::Mat preprocessed_image_mat;
	
	
	// AML NPU STUFF
	vsi_nn_graph_t* m_pNPUGraph;
	
	// IO tensors
	vsi_nn_tensor_t* m_pInputTensor;
	vsi_nn_tensor_t* m_pOutputTensorBox;
	vsi_nn_tensor_t* m_pOutputTensorScore;
	
	// buffers and data
	float* m_pOutputBufferBox;
	float* m_pOutputBufferScore;	
	uint8_t* m_pOutputTensorDataBox;
	uint8_t* m_pOutputTensorDataScore;
	// other dim
    uint32_t sz_boxes;
    uint32_t sz_score;	
	
	// labels
	std::vector<std::string> labels_;
		
	// inference time
	float m_inferenceTime;
		
	// class names/labels
	std::string m_classNames[100];	
	
};

#endif
