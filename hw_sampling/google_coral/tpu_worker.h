////////////////////////////////////////////////////////////////////
// base class interface for object detection with google's coral tpu
// Created by: Larry Lart
////////////////////////////////////////////////////////////////////
#ifndef _TPUWORKER_H
#define _TPUWORKER_H

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

// tensorflow
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

// other locals
#include "lib/edgetpu.h"

// defines
#define TPU_IMAGE_MEAN 128.0f
#define TPU_IMAGE_STD 128.0f
// tpu image resolution
#define CORAL_TPU_IMAGE_WIDTH 300
#define CORAL_TPU_IMAGE_HEIGHT 300

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


// namespaces in use
using namespace std;
using namespace cv;

/////////////////////////////////////////////////
class CTPUWorker 
{
// methods
public:
	CTPUWorker( int nDeviceId, const char* model_path, const char* model_labels );
	~CTPUWorker();

	
	bool FormatImage( const cv::Mat& src_image_mat, cv::Mat& preprocessed_image_mat );
	bool CopyCrop( const cv::Mat& inputMat );
	bool FormatImageSize( cv::Mat& preprocessed_image_mat );
	
	bool PushImage( cv::Mat* inputMat );
	bool RunInferenceOnImage();
	int SetInferenceResult();
	bool GetInference( std::vector<networkResult> &vectNetworkResult );

	bool IsProcessing();
	bool isImage();
	void setImageFlag( bool value );
	void setProcFlag( bool value );
	
	std::string GetClassName( int nObjId );
	
	// new
	bool LoadClassLabels(const char* file_name);
	
	// mfunc
	int i_height() { return(m_inputTensor->dims->data[1]); }
	int i_width() { return(m_inputTensor->dims->data[2]); }
	int i_channels() { return(m_inputTensor->dims->data[3]); }
	
	std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter( const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context );
	
/////////
// data
public:
	
	bool	m_bLife;
	int		m_bIsProcessing;
	int		m_bIsExit;
	
	bool	m_hasImage;
	bool	m_hasProc;
	bool	m_isReady;
		
	std::vector<networkResult>	m_vectNetworkResult;

	int m_nDeviceId;
		
	// globals for video resize
	int m_s_size;
	int m_x_start;
	int m_y_start;
	// image processing vars
	cv::Mat* m_pInputFrame;
	cv::Mat orig_image_mat;
	cv::Mat preprocessed_image_mat;
	
	// TPU STUFF
	std::unique_ptr<tflite::FlatBufferModel> m_tfModel;
	edgetpu::EdgeTpuContext* m_edgetpuContext;
	std::unique_ptr<tflite::Interpreter> m_tfInterpreter;
	
	std::vector<std::string> labels_;
	TfLiteTensor* m_inputTensor = nullptr;
	TfLiteTensor* m_outputLocations = nullptr;
	TfLiteTensor* m_outputClasses = nullptr;
	TfLiteTensor* m_outputScores = nullptr;
	TfLiteTensor* m_numDetections = nullptr;
		
	// inference time
	float m_inferenceTime;
		
	// class names/labels
	std::string m_classNames[100];	
	
};

#endif
