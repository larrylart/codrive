////////////////////////////////////////////////////////////////////
// base class for object detection with google's coral tpu
// Created by: Larry Lart
////////////////////////////////////////////////////////////////////
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <regex>

#include "lib/edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

// main header include
#include "tpu_worker.h"

// templates
template<typename T> T* TensorData(TfLiteTensor* tensor);

////////////////////////////////////////////////////////////////////
template<> float* TensorData(TfLiteTensor* tensor) 
{
    int nelems = 1;
    for (int i = 1; i < tensor->dims->size; i++) nelems *= tensor->dims->data[i];
    switch (tensor->type) 
	{
        case kTfLiteFloat32:
            return tensor->data.f;
        default:
			std::cerr << "Should not reach here!" << std::endl;
    }
    return nullptr;
}

////////////////////////////////////////////////////////////////////
template<> uint8_t* TensorData(TfLiteTensor* tensor) 
{
    int nelems = 1;
    for (int i = 1; i < tensor->dims->size; i++) nelems *= tensor->dims->data[i];
    switch (tensor->type) 
	{
        case kTfLiteUInt8:
            return tensor->data.uint8;
        default:
			std::cerr << "Should not reach here!" << std::endl;
    }
    return nullptr;
}	

////////////////////////////////////////////////////////////////////
CTPUWorker::CTPUWorker( int nDeviceId, const char* model_path, const char* model_labels ) 
{ 
	m_nDeviceId = nDeviceId;
	m_edgetpuContext = nullptr;	
	// globals for video resize
	m_s_size = 	CORAL_TPU_IMAGE_WIDTH;
	m_x_start = 0;
	m_y_start = 0;
	// flags
	m_hasImage = false;
	m_hasProc = false;	
	m_isReady = false;
	// other
	m_inferenceTime = 0;

	/* test enum devices currently not working
	const auto& available_tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
	/*if( available_tpus.size() < 2 ) 
	{
		std::cerr << "This example requires two Edge TPUs to run." << std::endl;
		//return(0);
	}*/
	//int ntpu = (int) available_tpus.size();
	//printf( "avail tpus=%d\n", ntpu );	*/
  
	// load from file
	m_tfModel = tflite::FlatBufferModel::BuildFromFile( model_path );
	if (m_edgetpuContext == nullptr) 
	{
		m_edgetpuContext = edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext().release();
	}	
	
	// build the interpreter
	m_tfInterpreter = BuildEdgeTpuInterpreter(*m_tfModel, m_edgetpuContext);
	
	// read labels
	if( !LoadClassLabels(model_labels) ) 
	{
		printf( "ERROR :: Failed to read labels file\n" );
	}	
}

////////////////////////////////////////////////////////////////////
CTPUWorker::~CTPUWorker( )
{		

}

////////////////////////////////////////////////////////////////////
bool CTPUWorker::LoadClassLabels(const char* file_name) 
{
    std::ifstream file(file_name);
	//std::ifstream file("data/coco_labels.txt");
    if (!file) 
	{
        printf( "ERROR :: Failed to open file\n" );
        return( false );
    }
	const std::regex label_regex("([0-9]+)[\\ \\t]+(.*?)$");
    std::string line;
    while (std::getline(file, line)) 
	{
		std::smatch m;
		if( std::regex_match(line, m, label_regex) ) 
		{
			if( m.size() == 3 ) 
			{
				std::string s_idx = m[1];
				int n_idx = std::atoi( s_idx.c_str() );
				m_classNames[n_idx] = m[2];
				//printf( "LABEL=%d,STR=%s\n", n_idx, m_classNames[n_idx].c_str() );
			}
        }				
	}
	
    return( true );
}
	
////////////////////////////////////////////////////////////////////
std::string CTPUWorker::GetClassName( int nObjId )
{
	std::string strName = m_classNames[nObjId];	
	return(strName);
}

////////////////////////////////////////////////////////////////////
std::unique_ptr<tflite::Interpreter> CTPUWorker::BuildEdgeTpuInterpreter( const tflite::FlatBufferModel& model, edgetpu::EdgeTpuContext* edgetpu_context ) 
{
	tflite::ops::builtin::BuiltinOpResolver resolver;
	resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
	std::unique_ptr<tflite::Interpreter> interpreter;
	if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) 
	{
		std::cerr << "Failed to build interpreter." << std::endl;
	}
	// Bind given context with interpreter.
	interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
	interpreter->SetNumThreads(2);
	if (interpreter->AllocateTensors() != kTfLiteOk) 
	{
		std::cerr << "Failed to allocate tensors." << std::endl;
	}
	
	// Find output tensors.
	if (interpreter->outputs().size() != 4) 
	{
		std::cerr << "Graph needs to have 4 and only 4 outputs!" << std::endl;
	}	
	
	m_inputTensor = interpreter->tensor(interpreter->inputs()[0]);
	
	m_outputLocations = interpreter->tensor(interpreter->outputs()[0]);
	m_outputClasses = interpreter->tensor(interpreter->outputs()[1]);
	m_outputScores = interpreter->tensor(interpreter->outputs()[2]);
	m_numDetections = interpreter->tensor(interpreter->outputs()[3]);
	
	return( interpreter );
}

////////////////////////////////////////////////////////////////////
bool CTPUWorker::CopyCrop( const cv::Mat& inputMat )
{
//	printf("p1 rows=%d, cols=%d\n", inputMat.rows, inputMat.cols );
	if( inputMat.rows <=0 || inputMat.cols <= 0 ) return( false );
	
	// get the mid of orig image
    int _mid_row = inputMat.rows / 2.0;
    int _mid_col = inputMat.cols / 2.0;	
	// get the smalest
	m_s_size = (inputMat.rows > inputMat.cols) ? inputMat.cols : inputMat.rows;
	if(inputMat.rows > m_s_size)
		m_y_start = _mid_row - m_s_size/2;
	else
		m_x_start = _mid_col - m_s_size/2;
	
	cv::Rect roi(m_x_start, m_y_start, m_s_size, m_s_size);

	// copy network area localy - should clone ? faster?
	inputMat(roi).copyTo(orig_image_mat);

    return true;
}

////////////////////////////////////////////////////////////////////
bool CTPUWorker::FormatImageSize( cv::Mat& preprocessed_image_mat )
{
//	printf("p1 rows=%d, cols=%d\n", orig_image_mat.rows, orig_image_mat.cols );
	if( orig_image_mat.rows <=0 || orig_image_mat.cols <= 0 ) return( false );

	double _ratio = (double)CORAL_TPU_IMAGE_WIDTH / (double)m_s_size;
	
//	printf("_ratio=%.4f m_x_start=%d, m_y_start=%d, m_s_size=%d\n", _ratio, m_x_start, m_y_start, m_s_size );

    cv::resize(orig_image_mat, preprocessed_image_mat, cv::Size(), _ratio, _ratio, cv::INTER_AREA);

    return true;
}


////////////////////////////////////////////////////////////////////
// resize by smallest dimension and crop the center part 
// todo: first crop rectagular from center than resize to net
////////////////////////////////////////////////////////////////////
bool CTPUWorker::FormatImage( const cv::Mat& src_image_mat, cv::Mat& preprocessed_image_mat )
{
//	double t = (double)getTickCount();
//	printf("p1 rows=%d, cols=%d\n", src_image_mat.rows, src_image_mat.cols );
	if( src_image_mat.rows <=0 || src_image_mat.cols <= 0 ) return( false );
	
	// get the mid of orig image
    int _mid_row = src_image_mat.rows / 2.0;
    int _mid_col = src_image_mat.cols / 2.0;	
	// get the smalest
	m_s_size = (src_image_mat.rows > src_image_mat.cols) ? src_image_mat.cols : src_image_mat.rows;
	if(src_image_mat.rows > m_s_size)
		m_y_start = _mid_row - m_s_size/2;
	else
		m_x_start = _mid_col - m_s_size/2;
		// custom net size
		//m_x_start = _mid_col - src_image_mat.cols/2;
	
	//cv::Rect roi(m_x_start, m_y_start, src_image_mat.cols/2, src_image_mat.rows/2);
	cv::Rect roi(m_x_start, m_y_start, m_s_size, m_s_size);

	double _ratio = (double)CORAL_TPU_IMAGE_WIDTH / (double)m_s_size;

    cv::resize(src_image_mat(roi), preprocessed_image_mat, cv::Size(), _ratio, _ratio, cv::INTER_AREA);
	
	// debug
	//cv::imwrite("img_cut.jpg", src_image_mat(roi));
	//cv::imwrite("img_cut_resize.jpg", preprocessed_image_mat);
	
    return true;
}

////////////////////////////////////////////////////////////////////
void CTPUWorker::setImageFlag( bool value )
{
//	wxMutexLocker lock(*m_pMutexPush);
	m_hasImage = value;
}

////////////////////////////////////////////////////////////////////
void CTPUWorker::setProcFlag( bool value )
{
//	wxMutexLocker lock(*m_pMutexProcessing);
	m_hasProc = value;
}

////////////////////////////////////////////////////////////////////
bool CTPUWorker::isImage()
{
//	wxMutexLocker lock(*m_pMutexPush);
	return(m_hasImage);
}

////////////////////////////////////////////////////////////////////
bool CTPUWorker::PushImage( cv::Mat* inputMat )
{
	// only allow image push when no image in or no processing
//	if( m_hasImage || m_hasProc || !m_isReady ) return(false);
	
	m_pInputFrame = inputMat;
	
	m_hasImage = true;
	m_hasProc = false;
	
//	m_pMutexImage->Lock();
//	m_pConditionImage->Signal();
//	m_pMutexImage->Unlock();
	
	return( true );	
}

////////////////////////////////////////////////////////////////////
bool CTPUWorker::IsProcessing()
{
	if( m_hasImage || m_hasProc )
		return( true );
	else
		return( false );
}

////////////////////////////////////////////////////////////////////
bool CTPUWorker::GetInference( std::vector<networkResult> &vectNetworkResult )
{
//	if( !m_hasProc ) return(false);
	m_hasProc = true;
			
	// if processing done and has results return copy
	if( m_hasProc && m_vectNetworkResult.size() > 0 )
	{
		vectNetworkResult = m_vectNetworkResult;
		m_hasProc = false;
		m_hasImage = false;
		return( true );
		
	} else if( m_hasProc && m_vectNetworkResult.size() <= 0 )
	{
		m_hasProc = false;
		m_hasImage = false;
		// return true as processing was done but nothing was found
		return( true );		
	}
	
	return(false);
}

// set inference result in local format
////////////////////////////////////////////////////////////////////
int CTPUWorker::SetInferenceResult()
{	
	// set defaults
	networkResult _retNet = {-1, -1, -1, -1, cv::Rect(0,0,0,0), -1, -1.0};
		
    float maxResult = 0.0;
    int maxIndex = -1;
	int bi = 0;
	
	// we should only clear if there are any results
	m_vectNetworkResult.clear();
		
	int rows = m_s_size; //m_pInputFrame->rows;
	int cols = m_s_size; //m_pInputFrame->cols;
	
	const float* detection_locations = TensorData<float>(m_outputLocations);
	const float* detection_classes = TensorData<float>(m_outputClasses);
	const float* detection_scores = TensorData<float>(m_outputScores);
	const int num_detections = *TensorData<float>(m_numDetections);
//	printf("DEBUG :: number of obj=%d\n", num_detections );
	
	for( int d = 0; d < num_detections; d++ ) 
	{
		//const std::string cls = labels_[detection_classes[d]];
		int obj_id = (int) detection_classes[d];
		//const std::string cls = labels_[detection_classes[d]];
		const std::string cls = GetClassName(obj_id);
		//const std::string cls = "label";
		
		const float score = detection_scores[d];
		const int ymin = detection_locations[4 * d] * rows;
		const int xmin = detection_locations[4 * d + 1] * cols;
		const int ymax = detection_locations[4 * d + 2] * rows;
		const int xmax = detection_locations[4 * d + 3] * cols;
				
//		printf( "Found objs=%d : score=%.4f : oid=%d:%s (%d,%d,%d,%d)\n", num_detections, score, obj_id, cls.c_str(), xmin, ymin, xmax, ymax );
		
		// ignore weak detections
		if (score < .03f) 
		{
			//cout << "Ignore detection " << d << " of '" << cls << "' with score " << score
			//	<< " @[" << xmin << "," << ymin << ":" << xmax << "," << ymax << "]" << std::endl;
//			printf( "DEBUG :: score low\n" );

		// :: set strong detections to output
		} else 
		{
			//cout << "Detected " << d << " of '" << cls << "' with score " << score
			//	<< " @[" << xmin << "," << ymin << ":" << xmax << "," << ymax << "]" << std::endl;
//			printf( "DEBUG :: score high\n" );
			
			_retNet.objectClass = (int) obj_id;
			_retNet.confidence = (float) score;		
			
			_retNet.left = m_x_start + xmin;
			_retNet.top = m_y_start + ymin;
			_retNet.right = m_x_start + xmax;
			_retNet.bottom = m_y_start + ymax;	
			_retNet.roi.x = _retNet.left;
			_retNet.roi.y = _retNet.top;
			_retNet.roi.width = _retNet.right - _retNet.left;
			_retNet.roi.height = _retNet.bottom - _retNet.top;
			strcpy( _retNet.objectName, cls.c_str() );
			
			m_vectNetworkResult.push_back(_retNet);
		
		}
	}	
	
	return( num_detections );
}

////////////////////////////////////////////////////////////////////
// Runs an inference and outputs result
////////////////////////////////////////////////////////////////////
bool CTPUWorker::RunInferenceOnImage()
{
	// check in no image return
	if( !m_hasImage ) return( false );
	// process input image
	//double t = (double)getTickCount();
	if( !FormatImage(*m_pInputFrame, preprocessed_image_mat) ) return(false);
	//t = 1000*((double)getTickCount() - t)/getTickFrequency();
	//cout << "Time for FormatImage =" << t << " milliseconds."<< endl;			
		
	bool _retCode;
	
	// check if processed image valid
    if (preprocessed_image_mat.rows != CORAL_TPU_IMAGE_HEIGHT ||
        preprocessed_image_mat.cols != CORAL_TPU_IMAGE_WIDTH) 
	{
        cout << "Error - preprocessed image is unexpected size!" << endl;
		//networkResults error = {-1, -1, "-1", -1};
		return( false );
    }	
		
	int _width = m_inputTensor->dims->data[2];
	int _height = m_inputTensor->dims->data[1];
    int _input_channels = m_inputTensor->dims->data[3];	
		
	switch (m_inputTensor->type) 
	{
		case kTfLiteFloat32:
		{
			float* dst = TensorData<float>(m_inputTensor);
			const int row_elems = _width * _input_channels;
			for (int row = 0; row < _height; row++) 
			{
				const uchar* row_ptr = preprocessed_image_mat.ptr(row);
				for (int i = 0; i < row_elems; i++) 
				{
					dst[i] = (row_ptr[i] - TPU_IMAGE_MEAN) / TPU_IMAGE_STD;
				}
				dst += row_elems;
			}
		}
		break;
		
		case kTfLiteUInt8:
		{
			uint8_t* dst = TensorData<uint8_t>(m_inputTensor);
			const int row_elems = _width * _input_channels;
			for (int row = 0; row < _height; row++) 
			{
				memcpy(dst, preprocessed_image_mat.ptr(row), row_elems);
				dst += row_elems;
			}
		}
		break;
		
		default:
		{
			printf( "ERROR :: unknown tensor type!\n" );
			return( false );
		}
	}		
	
	const auto& start_time = std::chrono::steady_clock::now();
	//printf( "DEBUG :: got before invoke\n" );
	if( m_tfInterpreter->Invoke() != kTfLiteOk ) 
	{
		printf( "ERROR :: failed to run inference\n" );
		return( false );
	}
	//printf( "DEBUG :: got after invoke\n" );
	std::chrono::duration<double> time_span = std::chrono::steady_clock::now() - start_time;
	m_inferenceTime = time_span.count();
	//std::cout << "inferences time: "  << time_span.count() << " seconds." << std::endl;	
	
	// now set my inference results
	SetInferenceResult();
		
	return( true );
}

