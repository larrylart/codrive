////////////////////////////////////////////////////////////////////
// base class for object detection with Khadas's AML NPU
// Created by: Larry Lart
////////////////////////////////////////////////////////////////////
#include <math.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <unistd.h>

#define _BASETSD_H

// aml includes
#include "vsi_nn_pub.h"

// local aml libs
#include "nn_anchors.h"

// main header include
#include "aml_worker.h"

////////////////////////////////////////////////////////////////////
static float box_overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

////////////////////////////////////////////////////////////////////
static int score_comparator(const void *pa, const void *pb)
{
    box a = *(box *)pa;
    box b = *(box *)pb;
	if ( a.prob_obj < b.prob_obj ) return(1);
    return(0);
}

////////////////////////////////////////////////////////////////////
static float logistic_activate(float x){return 1./(1. + exp(-x));}

// constructor
////////////////////////////////////////////////////////////////////
CAMLWorker::CAMLWorker( const char* model_path, const char* model_labels ) 
{ 
	int i =0;
	
	// globals for video resize
	m_s_size = 	NN_INPUT_IMAGE_WIDTH;
	m_x_start = 0;
	m_y_start = 0;
	// flags
	m_hasImage = false;
	m_hasProc = false;	
	// other
	m_inferenceTime = 0;

	//////////////////////////////////
	// create net
    m_pNPUGraph = NULL;
		
    m_pNPUGraph = vnn_CreateMobilenetSsdV2a(model_path, NULL);
  
	// get I/O tensors 
	m_pInputTensor = vsi_nn_GetTensor(m_pNPUGraph, m_pNPUGraph->input.tensors[0]);
	m_pOutputTensorBox = vsi_nn_GetTensor(m_pNPUGraph, m_pNPUGraph->output.tensors[0]);
	m_pOutputTensorScore = vsi_nn_GetTensor(m_pNPUGraph, m_pNPUGraph->output.tensors[1]);
	
	// other locals
	m_pOutputBufferBox = NULL;
	m_pOutputBufferScore = NULL;
	m_pOutputTensorDataBox = NULL;
	m_pOutputTensorDataScore = NULL;
    sz_boxes = 1;
    sz_score = 1;	
	
	// init output
    for(i = 0; i < m_pOutputTensorBox->attr.dim_num; i++) sz_boxes *= m_pOutputTensorBox->attr.size[i];
    m_pOutputBufferBox = (float *)malloc(sizeof(float) * sz_boxes);

    for(i = 0; i < m_pOutputTensorScore->attr.dim_num; i++) sz_score *= m_pOutputTensorScore->attr.size[i];
    m_pOutputBufferScore = (float *)malloc(sizeof(float) * sz_score);
	
	// read labels
	if( !LoadClassLabels(model_labels) ) 
	{
		printf( "ERROR :: Failed to read labels file\n" );
	}	
}

// destructor
////////////////////////////////////////////////////////////////////
CAMLWorker::~CAMLWorker( )
{		
	vnn_ReleaseMobilenetSsdV2a( m_pNPUGraph, TRUE );

	free( m_pOutputBufferBox );
	free( m_pOutputBufferScore );
}

////////////////////////////////////////////////////////////////////
bool CAMLWorker::LoadClassLabels(const char* file_name) 
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
std::string CAMLWorker::GetClassName( int nObjId )
{
	std::string strName = m_classNames[nObjId];	
	return(strName);
}

////////////////////////////////////////////////////////////////////
// resize by smallest dimension and crop the center part 
// todo: first crop rectagular from center than resize to net
////////////////////////////////////////////////////////////////////
bool CAMLWorker::FormatImage( const cv::Mat& src_image_mat, cv::Mat& preprocessed_image_mat )
{
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
	
	cv::Rect roi(m_x_start, m_y_start, m_s_size, m_s_size);
	double _ratio = (double)NN_INPUT_IMAGE_WIDTH / (double)m_s_size;
    cv::resize(src_image_mat(roi), preprocessed_image_mat, cv::Size(), _ratio, _ratio, cv::INTER_AREA);	
	
	// debug
	//cv::imwrite("img_cut.jpg", src_image_mat(roi));
	//cv::imwrite("img_cut_resize.jpg", preprocessed_image_mat);
	
    return( true );
}

////////////////////////////////////////////////////////////////////
bool CAMLWorker::PushImage( cv::Mat* inputMat )
{	
	m_pInputFrame = inputMat;
	
	m_hasImage = true;
	m_hasProc = false;
		
	return( true );	
}

////////////////////////////////////////////////////////////////////
bool CAMLWorker::IsProcessing()
{
	if( m_hasImage || m_hasProc )
		return( true );
	else
		return( false );
}

////////////////////////////////////////////////////////////////////
bool CAMLWorker::GetInference( std::vector<networkResult> &vectNetworkResult )
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
int CAMLWorker::SetInferenceResult()
{	
	// set defaults
	networkResult _retNet = {-1, -1, -1, -1, cv::Rect(0,0,0,0), -1, -1.0};		
	int num_detections = 0;
	
	// we should only clear if there are any results
	m_vectNetworkResult.clear();
		
	// resize ration
	double _ratio = (double)m_s_size / (double)NN_INPUT_IMAGE_WIDTH;	
			
	uint32_t i,j,stride;
	vsi_status status = VSI_FAILURE;
	
	//////////////////////////////////////////////
	// :: boxes 	
    stride = vsi_nn_TypeGetBytes(m_pOutputTensorBox->attr.dtype.vx_type);
    m_pOutputTensorDataBox = (uint8_t *)vsi_nn_ConvertTensorToData(m_pNPUGraph, m_pOutputTensorBox);
    for(i = 0; i < sz_boxes; i++)
		status = vsi_nn_DtypeToFloat32(&m_pOutputTensorDataBox[stride * i], &m_pOutputBufferBox[i], &m_pOutputTensorBox->attr.dtype);
	if(m_pOutputTensorDataBox) vsi_nn_Free(m_pOutputTensorDataBox);			
	
	//////////////////////////////////////////////
	// :: score 
    stride = vsi_nn_TypeGetBytes(m_pOutputTensorScore->attr.dtype.vx_type);
    m_pOutputTensorDataScore = (uint8_t *)vsi_nn_ConvertTensorToData(m_pNPUGraph, m_pOutputTensorScore);
    for(i = 0; i < sz_score; i++)
        status = vsi_nn_DtypeToFloat32(&m_pOutputTensorDataScore[stride * i], &m_pOutputBufferScore[i], &m_pOutputTensorScore->attr.dtype);
	if(m_pOutputTensorDataScore) vsi_nn_Free(m_pOutputTensorDataScore);			
		
	///////////////////////////////////////////////
	// :: PROCESS OUTPUT
	box boxes[NN_ANCHORS_NO]; 
	memset(boxes,0,NN_ANCHORS_NO*sizeof(box));		
		
	int index = 0;
    for(i=0; i<sz_score; i++)
    {	
		float logi = (float) 1./(1. + exp(-m_pOutputBufferScore[i]));
		//float logi = logistic_activate(m_pOutputBufferScore[i]);
		if( logi > 0.5 )
		{
			int classId = i % 91;
			int idx = (int) (i / 91);
			if( classId != 0 )
			{
				// location decoding
				float ycenter =     m_pOutputBufferBox[idx*4 + 0] / NN_OUT_Y_SCALE * (float)vect_anchors[idx*4 + 2] + (float)vect_anchors[idx*4 + 0];
				float xcenter =     m_pOutputBufferBox[idx*4 + 1] / NN_OUT_X_SCALE * (float)vect_anchors[idx*4 + 3] + (float)vect_anchors[idx*4 + 1];
				float h       = (float) expf(m_pOutputBufferBox[idx*4 + 2] / NN_OUT_H_SCALE ) * (float)vect_anchors[idx*4 + 2];
				float w       = (float) expf(m_pOutputBufferBox[idx*4 + 3] / NN_OUT_W_SCALE ) * (float)vect_anchors[idx*4 + 3];

				float ymin    = ( ycenter - h * 0.5 ) * NN_INPUT_IMAGE_HEIGHT;
				float xmin    = ( xcenter - w * 0.5 ) * NN_INPUT_IMAGE_WIDTH;
				float ymax    = ( ycenter + h * 0.5 ) * NN_INPUT_IMAGE_HEIGHT;
				float xmax    = ( xcenter + w * 0.5 ) * NN_INPUT_IMAGE_WIDTH;	

				// boxes[index] = get_region_box(predictions, biases, n, box_index, col, row, modelWidth, modelHeight);
				boxes[index].x = xmin;
				boxes[index].y = ymin;
				boxes[index].w = xmax - xmin;
				boxes[index].h = ymax - ymin;
				boxes[index].prob_obj = logi;
				boxes[index].classId = classId;
				
				index++;
				
				//printf("RAW :: %d(%d)(%d): %.6f --- %.6f :::: (%.2f,%.2f,%.2f,%.2f)_(%.2f,%.2f,%.2f,%.2f) \n", i, idx, classId, m_pOutputBufferScore[i], logi, 
				//		(float) vect_anchors[idx*4 + 0], (float)vect_anchors[idx*4 + 1], (float)vect_anchors[idx*4 + 2] , (float)vect_anchors[idx*4 + 3],
				//		//(float) buffer_a[idx*4 + 0], (float)buffer_a[idx*4 + 1], (float)buffer_a[idx*4 + 2] , (float)buffer_a[idx*4 + 3] );
				//		ymin, xmin, ymax, xmax);
			}
		}
	}

	// loop through boxes
	float m_overlap_threshold = 0.4;
	for(i=0; i<index; i++)
	{
		// skip box with prob == 0 (already processed)
		if( boxes[i].prob_obj <= 0.0 ) continue;
		
		for(j=0; j<index; j++)
		{
			// only check same class id as base and not the base or already done
			if( i == j || boxes[j].prob_obj <= 0.0 || 
				boxes[i].classId != boxes[j].classId ) continue;
			
			//////////////////
			// calculate overlap
			box a = boxes[i];
			box b = boxes[j];

			// intersection
			float box_intersection = 0;
			float iw = box_overlap(a.x, a.w, b.x, b.w);
			float ih = box_overlap(a.y, a.h, b.y, b.h);
			if(iw > 0 && ih > 0) box_intersection = iw*ih;
			
			// union
			float box_union = a.w*a.h + b.w*b.h - box_intersection;
			// overlap
			float box_iou = box_intersection/box_union;
			
			// check if overlap if over thresh
			if( box_iou > m_overlap_threshold )
			{
				// now check which has lower score
				if( boxes[i].prob_obj >= boxes[j].prob_obj ) 
				{
					boxes[j].prob_obj = 0;
					
				} else
				{
					// set base to zero and break this loop
					boxes[i].prob_obj = 0;
					break;
				}
			}
			
		}
	}

	// extract output - debug
	/*for(i=0; i<index; i++)
	{
		// skip box with prob == 0 (already processed)
		if( boxes[i].prob_obj <= 0.0 ) continue;
	
		printf("BOX :%d: (%d)(%.6f): (%.2f,%.2f,%.2f,%.2f)\n", 
				i, boxes[i].classId, boxes[i].prob_obj, 
				boxes[i].x, boxes[i].y, boxes[i].w , boxes[i].h);
	}*/
	
	// sort descending
	qsort(boxes, index, sizeof(box), score_comparator);
	//std::sort(boxes, boxes+index, &customer_sorter);
	
	// extract output
	for(i=0; i<index; i++)
	{
		// skip box with prob == 0 (already processed)
		if( boxes[i].prob_obj <= 0.0 ) continue;
	
		int obj_id = (int) boxes[i].classId;
		const std::string cls = GetClassName(obj_id);
		
		const float score = boxes[i].prob_obj;
		const int ymin = boxes[i].y * _ratio;
		const int xmin = boxes[i].x * _ratio;
		const int ymax = (boxes[i].y+boxes[i].h) * _ratio;
		const int xmax = (boxes[i].x+boxes[i].w) * _ratio;
		
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
	
		num_detections++;
		
		// debug
		if( _DEBUG_FPS > 2 )
		{
			printf("BOXS :%d: [%s](%d)(%.6f): (%.2f,%.2f,%.2f,%.2f)\n", 
					i, cls.c_str(), boxes[i].classId, boxes[i].prob_obj, 
					boxes[i].x, boxes[i].y, boxes[i].w , boxes[i].h);
		}
	}
		
	return( num_detections );
}

////////////////////////////////////////////////////////////////////
// Runs an inference and outputs result
////////////////////////////////////////////////////////////////////
bool CAMLWorker::RunInferenceOnImage()
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
    if (preprocessed_image_mat.rows != NN_INPUT_IMAGE_HEIGHT ||
        preprocessed_image_mat.cols != NN_INPUT_IMAGE_WIDTH) 
	{
        cout << "Error - preprocessed image is unexpected size!" << endl;
		//networkResults error = {-1, -1, "-1", -1};
		return( false );
    }	
				
	//////////////////////////
	//copy input data to npu tensor
	vsi_status status = VSI_FAILURE;	
	status = vsi_nn_CopyDataToTensor(m_pNPUGraph, m_pInputTensor, preprocessed_image_mat.data);
	
	// check return
	if (status) 
	{
        cout << "Error - Vsi_nn_CopyDataToTensor fail!" << endl;
		//networkResults error = {-1, -1, "-1", -1};
		return( false );				
	}					
	
	const auto& start_time = std::chrono::steady_clock::now();
	
    // DEBUG :: Verify graph 
	// status = vnn_VerifyGraph(graph);
	// TEST_CHECK_STATUS(status, final);

    // Process graph
	status = vsi_nn_RunGraph(m_pNPUGraph);
    //TEST_CHECK_STATUS( status, final );
	
	//printf( "DEBUG :: got after invoke\n" );
	std::chrono::duration<double> time_span = std::chrono::steady_clock::now() - start_time;
	m_inferenceTime = time_span.count();
	//std::cout << "inferences time: "  << time_span.count() << " seconds." << std::endl;	
	
	// now set my inference results
	SetInferenceResult();
		
	return( true );
}

