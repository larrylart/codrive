////////////////////////////////////////////////////////////////////
// Objects detection using with Khadas's AML NPU with Mobilenet SSD v2
// Created by: Larry Lart
////////////////////////////////////////////////////////////////////
#include <time.h>
#include <iostream>
#include <string>

#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

// main header include
#include "aml_worker.h"

String modelBinary = "data/mobilenet_ssd_v2a.nb"; 
String modelLabels = "data/coco_labels.txt";

const char* keys =
      "{ help           | false | print usage         }"
	  "{ annotate       | false | annotate detected image and save }"
	  "{ gui            | false | display gui anotated video }"
      "{ label          |  | model labels }"
      "{ model          |  | model weights }"
      "{ camera_device  | 0     | camera device number }"
      "{ camera_width   | 640   | camera device width  }"
      "{ camera_height  | 480   | camera device height }"
      "{ video          |       | video or image for detection}"
      "{ min_confidence | 0.2   | min confidence      }";
	  
////////////////////////////////////////////////////////////////////
bool annotateImage( cv::Mat& inputMat, std::vector<networkResult> vectMatch, bool doWrite )
{
	if( vectMatch.size() <= 0 ) return(false);
	char strLabel[128];
	
	for( int i=0; i<vectMatch.size(); i++ )
	{
		// skip in not min 0.2
		if( vectMatch[i].confidence < 0.2 ) continue;
		
		//std::string strObjName = pAMLWorker->GetClassName(vectMatch[i].objectClass);
		
		int x1 = vectMatch[i].left;
		int y1 = vectMatch[i].top;
		int x2 = vectMatch[i].right;
		int y2 = vectMatch[i].bottom;		
				
		cv::rectangle(inputMat, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0, 0, 255), 2);
		sprintf(strLabel,"%d:%s", i, vectMatch[i].objectName );
		cv::putText(inputMat, strLabel, cv::Point(x1, y2 - 5), 0, 0.8, cv::Scalar(0, 0, 255));

		if( doWrite ) 
			cv::imwrite("obj_detect_note.jpg", inputMat);
		else
		{
			imshow("Object Detections", inputMat);		
			if (waitKey(1) >= 0) return(false);
		}
	}
	
	return(true);
}

// helper functions copatibility cv 2/3
////////////////////////////////////////////////////////////////////
void parser_printMessage(CommandLineParser& parser)
{
#ifndef _CV_OLD_
	parser.printMessage();
#else	
	parser.printParams();
#endif
}
bool parser_has(CommandLineParser& parser, const std::string& keys)
{
#ifndef _CV_OLD_
	return( parser.has(keys) );
#else	
	return( parser.get<bool>(keys) );
#endif
}

////////////////////////////	  
// main 
int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
#ifndef _CV_OLD_
    parser.about("This sample uses Coral TPU Single-Shot Detector "
                 "to detect objects from webcam/video\n" );
#endif

    if (parser.get<bool>("help"))
    {
		parser_printMessage( parser );
        return 0;
    }
	// set to annotate
	bool do_annotate = false;
	if (parser.get<bool>("annotate")) do_annotate = true;
	// set to annotate
	bool do_gui = false;
	if (parser.get<bool>("gui")) do_gui = true;

	// set/force log level
	setenv("VSI_NN_LOG_LEVEL", "1", 1);	
	
    VideoCapture cap;
    if (!parser_has(parser,"video"))
    {
        int cameraDevice = parser.get<int>("camera_device");
        cap = VideoCapture(cameraDevice,CAP_V4L);
		cout << "open camera: " << cameraDevice << endl;
        if(!cap.isOpened())
        {
            cout << "Couldn't find camera: " << cameraDevice << endl;
            return -1;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, parser.get<int>("camera_width"));
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, parser.get<int>("camera_height"));
		
    } else
    {
        cap.open(parser.get<String>("video"));
		cout << "open video: " << parser.get<String>("video") << endl;
        if(!cap.isOpened())
        {
            cout << "Couldn't open image or video: " << parser.get<String>("video") << endl;
            return -1;
        }
    }

	// test write video
	//VideoWriter outputVideo;  
	//Size inVideoSize(640,480);
	//outputVideo.open("appsrc ! autovideoconvert ! h264_omx ! avimux ! filesink location=test_n1.avi ", cv::CAP_V4L, VideoWriter::fourcc('H','2','6','4'), (double)25, inVideoSize, true);

	//////////////////////////////				 
	// create TPU object
	CAMLWorker* pAMLWorker = NULL;
	pAMLWorker = new CAMLWorker( modelBinary.c_str(), modelLabels.c_str() );
					 
    // Start and end times
    time_t start, end;

	////////////////////////////
	// loop for webcam frames
	unsigned long num_frames = 1;
    time(&start);					 
	double last_seconds = start;
	double total_inf_time = 0;	
	std::vector<networkResult> vectMatchResult;
					 
    for(;;)
    {
        Mat frame;
		double t = (double)getTickCount();	
        cap >> frame; // get a new frame from camera/video or read image

        if( frame.empty() )
        {
            waitKey();
            break;
        }

		// debug write video
		//if( num_frames < 1800 )			
		//	outputVideo.write(frame);
		//else
		//	if( outputVideo.isOpened() ) outputVideo.release();

        if( frame.channels() == 4 ) cvtColor(frame, frame, COLOR_BGRA2BGR);
		if( pAMLWorker == NULL ) continue;
		
		//////////////////
		// push image & run inference
		
		bool isInf = false;
		pAMLWorker->PushImage( &frame );
		isInf = pAMLWorker->RunInferenceOnImage();
		if( !isInf ) continue;
		
		pAMLWorker->GetInference( vectMatchResult );
		
		// End Time
		time(&end);
	
		// Time elapsed
		double seconds = difftime (end, start);
		 
		// Calculate frames per second
		double fps  = num_frames / seconds;
		
		// add up inference time
		total_inf_time += pAMLWorker->m_inferenceTime;

		// DEBUG :: display text version check - only every 1 seconds or so interval
		if( end > last_seconds ) 
		{
			if( vectMatchResult.size() > 0 )
			{
				// :: JUST DEBUG DISPLAY CASE - only first object
				int obj_id = vectMatchResult[0].objectClass;
				std::string strObjName = pAMLWorker->GetClassName(obj_id);
				int x1 = vectMatchResult[0].left;
				int y1 = vectMatchResult[0].top;
				int x2 = vectMatchResult[0].right;
				int y2 = vectMatchResult[0].bottom;	
				float cnfd = vectMatchResult[0].confidence;		

				int inf_time = (int) round(pAMLWorker->m_inferenceTime*1000);
				//float inf_fps = 1000.0/(pAMLWorker->m_inferenceTime*1000.0);
				double avg_inf_time = (double) total_inf_time/num_frames;
				float inf_fps = 1000.0/(avg_inf_time*1000.0);
				
				printf("FPS/I: %.2f / %.2f (%dms) : time : %.2f s :: OBJECT(%.2f) :%d: %s :: (%d,%d,%d,%d)\n", fps, inf_fps, inf_time, seconds, cnfd, obj_id, strObjName.c_str(), x1, y1, x2, y2 );
				last_seconds = end;
			
				// write detection on image and save
				if( do_annotate ) 
					annotateImage( frame, vectMatchResult, true );					
			}			
		}
		// gui display fast
		if( do_gui ) annotateImage( frame, vectMatchResult, false );		

		num_frames++;
					
    }	
	
	// delete tpu worker
	delete( pAMLWorker );
	// close
	//outputVideo.release();	
	
    return( 0 );
} // main
	
	
