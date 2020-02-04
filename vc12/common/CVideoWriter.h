
#pragma once

#include <string>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class CVideoWriter
{
public:
	bool writeVideoBegin(const std::string& filename, int width, int height, float rate=0, 
		float fps = 25.0f,
		int fmt = CV_FOURCC('M', 'P', '4', 'V') );
    void writeVideoKernel(cv::Mat &img);
    void writeVideoEnd();
    
    CVideoWriter();
    ~CVideoWriter();
protected:
	cv::VideoWriter m_VideoWriter;

};