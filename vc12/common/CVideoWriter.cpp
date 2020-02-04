
#include "CVideoWriter.h"
#include <iostream>

CVideoWriter::CVideoWriter()
{

}

CVideoWriter::~CVideoWriter()
{

}   
    
bool CVideoWriter::writeVideoBegin(const std::string& filename, int width, int height, float rate, 
	float fps, int fmt)
{
	m_VideoWriter.open(filename, fmt, fps, cv::Size(width, height));
	if (fabs(rate) > 1e-3)
		m_VideoWriter.set(cv::VIDEOWRITER_PROP_QUALITY, rate);

	// check if we succeeded
	if (!m_VideoWriter.isOpened()) {
		std::cerr << "Could not open the output video file for write\n";
		return false;
	}

	return true;
}

void CVideoWriter::writeVideoKernel(cv::Mat &img)
{
	if (!m_VideoWriter.isOpened())
		return;

	m_VideoWriter.write(img);
}

void CVideoWriter::writeVideoEnd()
{
	if (!m_VideoWriter.isOpened())
		return;

	m_VideoWriter.release();
}
