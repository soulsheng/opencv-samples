// WARNING: this sample is under construction! Use it on your own risk.
#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif


#include <iostream>
#include <iomanip>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include "CVideoWriter.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;


static void help()
{
    cout << "Usage: ./cascadeclassifier \n\t--cascade <cascade_file>\n\t(<image>|--video <video>|--camera <camera_id> --show <0|1> --write <0|1> )\n"
            "Using OpenCV version " << CV_VERSION << endl << endl;
}


static void convertAndResize(const Mat& src, Mat& gray, Mat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        cv::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}

static void onlyResize(const Mat& src, Mat& resized, double scale)
{
	Size sz(cvRound(src.cols * scale), cvRound(src.rows * scale));

	if (scale != 1)
	{
		cv::resize(src, resized, sz);
	}
	else
	{
		resized = src;
	}
}

static void convertAndResize(const GpuMat& src, GpuMat& gray, GpuMat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cuda::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        cv::cuda::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}


static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, Scalar(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}


static void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorRed = Scalar(255,0,0);
    Scalar fontColorNV  = Scalar(118,185,0);

    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
#if 0
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        (bGpu ? "GPU, " : "CPU, ") <<
        (bLargestFace ? "OneFace, " : "MultiFace, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");
    matPrint(canvas, 1, fontColorRed, ss.str());

    // by Anatoly. MacOS fix. ostringstream(const string&) is a private
    // matPrint(canvas, 2, fontColorNV, ostringstream("Space - switch GPU / CPU"));
    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "Space - switch GPU / CPU");
        matPrint(canvas, 3, fontColorNV, "M - switch OneFace / MultiFace");
        matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
        matPrint(canvas, 6, fontColorNV, "1/Q - increase/decrease scale");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
#endif
}

std::string MakeFourCCString(uint32_t x) {
	std::string str;
	str += x >> 24;
	str += (x >> 16) & 0xff;
	str += (x >> 8) & 0xff;
	str += x & 0xff;
	return str;
}

void rectangleCorner(Mat& img, Rect rec,
	const Scalar& color, int thickness = 1)
{
	float ratio = 0.2f;
	float lineWidth = ratio * rec.width;
	float lineHeight = ratio * rec.height;
	cv::Point pLeftTop = rec.tl();
	cv::Point pRightBottom = rec.br();

	cv::Point pStart, pOffset;

	// left top  |~
	pStart = pLeftTop;

	pOffset = cvPoint(0, lineHeight);
	cv::line(img, pStart, pStart + pOffset, color, thickness);

	pOffset = cvPoint(lineWidth, 0);
	cv::line(img, pStart, pStart + pOffset, color, thickness);

	// left bottom |_
	pStart = pLeftTop + cv::Point(0, rec.height);

	pOffset = cvPoint(0, -lineHeight);
	cv::line(img, pStart, pStart + pOffset, color, thickness);

	pOffset = cvPoint(lineWidth, 0);
	cv::line(img, pStart, pStart + pOffset, color, thickness);

	// right top  ~|
	pStart = pLeftTop + cv::Point(rec.width, 0);

	pOffset = cvPoint(0, lineHeight);
	cv::line(img, pStart, pStart + pOffset, color, thickness);

	pOffset = cvPoint(-lineWidth, 0);
	cv::line(img, pStart, pStart + pOffset, color, thickness);

	// right top  _|
	pStart = pRightBottom;

	pOffset = cvPoint(0, -lineHeight);
	cv::line(img, pStart, pStart + pOffset, color, thickness);

	pOffset = cvPoint(-lineWidth, 0);
	cv::line(img, pStart, pStart + pOffset, color, thickness);

}

int main(int argc, const char *argv[])
{
    if (argc == 1)
    {
        help();
        return -1;
    }

    if (getCudaEnabledDeviceCount() == 0)
    {
        return cerr << "No GPU found or the library is compiled without CUDA support" << endl, -1;
    }

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    string cascadeName;
    string inputName;
    bool isInputImage = false;
    bool isInputVideo = false;
    bool isInputCamera = false;
	bool isShow = true;
	bool isWrite = true;
	float rate = 0.0;

	std::string tmpStr;
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--cascade")
            cascadeName = argv[++i];
        else if (string(argv[i]) == "--video")
        {
            inputName = argv[++i];
            isInputVideo = true;
        }
        else if (string(argv[i]) == "--camera")
        {
            inputName = argv[++i];
            isInputCamera = true;
        }
        else if (string(argv[i]) == "--help")
        {
            help();
            return -1;
        }
		else if (string(argv[i]) == "--show")
		{
			tmpStr = argv[++i];
			isShow = atoi(tmpStr.c_str());
		}
		else if (string(argv[i]) == "--write")
		{
			tmpStr = argv[++i];
			isWrite = atoi(tmpStr.c_str());
		}
		else if (string(argv[i]) == "--rate")
		{
			tmpStr = argv[++i];
			rate = atof(tmpStr.c_str());
		}
        else if (!isInputImage)
        {
            inputName = argv[i];
            isInputImage = true;
        }
        else
        {
            cout << "Unknown key: " << argv[i] << endl;
            return -1;
        }
    }

	cout << "isWrite = " << isWrite << ", isShow = " << isShow << ", rate = " << rate << endl;

    Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create(cascadeName);

    cv::CascadeClassifier cascade_cpu;
    if (!cascade_cpu.load(cascadeName))
    {
        return cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"" << endl, help(), -1;
    }

    VideoCapture capture;
    Mat image;

    if (isInputImage)
    {
        image = imread(inputName);
        CV_Assert(!image.empty());
    }
    else if (isInputVideo)
    {
        capture.open(inputName);
        CV_Assert(capture.isOpened());
    }
    else
    {
        capture.open(atoi(inputName.c_str()));
        CV_Assert(capture.isOpened());
    }

	if (isShow)
		namedWindow("result", 1);

    Mat frame, frame_cpu, gray_cpu, resized_cpu, frameDisp;
    vector<Rect> faces;

    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;

    /* parameters */
    bool useGPU = true;
    double scaleFactor = 1.0;
    bool findLargestObject = false;
    bool filterRects = true;
    bool helpScreen = false;

	// ready to write result video
	int frame_num = capture.get(cv::CAP_PROP_FRAME_COUNT);
	CVideoWriter videoWriter;
	bool bRet = videoWriter.writeVideoBegin(inputName + "_detect.mp4",
		capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT), rate,
		capture.get(CAP_PROP_FPS), capture.get(CAP_PROP_FOURCC));

	cout << "FOURCC = " << MakeFourCCString((unsigned int)capture.get(CAP_PROP_FOURCC)) << endl;

	for (int iFrame = 0; iFrame<frame_num; iFrame++)
    {
        if (isInputCamera || isInputVideo)
        {
            capture >> frame;
            if (frame.empty())
            {
                break;
            }
        }

        (image.empty() ? frame : image).copyTo(frame_cpu);
        frame_gpu.upload(image.empty() ? frame : image);

        convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
        convertAndResize(frame_cpu, gray_cpu, resized_cpu, scaleFactor);

        TickMeter tm;
        tm.start();

        if (useGPU)
        {
            cascade_gpu->setFindLargestObject(findLargestObject);
            cascade_gpu->setScaleFactor(1.2);
            cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);

            cascade_gpu->detectMultiScale(resized_gpu, facesBuf_gpu);
            cascade_gpu->convert(facesBuf_gpu, faces);
        }
        else
        {
            Size minSize = cascade_gpu->getClassifierSize();
            cascade_cpu.detectMultiScale(resized_cpu, faces, 1.2,
                                         (filterRects || findLargestObject) ? 4 : 0,
                                         (findLargestObject ? CASCADE_FIND_BIGGEST_OBJECT : 0)
                                            | CASCADE_SCALE_IMAGE,
                                         minSize);
        }

		onlyResize(frame_cpu, frameDisp, scaleFactor);

        for (size_t i = 0; i < faces.size(); ++i)
        {
			rectangleCorner(frameDisp, faces[i], Scalar(0, 255, 0), 2);

			int fontFace = FONT_HERSHEY_DUPLEX;
			double fontScale = 0.8;
			int fontThickness = 2;
			double t = 36.8 + i*0.2;
			Scalar fontColorRed = Scalar(255, 255, 0);

			if ( t > 37.3 )
				fontColorRed = Scalar(0, 255, 255);

			ostringstream ss;
			ss << setprecision(1) << fixed << t;// << "¡ãC";

			cv::Point posTemp( faces[i].tl() );
			cv::Point offset(-10, -10);//cv::Point offset = cv::Point(faces[i].height * 0.4, -faces[i].height * 0.1);

			putText(frameDisp, ss.str().c_str(), posTemp + offset,
				fontFace, fontScale, fontColorRed, 1, 16);
        }

        tm.stop();
        double detectionTime = tm.getTimeMilli();
        double fps = 1000 / detectionTime;

        //print detections to console
        cout << setfill(' ') << setprecision(2);
        cout << setw(6) << fixed << fps << " FPS, " << faces.size() << " det";
        if ((filterRects || findLargestObject) && !faces.empty())
        {
            for (size_t i = 0; i < faces.size(); ++i)
            {
                cout << ", [" << setw(4) << faces[i].x
                     << ", " << setw(4) << faces[i].y
                     << ", " << setw(4) << faces[i].width
                     << ", " << setw(4) << faces[i].height << "]";
            }
        }
        cout << endl;
		cout << "frame " << iFrame << " / " << frame_num << ", ";
        //cv::cvtColor(resized_cpu, frameDisp, COLOR_GRAY2BGR);
        displayState(frameDisp, helpScreen, useGPU, findLargestObject, filterRects, fps);
		if (isShow)
			imshow("result", frameDisp);
		if (isWrite)
			videoWriter.writeVideoKernel(frameDisp);

        char key = (char)waitKey(5);
        if (key == 27)
        {
            break;
        }

        switch (key)
        {
        case ' ':
            useGPU = !useGPU;
            break;
        case 'm':
        case 'M':
            findLargestObject = !findLargestObject;
            break;
        case 'f':
        case 'F':
            filterRects = !filterRects;
            break;
        case '1':
            scaleFactor *= 1.05;
            break;
        case 'q':
        case 'Q':
            scaleFactor /= 1.05;
            break;
        case 'h':
        case 'H':
            helpScreen = !helpScreen;
            break;
        }
    }

	videoWriter.writeVideoEnd();

    return 0;
}
