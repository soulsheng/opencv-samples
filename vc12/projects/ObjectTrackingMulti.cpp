#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;
 
// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()
 
Ptr<Tracker> createTracker(int nTrackerType, string& trackerType)
{
	// OpenCV 3.0.0 :		BOOSTING, MIL, TLD, MEDIANFLOW
	// OpenCV 3.1.0 :		+	KCF
	// OpenCV 3.2.0 :		+	GOTURN
	// OpenCV 3.4.0 :		+	MOSSE
	// OpenCV 3.4.5 :		+	CSRT

	// List of tracker types in OpenCV 3.2
	// NOTE : GOTURN implementation is buggy and does not work.
	string trackerTypes[8] = { "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN",
		"MOSSE", "CSRT" };
	// vector <string> trackerTypes(types, std::end(types));

	// Create a tracker
	trackerType = trackerTypes[nTrackerType];

	Ptr<Tracker> tracker;

#if (CV_MINOR_VERSION < 3)
	{
		tracker = Tracker::create(trackerType);
	}
#else
	{
		if (trackerType == "BOOSTING")
			tracker = TrackerBoosting::create();
		if (trackerType == "MIL")
			tracker = TrackerMIL::create();
		if (trackerType == "KCF")
			tracker = TrackerKCF::create();
		if (trackerType == "TLD")
			tracker = TrackerTLD::create();
		if (trackerType == "MEDIANFLOW")
			tracker = TrackerMedianFlow::create();
		if (trackerType == "GOTURN")
			tracker = TrackerGOTURN::create();
	}
#endif

	return tracker;
}

int main(int argc, char **argv)
{

	int nTrackerType = 2;
	if (argc >= 3)
		nTrackerType = atoi(argv[2]);

	string videofile;
	if (argc >= 2)
		videofile = argv[1];

	bool bVideoOrImages = true;	//a video file or a path of images
	if( string::npos == videofile.find_last_of('.') )
		bVideoOrImages = false;

	VideoCapture video;
	if (bVideoOrImages)
	{
		// Read video
		video.open(videofile);

        // Exit if video is not opened
        if (!video.isOpened())
        {
            cout << "Could not read video file" << endl;
            return 1;
        
        }
	}

	int baseIndex = 1;
	if (argc >= 4)
		baseIndex = atoi(argv[3]);

	std::ostringstream os;

    // Read first frame
    Mat frame;
	if (bVideoOrImages)
		bool ok = video.read(frame);
	else
	{
		os << videofile << std::setw(4) << std::setfill('0') << baseIndex << ".jpg";
		frame = imread(os.str());
	}

	if (frame.empty())
	{
		cout << "Could not read frame" << endl;
		return 1;
	}
	
	// Define initial boundibg box
	vector<Rect> rois;

	selectROIs("rois", frame, rois, false);
	if (rois.size() < 1)
		return 0;


	MultiTracker trackers;
	vector<Rect2d> obj;
	vector<Ptr<Tracker>> algorithms;

	string trackerType;
	for (auto i = 0; i < rois.size(); i++) {
		obj.push_back(rois[i]);
		algorithms.push_back(createTracker(nTrackerType, trackerType));
	}
	trackers.add(algorithms, frame, obj);
	

	cv::String title("目标跟踪系统v0.9-山仪所     ");
	title += "跟踪模式：";
	title += trackerType;


	string videofileOut("D:\\out.avi");
	if (argc >= 5)
		videofileOut = argv[4];

	string videoFormat("XVID");
	if (argc >= 6)
		videoFormat = argv[5];


	// ready to write result to video file
	VideoWriter out;
	out.open(videofileOut, CV_FOURCC(videoFormat.at(0), videoFormat.at(1), videoFormat.at(2), videoFormat.at(3)), 15,
		Size(frame.cols, frame.rows));

	if (!out.isOpened())
	{
		cout << "Could not create video file" << endl;
	}

    while(1)
    {    
		os.str("");
		os << videofile << std::setw(4) << std::setfill('0') << ++baseIndex << ".jpg";

		if (bVideoOrImages)
			video.read(frame);
		else
			frame = imread(os.str());

		if (frame.empty())
		{
			cout << "finish tracking" << endl;
			break;;
		}

        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result

		bool ok = trackers.update(frame);
		if (ok) {
			for (auto j = 0; j < trackers.getObjects().size(); j++) {
				rectangle(frame, trackers.getObjects()[j], Scalar(255, 127 * j, 255 - 64 * j), 2, 1);
			}
		}
		else
		{
			// Tracking failure detected.
			putText(frame, "Tracking failure detected", Point(50,40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255),1);
		}
		

        // Display tracker type on frame
        //putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

		// Calculate Frames per second (FPS)
		float fps = getTickFrequency() / ((double)getTickCount() - timer);

        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(50,25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(50,170,50), 1);

        // Display frame.
		imshow(title, frame);
		
		out << frame;

        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
 
    }
}