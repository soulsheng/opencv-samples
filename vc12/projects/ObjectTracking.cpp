#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
 
using namespace cv;
using namespace std;
 
// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()
 
int main(int argc, char **argv)
{
	// OpenCV 3.0.0 :		BOOSTING, MIL, TLD, MEDIANFLOW
	// OpenCV 3.1.0 :		+	KCF
	// OpenCV 3.2.0 :		+	GOTURN
	// OpenCV 3.4.0 :		+	MOSSE
	// OpenCV 3.4.5 :		+	CSRT

	// List of tracker types in OpenCV 3.2
    // NOTE : GOTURN implementation is buggy and does not work.
    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN",
	"MOSSE", "CSRT"};
    // vector <string> trackerTypes(types, std::end(types));
 
	int nTrackerType = 2;
	if (argc >= 3)
		nTrackerType = atoi(argv[2]);

    // Create a tracker
	string trackerType = trackerTypes[nTrackerType];
 
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

    // Read first frame
    Mat frame;
	if (bVideoOrImages)
		bool ok = video.read(frame);
	else
	{
		string videofile0 = videofile + "0001.jpg";
		frame = imread(videofile0);
	}

    // Define initial boundibg box
    Rect2d bbox(287, 23, 86, 320);
     
    // Uncomment the line below to select a different bounding box
    bbox = selectROI(frame, false);
 
	cv::String title("目标跟踪系统v0.9-山仪所     ");
	title += "跟踪模式：";
	title += trackerType;

    // Display bounding box.
    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
	imshow(title, frame);
     
    tracker->init(frame, bbox);
     
	int indexFile = 1;
	std::ostringstream os;
    while(!frame.empty())
    {    
		os.str("");
		os << videofile << std::setw(4) << std::setfill('0') << ++indexFile << ".jpg";

		if (bVideoOrImages)
			video.read(frame);
		else
			frame = imread(os.str());

        // Start timer
        double timer = (double)getTickCount();
         
        // Update the tracking result
        bool ok = tracker->update(frame, bbox);
         
        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok)
        {
            // Tracking success : Draw the tracked object
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        }
        else
        {
            // Tracking failure detected.
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        // Display tracker type on frame
        //putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

        // Display FPS on frame
        //putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Display frame.
		imshow(title, frame);
         
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
 
    }
}