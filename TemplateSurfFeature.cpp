#include <opencv2\core\core.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
# define ZoomScaling_t     0.2      // 控制模板图像缩放比率  
# define minHessian        1200      // 参数设置为 400 / 1200 效果比较好
using namespace cv;
using namespace std;

void ObjectTemplateProcess( vector<KeyPoint> &keypoints_object, Mat &descriptors_object )
{
	Mat img_object_orig = imread( "Test_7.jpg" );        //  读取模板图图像 Test_3 (彩)  Test_2 (黑)
	Mat img_object;
	if ( img_object_orig.cols > 400 || img_object_orig.rows > 400 )
	{
		double scale_t = ZoomScaling_t;
		Size dsize_t = Size( img_object_orig.cols*scale_t, img_object_orig.rows*scale_t );
		img_object = Mat( dsize_t, CV_32S );
		resize( img_object_orig, img_object, dsize_t );   // 对模板图像进行尺度缩放
	}
	else
	{
		img_object = img_object_orig.clone( );          //  模板图不进行缩放的话则进行复制
	}
	vector<Mat> channels;
	split( img_object, channels );     // 分离色彩通道, 把一个3通道图像转换成3个单通道图像  
	Mat img_object_BlueChannel = channels.at( 0 );   // 蓝通道
	Mat img_object_GreenChannel = channels.at( 1 );  // 绿通道
	Mat img_object_RedChannel = channels.at( 2 );    // 红通道

	// step1:检测模板图SURF特征点  <红色通道>  //////////////////////////////////////////
	SurfFeatureDetector detector( minHessian );
	/*std::vector<KeyPoint> keypoints_object, keypoints_scene;*/
	detector.detect( img_object_RedChannel, keypoints_object );   // 检测模板图中的特征点

	//step2:计算模板图特征向量////////////////////////////////////////////
	SurfDescriptorExtractor extractor;
	//Mat descriptors_object, descriptors_scene;
	extractor.compute( img_object, keypoints_object, descriptors_object );
}