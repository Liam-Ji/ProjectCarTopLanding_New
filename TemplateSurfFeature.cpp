#include <opencv2\core\core.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
# define ZoomScaling_t     0.2      // ����ģ��ͼ�����ű���  
# define minHessian        1200      // ��������Ϊ 400 / 1200 Ч���ȽϺ�
using namespace cv;
using namespace std;

void ObjectTemplateProcess( vector<KeyPoint> &keypoints_object, Mat &descriptors_object )
{
	Mat img_object_orig = imread( "Test_7.jpg" );        //  ��ȡģ��ͼͼ�� Test_3 (��)  Test_2 (��)
	Mat img_object;
	if ( img_object_orig.cols > 400 || img_object_orig.rows > 400 )
	{
		double scale_t = ZoomScaling_t;
		Size dsize_t = Size( img_object_orig.cols*scale_t, img_object_orig.rows*scale_t );
		img_object = Mat( dsize_t, CV_32S );
		resize( img_object_orig, img_object, dsize_t );   // ��ģ��ͼ����г߶�����
	}
	else
	{
		img_object = img_object_orig.clone( );          //  ģ��ͼ���������ŵĻ�����и���
	}
	vector<Mat> channels;
	split( img_object, channels );     // ����ɫ��ͨ��, ��һ��3ͨ��ͼ��ת����3����ͨ��ͼ��  
	Mat img_object_BlueChannel = channels.at( 0 );   // ��ͨ��
	Mat img_object_GreenChannel = channels.at( 1 );  // ��ͨ��
	Mat img_object_RedChannel = channels.at( 2 );    // ��ͨ��

	// step1:���ģ��ͼSURF������  <��ɫͨ��>  //////////////////////////////////////////
	SurfFeatureDetector detector( minHessian );
	/*std::vector<KeyPoint> keypoints_object, keypoints_scene;*/
	detector.detect( img_object_RedChannel, keypoints_object );   // ���ģ��ͼ�е�������

	//step2:����ģ��ͼ��������////////////////////////////////////////////
	SurfDescriptorExtractor extractor;
	//Mat descriptors_object, descriptors_scene;
	extractor.compute( img_object, keypoints_object, descriptors_object );
}