#include <opencv2\legacy\legacy.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include "vfc.h"
#define PLOT                2      // �����Ƿ���л�ͼ�Լ���ͼ�ķ�ʽ
#define ZoomScaling_s     0.2      // ���Ƴ���ͼ�����ű���
#define minHessian        1200      // ��������Ϊ 400 / 1200 Ч���ȽϺ�
using namespace cv;
using namespace std;

void BackgroundSurfFeature(
	std::vector<KeyPoint> &keypoints_object,
	Mat &descriptors_object,
	Mat &img_scene_orig,
	std::vector<KeyPoint> &keypoints_scene,
	Mat &descriptors_scene
	)
{
	Mat img_scene;
	resize( img_scene_orig, img_scene, cvSize( 0, 0 ), 0.2, 0.2 );  //��ͼ����������ʵĴ�С
	vector<Mat> channels;
	split( img_scene, channels );     // ����ɫ��ͨ��, ��һ��3ͨ��ͼ��ת����3����ͨ��ͼ��  
	Mat img_scene_BlueChannel = channels.at( 0 );   // ��ͨ��
	Mat img_scene_GreenChannel = channels.at( 1 );  // ��ͨ��
	Mat img_scene_RedChannel = channels.at( 2 );    // ��ͨ��

	SurfFeatureDetector detector( minHessian );
	detector.detect( img_scene_RedChannel, keypoints_scene );
	
	int SceneZoom = 0;   // �����Ҫ�Գ���ͼ�������������Ŵ���
	bool flagZoom = false;
	if ( keypoints_scene.size( ) == 0 )
	{
		double scale_s = 0.2;

		while ( keypoints_scene.size( ) == 0 && SceneZoom < 3 )
		{
			scale_s += 0.02;
			Size dsize_s = Size( img_scene_orig.cols*scale_s, img_scene_orig.rows*scale_s );
			img_scene = Mat( dsize_s, CV_32S );
			resize( img_scene_orig, img_scene, dsize_s );   // ����С��ĳ���ͼ����зŴ�
			detector.detect( img_scene_RedChannel, keypoints_scene );  // �ԷŴ��ĳ���ͼ�������������
			SceneZoom = SceneZoom + 1;
		}

		if ( keypoints_scene.size( ) > 0 )//  �����ź󳡾�ͼ������������λ�ý��л�ԭ
		{
			for ( int i = 0; i < keypoints_scene.size( ); i++ )
			{
				keypoints_scene[i].pt.x = keypoints_scene[i].pt.x / ( SceneZoom * 0.02 + ZoomScaling_s ) * ZoomScaling_s;
				keypoints_scene[i].pt.y = keypoints_scene[i].pt.y / ( SceneZoom * 0.02 + ZoomScaling_s ) * ZoomScaling_s;
			}
			double scale_s = ZoomScaling_s;
			Size dsize_s = Size( img_scene_orig.cols*scale_s, img_scene_orig.rows*scale_s );
			img_scene = Mat( dsize_s, CV_32S );
			resize( img_scene_orig, img_scene, dsize_s );
			flagZoom = true;  // ��ǰ�ѽ��й�����ͼ����ٴηŴ�
		}

	}

	SurfDescriptorExtractor extractor;
	extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;        //  ԭ FlannBasedMatcher ��Ӧ match ����
	matcher.match( descriptors_object, descriptors_scene, matches );

	double max_dist = 0;
	double min_dist = 100;
	double MeanDistance = 0;

	for ( int i = 0; i < matches.size( ); i++ )
	{
		double dist = matches[i].distance;
		if ( dist < min_dist )  min_dist = dist;
		if ( dist > max_dist )  max_dist = dist;

		MeanDistance = MeanDistance + dist;
	}

	MeanDistance = MeanDistance / descriptors_object.rows;

	vector<Point2f> X;
	vector<Point2f> Y;
	X.clear( );
	Y.clear( );
	for ( unsigned int i = 0; i < matches.size( ); i++ )
	{
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;
		X.push_back( keypoints_object[idx1].pt );
		Y.push_back( keypoints_scene[idx2].pt );
	}
	// main process
	VFC myvfc;
	myvfc.setData( X, Y );
	myvfc.optimize( );
	vector<int> matchIdx = myvfc.obtainCorrectMatch( );

	// postprocess data format
	std::vector< DMatch > good_matches;
	std::vector<KeyPoint> correctKeypoints_1, correctKeypoints_2;
	good_matches.clear( );

	

	for ( unsigned int i = 0; i < matchIdx.size( ); i++ )
	{
		int idx = matchIdx[i];

		if ( idx >= keypoints_object.size( ) || idx >= keypoints_scene.size( ) ) continue;

		good_matches.push_back( matches[idx] );
		correctKeypoints_1.push_back( keypoints_object[idx] );
		correctKeypoints_2.push_back( keypoints_scene[idx] );
	}
	//////////////////////////////////////////////////   ���� vector field consensus (VFC) ���д���ƥ����޳�
	double * Matches_x = new double[good_matches.size( )];
	double * Matches_y = new double[good_matches.size( )];

	//////////////////////////////////////////////////////////////   ��ȡƥ���
	int temp = 0;
	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		temp = good_matches[i].trainIdx;
		Matches_x[i] = keypoints_scene[temp].pt.x;   // ��ȡ����ƥ���Ե����� x
		Matches_y[i] = keypoints_scene[temp].pt.y;   // ��ȡ����ƥ���Ե����� y
	}


	//////////////////////////////////////////////////   ��ģ��ͼ�볡��ͼ�м�⵽��������зֱ���ʾ
	Mat img_scene_fea_match;     // �����ƥ��ɹ�������ĳ���ͼ
	img_scene.copyTo( img_scene_fea_match );

	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		circle( img_scene_fea_match, cvPoint( Matches_x[i], Matches_y[i] ), 0, CV_RGB( 255, 255, 120 ), 3, 8, 0 );  // �ڳ���ͼ�л���������
	}

	//namedWindow("img_scene_fea_match", WINDOW_NORMAL);
	imshow( "img_scene_fea_match", img_scene_fea_match );


	////////////////////////////////////////////////////////////    ����ƥ��㼯����������
	double * MatchesDistance = new double[good_matches.size( )];  // ��ǰ�����������е�ľ����

	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		MatchesDistance[i] = 0;
		for ( int j = 0; j < good_matches.size( ); j++ )
		{
			MatchesDistance[i] = MatchesDistance[i] + sqrt( pow( Matches_x[i] - Matches_x[j], 2 ) +
															pow( Matches_y[i] - Matches_y[j], 2 ) );
		}

	}

	//////////////////////////////////////////////////    ��ȡƥ��������֮����̾���
	double MinMatchesDistance = MatchesDistance[0];
	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		if ( MinMatchesDistance > MatchesDistance[i] )
		{
			MinMatchesDistance = MatchesDistance[i];
		}
	}


	double Object_x = 0;   // ���峡��ͼ��Ŀ������ X
	double Object_y = 0;   // ���峡��ͼ��Ŀ������ Y

	int PerfectMatchesNum = 0;
	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		if ( MatchesDistance[i] <= 1.2 * MinMatchesDistance )   // �����ǰ������ǹ�����
		{
			temp = good_matches[i].trainIdx;
			Object_x = Object_x + keypoints_scene[temp].pt.x;
			Object_y = Object_y + keypoints_scene[temp].pt.y;

			PerfectMatchesNum = PerfectMatchesNum + 1;
		}

	}

	Object_x = floor( Object_x / PerfectMatchesNum );   // ��ȡĿ�����ĵ����� x
	Object_y = floor( Object_y / PerfectMatchesNum );   // ��ȡĿ�����ĵ����� y

	delete Matches_x;
	delete Matches_y;
	delete MatchesDistance;
}
