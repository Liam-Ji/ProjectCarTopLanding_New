#include <opencv2\legacy\legacy.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include "vfc.h"
#define PLOT                2      // 控制是否进行画图以及绘图的方式
#define ZoomScaling_s     0.2      // 控制场景图像缩放比率
#define minHessian        1200      // 参数设置为 400 / 1200 效果比较好
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
	resize( img_scene_orig, img_scene, cvSize( 0, 0 ), 0.2, 0.2 );  //把图像调整到合适的大小
	vector<Mat> channels;
	split( img_scene, channels );     // 分离色彩通道, 把一个3通道图像转换成3个单通道图像  
	Mat img_scene_BlueChannel = channels.at( 0 );   // 蓝通道
	Mat img_scene_GreenChannel = channels.at( 1 );  // 绿通道
	Mat img_scene_RedChannel = channels.at( 2 );    // 红通道

	SurfFeatureDetector detector( minHessian );
	detector.detect( img_scene_RedChannel, keypoints_scene );
	
	int SceneZoom = 0;   // 如果需要对场景图进行缩放其缩放次数
	bool flagZoom = false;
	if ( keypoints_scene.size( ) == 0 )
	{
		double scale_s = 0.2;

		while ( keypoints_scene.size( ) == 0 && SceneZoom < 3 )
		{
			scale_s += 0.02;
			Size dsize_s = Size( img_scene_orig.cols*scale_s, img_scene_orig.rows*scale_s );
			img_scene = Mat( dsize_s, CV_32S );
			resize( img_scene_orig, img_scene, dsize_s );   // 对缩小后的场景图像进行放大
			detector.detect( img_scene_RedChannel, keypoints_scene );  // 对放大后的场景图像进行特征点检测
			SceneZoom = SceneZoom + 1;
		}

		if ( keypoints_scene.size( ) > 0 )//  对缩放后场景图的特征点坐标位置进行还原
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
			flagZoom = true;  // 当前已进行过场景图像的再次放大
		}

	}

	SurfDescriptorExtractor extractor;
	extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;        //  原 FlannBasedMatcher 对应 match 函数
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
	//////////////////////////////////////////////////   利用 vector field consensus (VFC) 进行错误匹配点剔除
	double * Matches_x = new double[good_matches.size( )];
	double * Matches_y = new double[good_matches.size( )];

	//////////////////////////////////////////////////////////////   提取匹配点
	int temp = 0;
	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		temp = good_matches[i].trainIdx;
		Matches_x[i] = keypoints_scene[temp].pt.x;   // 获取最优匹配点对的坐标 x
		Matches_y[i] = keypoints_scene[temp].pt.y;   // 获取最优匹配点对的坐标 y
	}


	//////////////////////////////////////////////////   对模板图与场景图中检测到特征点进行分别显示
	Mat img_scene_fea_match;     // 定义带匹配成功特征点的场景图
	img_scene.copyTo( img_scene_fea_match );

	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		circle( img_scene_fea_match, cvPoint( Matches_x[i], Matches_y[i] ), 0, CV_RGB( 255, 255, 120 ), 3, 8, 0 );  // 在场景图中画出特征点
	}

	//namedWindow("img_scene_fea_match", WINDOW_NORMAL);
	imshow( "img_scene_fea_match", img_scene_fea_match );


	////////////////////////////////////////////////////////////    计算匹配点集合两两距离
	double * MatchesDistance = new double[good_matches.size( )];  // 当前点与其它多有点的距离和

	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		MatchesDistance[i] = 0;
		for ( int j = 0; j < good_matches.size( ); j++ )
		{
			MatchesDistance[i] = MatchesDistance[i] + sqrt( pow( Matches_x[i] - Matches_x[j], 2 ) +
															pow( Matches_y[i] - Matches_y[j], 2 ) );
		}

	}

	//////////////////////////////////////////////////    获取匹配点对两两之间最短距离
	double MinMatchesDistance = MatchesDistance[0];
	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		if ( MinMatchesDistance > MatchesDistance[i] )
		{
			MinMatchesDistance = MatchesDistance[i];
		}
	}


	double Object_x = 0;   // 定义场景图中目标坐标 X
	double Object_y = 0;   // 定义场景图中目标坐标 Y

	int PerfectMatchesNum = 0;
	for ( int i = 0; i < good_matches.size( ); i++ )
	{
		if ( MatchesDistance[i] <= 1.2 * MinMatchesDistance )   // 如果当前特征点非孤立点
		{
			temp = good_matches[i].trainIdx;
			Object_x = Object_x + keypoints_scene[temp].pt.x;
			Object_y = Object_y + keypoints_scene[temp].pt.y;

			PerfectMatchesNum = PerfectMatchesNum + 1;
		}

	}

	Object_x = floor( Object_x / PerfectMatchesNum );   // 获取目标中心点坐标 x
	Object_y = floor( Object_y / PerfectMatchesNum );   // 获取目标中心点坐标 y

	delete Matches_x;
	delete Matches_y;
	delete MatchesDistance;
}
