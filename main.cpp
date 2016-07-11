#pragma warning (disable:4819)
#include <opencv2\nonfree\nonfree.hpp>
#include "objectDetect.h"
#include "vfc.h"
#define minHessian        1200      // ��������Ϊ 400 / 1200 Ч���ȽϺ�

int main()
{
	VideoCapture capture( "1.mp4" );  // ��ȡ�Ѿ�¼�Ƶ���Ƶ"MOV_0021.mp4"

	// ����Ƿ�������:�ɹ���ʱ��isOpened����ture   
	if (!capture.isOpened())  cout << "fail to open!" << endl;

	Mat img_scene;
	capture >> img_scene;

	double FrameCount = capture.get(CV_CAP_PROP_FRAME_COUNT); // ��ȡ������Ƶ�ܵ�֡��Ŀ
	double frame = 0;
	cout << "������Ƶ��" << FrameCount << "֡" << endl;

	Size s(1164, 660);
	VideoWriter writer = VideoWriter("Myvideo.avi", CV_FOURCC('X', 'V', 'I', 'D'), 25, s);  // �������������Ƶ

	if (!writer.isOpened())  // ���Դ�����Ƶ�ļ��Ĵ�
	{
		cerr << "Can not creat video file.\n" << endl;
		return -1;
	}

	bool pause = false;  // �����е���ͣ���
	
	vector<Point2i> coordinates;  // ��Ƶ��ÿһ֡�е�Ŀ������

	ofstream ofile;
	ofile.open(".\\TestResult.txt");


	/*********************************************************************/
	/*********************************************************************/
	/*********************************************************************/
	/**ģ�崦����*******************************************************/
	Mat img_object = imread( "Test1.jpg" );
	if (img_object.empty()){
		cout << "error opening Test1.jpg" << endl;
		return -1;
	}
	vector<Mat> channels;
	split( img_object, channels );     // ����ɫ��ͨ��, ��һ��3ͨ��ͼ��ת����3����ͨ��ͼ��  
	Mat img_object_BlueChannel = channels.at( 0 );   // ��ͨ��
	Mat img_object_GreenChannel = channels.at( 1 );  // ��ͨ��
	Mat img_object_RedChannel = channels.at( 2 );    // ��ͨ��

	// step1:���ģ��ͼSURF������  <��ɫͨ��>  //////////////////////////////////////////
	SurfFeatureDetector detector( minHessian );
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	detector.detect( img_object_RedChannel, keypoints_object );   // ���ģ��ͼ�е�������

	//step2:����ģ��ͼ��������////////////////////////////////////////////
	SurfDescriptorExtractor extractor;
	Mat descriptors_object, descriptors_scene;
	extractor.compute( img_object, keypoints_object, descriptors_object );
	ofile << "��׼ͼ������������" << keypoints_object.size( ) << endl;
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;



	/*********************************************************************/
	/*********************************************************************/
	/*********************************************************************/





	while (frame < FrameCount-1)   // ��ESC����ֱ���˳�
	{
		++frame;
		cout << "Current frame:" << frame << endl;
		if (!capture.read(img_scene)){
			cout << "��ȡ��Ƶʧ��" << endl;
			return -1;
		}
		if (img_scene.rows > 400){
			//resize(img_scene,img_scene,cvSize(0,0),0.2,0.2);  //��ͼ����������ʵĴ�С
			resize(img_scene, img_scene, cvSize(380, 220), 0, 0);  //��ͼ����������ʵĴ�С
		}

		namedWindow("Original Image");
		imshow("Original Image", img_scene);  // ��ʾԭʼ��ɫͼ��
		/*********************************************************************/
		/*********************************************************************/
		/*********************************************************************/
		/**����ͼ��������ȡ���������ƥ��***************************************/
		vector<Mat> channels;
		split( img_scene, channels );     // ����ɫ��ͨ��, ��һ��3ͨ��ͼ��ת����3����ͨ��ͼ��  
		Mat img_scene_BlueChannel = channels.at( 0 );   // ��ͨ��
		Mat img_scene_GreenChannel = channels.at( 1 );  // ��ͨ��
		Mat img_scene_RedChannel = channels.at( 2 );    // ��ͨ��
		detector.detect( img_scene_RedChannel, keypoints_scene );

		int a = keypoints_scene.size( );
		Mat img_object_fea;    // ������������ģ��ͼ
		Mat img_scene_fea;     // �����������ĳ���ͼ
		img_object.copyTo( img_object_fea );
		img_scene.copyTo( img_scene_fea );
		for ( unsigned int i = 0; i < keypoints_object.size( ); i++ )
			circle( img_object_fea, cvPoint( ( int )keypoints_object[i].pt.x, ( int )keypoints_object[i].pt.y ), 15, CV_RGB( 255, 255, 120 ), 2, 8, 0 );  // ��ģ��ͼ�л���������
		for ( unsigned int i = 0; i < keypoints_scene.size( ); i++ )
			circle( img_scene_fea, cvPoint( ( int )keypoints_scene[i].pt.x, ( int )keypoints_scene[i].pt.y ), 0, CV_RGB( 0, 255, 0 ), 3, 8, 0 );  // �ڳ���ͼ�л���������
		//namedWindow("img_scene_fea", WINDOW_NORMAL);
		imshow( "img_scene_fea", img_scene_fea );
		//namedWindow("img_object_fea", WINDOW_NORMAL);
		//imshow( "img_object_fea", img_object_fea );
		imwrite( "img_object_fea.png", img_object_fea );
		extractor.compute( img_scene, keypoints_scene, descriptors_scene );
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match( descriptors_object, descriptors_scene, matches );
		cout << matches.size( ) << endl;

		double max_dist = 0;
		double min_dist = 100;
		double MeanDistance = 0;
		for ( unsigned int i = 0; i < matches.size( ); i++ )	{
			double dist = matches[i].distance;
			if ( dist < min_dist )  min_dist = dist;
			if ( dist > max_dist )  max_dist = dist;
			MeanDistance = MeanDistance + dist;
		}
		MeanDistance = MeanDistance / descriptors_object.rows;
		cout << min_dist << "\t" << max_dist << "\t" << MeanDistance << endl;
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
			unsigned int idx = matchIdx[i];

			if ( idx >= keypoints_object.size( ) || idx >= keypoints_scene.size( ) ) continue;

			good_matches.push_back( matches[idx] );
			correctKeypoints_1.push_back( keypoints_object[idx] );
			correctKeypoints_2.push_back( keypoints_scene[idx] );
		}
		cout << good_matches.size( ) << endl;
		double * Matches_x = new double[good_matches.size( )];
		double * Matches_y = new double[good_matches.size( )];
		Point2i Matche;
		vector<Point2i>  Matches;
		//////////////////////////////////////////////////////////////   ��ȡƥ���
		int temp = 0;
		for ( unsigned int i = 0; i < good_matches.size( ); i++ )
		{
			temp = good_matches[i].trainIdx;
			Matches_x[i] = keypoints_scene[temp].pt.x;   // ��ȡ����ƥ���Ե����� x
			Matches_y[i] = keypoints_scene[temp].pt.y;   // ��ȡ����ƥ���Ե����� y
			Matche.x = (int)keypoints_scene[temp].pt.x;
			Matche.y = (int)keypoints_scene[temp].pt.y;
			Matches.push_back( Matche );
		}
		//////////////////////////////////////////////////   ��ģ��ͼ�볡��ͼ�м�⵽��������зֱ���ʾ
		Mat img_scene_fea_match;     // �����ƥ��ɹ�������ĳ���ͼ
		img_scene.copyTo( img_scene_fea_match );

		for ( unsigned int i = 0; i < good_matches.size( ); i++ )
		{
			circle( img_scene_fea_match, cvPoint( ( int )Matches_x[i], ( int )Matches_y[i] ), 0, CV_RGB( 255, 255, 120 ), 3, 8, 0 );  // �ڳ���ͼ�л���������
		}

		//namedWindow("img_scene_fea_match", WINDOW_NORMAL);
		imshow( "img_scene_fea_match", img_scene_fea_match );


		/*********************************************************************/
		/*********************************************************************/
		/*********************************************************************/
		/*********************************************************************/
		vector<Mat> co_image;
		putText(img_scene, "The Original Image", Point(1, 15), 1, 1, Scalar(255, 255, 255), 1);
		co_image.push_back(img_scene);



		//// ��ԭʼͼ�������ǿ����  <����ͼ����ǿ����ı�Ҫ��>
		//Mat new_image = Mat::zeros(img_scene.size(), img_scene.type());
		//ImageEnhance(img_scene, new_image);   // ͼ�񱥺Ͷ���ǿ

		//putText(new_image, "The Original Image After Enhanced", Point(1, 15), 1, 1, Scalar(255, 255, 255), 1);
		//co_image.push_back(new_image);

		//img_scene = new_image;  // ����ǿ���ͼ����к����Ĵ������




		Mat Seg_img_red;   // ��ɫͨ����ֵ�ָ���ͼ
		Mat Seg_img_blue;  // ��ɫͨ����ֵ�ָ���ͼ
		preprocess(img_scene, Seg_img_red, Seg_img_blue); // ͨ���ָ�����һ����ȡĿ�����ɫ�ͺ�ɫ����

		// ׼������ʾ��Seg_img_blue
		Mat Seg_img_blue_disp;
		Seg_img_blue.convertTo(Seg_img_blue_disp, CV_8U);    // ת��Ϊ��λ�޷�������
		cvtColor(Seg_img_blue_disp*255, Seg_img_blue_disp, CV_GRAY2BGR);
		putText(Seg_img_blue_disp, "The Seg_img_blue Image", Point(3, 15), 1, 1, Scalar(255, 255, 255), 1);
		co_image.push_back(Seg_img_blue_disp);

		// ׼������ʾ��Seg_img_red
		Mat Seg_img_red_disp;
		Seg_img_red.convertTo(Seg_img_red_disp, CV_8U);    // ת��Ϊ��λ�޷�������
		cvtColor(Seg_img_red_disp*255, Seg_img_red_disp, CV_GRAY2BGR);
		Seg_img_red_disp.convertTo(Seg_img_red_disp, CV_8U);    // ת��Ϊ��λ�޷�������
		putText(Seg_img_red_disp, "The Seg_img_red Image", Point(3, 15), 1, 1, Scalar(255, 255, 255), 1);
		co_image.push_back(Seg_img_red_disp);

//		Mat Object_img;
//		Mat Object_img_float;
//		bitwise_xor(Seg_img_red, Seg_img_blue, Object_img_float);   // ������ͼ��Ľ�����ȡĿ���Ǳ����ɫ����
//		Object_img_float.convertTo(Object_img, CV_8U);

		Mat Seg_img_blue_cc;
		Seg_img_blue.convertTo(Seg_img_blue_cc, CV_8U);    // ת��Ϊ��λ�޷�������

		dilate(Seg_img_blue_cc, Seg_img_blue_cc, Mat(), Point(-1, -1), dilate_size);  // ��ͼ�����ͽ��жϿ�����ճ��
		//erode(Seg_img_blue_cc, Seg_img_blue_cc, Mat(), Point(-1, -1), 2);   // ��ͼ���е����������޳�

		vector<vector<Point> > contours;
		findContours(Seg_img_blue_cc, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);   // �ҳ���ͨ�������������������
		cout << "contours.size()" << contours.size( ) << endl;
		// ����ͼ�������е���ͨ��������
		Mat result(Seg_img_blue_cc.size(), CV_8U, Scalar(0));
		drawContours(result, contours,      //��������
			-1, // �������е�����
			Scalar(255), // �ð��߻���
			1); // �����ߵĴ�ϸΪ1


		putText(result,"contours", Point(3,15), 1, 1, Scalar(255), 1);
		
		cvtColor(result, result, CV_GRAY2BGR);
		co_image.push_back(result);

		namedWindow("Contours");
		imshow("Contours", result);  // ��ʾͼ�������е���ͨ������
		

		// ������ͨ�������ߵĳ��ȴ�С��������ͨ�������
		Mat original(Seg_img_blue_cc.size(), CV_8U, Scalar(0));
		if (contours.size() > 3)
		{
			cnsr(contours);			// ����������С��ͨ�������
		}
		else
		{
			cnsr_mini(contours);	// ����̫С��С��ͨ�������
		}
			
		
		drawContours(original, contours,
			-1, // �����µ����е�����
			Scalar(255), // �ð��߻���
			1); // �����ߵĴֶ�Ϊ1
		namedWindow("Contours after noise reduced");
		imshow("Contours after noise reduced", original);				// ����ȥ����ͨ�����������ͨ��
		
		putText(original, "After Contours Noise Reduced", Point(3, 15), 1, 1, Scalar(255), 1);

		cvtColor(original, original, CV_GRAY2BGR);
		co_image.push_back(original);


		// ������Ӿ��εĳ����ֵ��С����ȥ�봦�����
		if (contours.size() > 2)
		{
			cout << "before rnsr:" << contours.size() << endl;
			rnsr(contours);			// ������Ӿ��γ���Ȱ�����ȥ��
		}
		else
		{
			cout << "without rnsr process:" << contours.size() << endl;
			rnsr_mini(contours);		// ������Ӿ��γ���Ȱ���Сֵȥ��
		}


	
		// �������������ͨ����Ӿ���ȥ����Ŀ��ͼ��
		Mat obj_rec_thr = Mat::zeros(Seg_img_red.size(), CV_8UC3);

		Rect *r = new Rect[contours.size()];         // ������Ӿ�������
		for (unsigned int i = 0; i < contours.size(); i++)
		{
			r[i] = boundingRect(Mat(contours[i]));                // ��ȡ��ǰ��ͨ����������Ӿ���;
			rectangle(obj_rec_thr, r[i], Scalar(0, 0, 255), 1);   // �ú�ɫ���ο����Ӿ��α�ʾ����
		}

		delete[] r;

		imshow("obj_rec_thr", obj_rec_thr);

		putText(obj_rec_thr, "After Rectange Ratio Noise Reduced", Point(3, 15), 1, 1, Scalar(255, 255, 255), 1);
		co_image.push_back(obj_rec_thr);




		// ����ͬһ����ͨ���ں������������ı�ֵ��С����ȥ�봦�����
		cout << "before pxsnsr:"<< contours.size() << endl;
		pxsnsr(contours, Seg_img_red, Seg_img_blue);			//��Ȥ�����غͱ�ֵȥ��	

		vector<Point2i> NormMatches;
		NormMatche( contours, Matches );


		
		// �������������ͨ��������������ı�ֵȥ����Ŀ��ͼ��
		Mat obj_px_thr = Mat::zeros(Seg_img_red.size(), CV_8UC3);

		Rect *rc = new Rect[contours.size()];         // ������Ӿ�������
		for (unsigned int i = 0; i < contours.size(); i++)
		{
			rc[i] = boundingRect(Mat(contours[i]));          // ��ȡ��ǰ��ͨ����������Ӿ���;
			rectangle(obj_px_thr, rc[i], Scalar(10, 128, 255), 1);   // �ú�ɫ���ο����Ӿ��α�ʾ����
		}

		delete[] rc;
		vector<Point2i>::const_iterator it = Matches.begin( );
		Mat obj_px_thr2 = obj_px_thr.clone();
		while ( it != Matches.end( ) ) {
			circle( obj_px_thr2, cvPoint( ( int )it->x, ( int )it->y ), 0, CV_RGB( 0, 255, 0 ), 2, 8, 0 );
			++it;
		}
		imshow("obj_px_thr", obj_px_thr);
		//ofile << frame << "\t" << Matches.size() << endl;
		putText(obj_px_thr, "After Red-Blue Ratio Noise Reduced", Point(3, 15), 1, 1, Scalar(255, 255, 255), 1);
		co_image.push_back(obj_px_thr);
		co_image.push_back(obj_px_thr2);
		co_image.push_back(img_scene_fea_match);
		// ���괦�����,������ͨ����������ȡĿ�����ĵ�
		ExtraCoordinates(contours, coordinates);
		Mat img_scene_2 = img_scene.clone();   //clone��ֵ��ʽ
		if (coordinates.empty())
			ofile << endl;
		else
		{
			Point2i point = coordinates.back();
			/*ofile << point.x << "\t" << point.y << endl;*/
			circle( img_scene_2, Point( point.x, point.y ), 2, Scalar( 0, 255, 0 ), 2 );
			putText(img_scene_2, intToString(point.x) + "," + intToString(point.y), Point(point.x + 5, point.y), 1, 1, Scalar(0, 255, 0), 2);
		}
		putText(img_scene_2, "Detection Result", Point(3, 33), 1, 1, Scalar(255, 255, 255), 1);
		co_image.push_back(img_scene_2);
		co_image.push_back(img_scene_fea);

		// �м䴦�������ĺϲ���ʾ
		Mat Image_com;
		Image_com = CombineMultiImages(co_image, 3, 4, 3, img_scene.cols, img_scene.rows);
		imshow("ProjectCarTopLanding_ColorBased", Image_com);
		resize(Image_com, Image_com, s);
		writer << Image_com;
			
		// ����������ʱ ��ͣ-����-�ն˲���
		if ((frame == 27)){
			imwrite( "img_object_fea.png", img_object_fea );		//ģ��
			imwrite("img_scene.png", img_scene);				//ԭͼ
			imwrite("Seg_img_blue_disp.png", Seg_img_blue_disp);//��ɫͨ��
			imwrite("Seg_img_red_disp.png", Seg_img_red_disp);	//��ɫͨ��
			imwrite("result.png", result);						//�ҳ���ͨ��
			imwrite("original.png", original);					//��ͨ��ȥ��1
			imwrite("obj_rec_thr.png", obj_rec_thr);			//��ͨ��ȥ��2
			imwrite("obj_px_thr.png", obj_px_thr);				//��ͨ��ȥ��3
			


			imwrite("img_scene_fea.png", img_scene_fea);		//ԭͼ��������
			imwrite("img_scene_fea_match.png", img_scene_fea_match);//ԭͼ��ɸѡ���������

			imwrite("obj_px_thr2.png", obj_px_thr2);			//�淶���������
			ofile << "ģ��ͼ��һ�μ�⵽��������������" << a << endl;
			ofile << "���������������" << Matches.size() << endl;
			waitKey(0);
			return 0;
		}
		switch (waitKey(10)){

		case 27: //'esc' key has been pressed, exit program.
			return 0;
		case 112: //'p' has been pressed. this will pause/resume the code.
			pause = !pause;
			if (pause == true){
				cout << "Code paused, press 'p' again to resume" << endl;
				while (pause == true){
					//stay in this loop until 
					switch (waitKey()){
						//a switch statement inside a switch statement? Mind blown.
					case 112:
						//change pause back to false
						pause = false;
						cout << "Code resumed." << endl;
						break;
					}
				}
			}


		}
		cout << endl;    // ����	
	}
	return 0;
}
