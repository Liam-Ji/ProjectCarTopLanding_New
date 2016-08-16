/*用于对图像进行预处理*/
/*第一个参数为读取的帧，第二、三个参数分别为红蓝通道阈值分割结果*/
# pragma warning (disable:4819)
#include "objectDetect.h"

void preprocess(Mat &img_scene, Mat &Seg_img_red, Mat &Seg_img_blue)
{
	Mat HSV;
	cv::cvtColor(img_scene, HSV, CV_BGR2HSV);        // 将RGB空间转换为HSV空间

	vector<Mat> channels;
	split(img_scene, channels);     // 分离色彩通道, 把一个3通道图像转换成3个单通道图像
	Mat img_scene_BlueChannel = channels.at(0);   // 蓝通道
	Mat img_scene_GreenChannel = channels.at(1);  // 绿通道
	Mat img_scene_RedChannel = channels.at(2);    // 红通道

	//vector<Mat> channels_HSV;
	//split(HSV, channels_HSV);     // 分离色彩通道, 把一个3通道图像转换成3个单通道图像
	//Mat img_HSV_VChannel = channels_HSV.at(0);    //  V通道
	//Mat img_HSV_SChannel = channels_HSV.at(1);    // S通道
	//Mat img_HSV_HChannel = channels_HSV.at(2);    // H通道


	/*图像类型转化为32位的浮点型*/
	Mat red_float;
	Mat green_float;
	Mat blue_float;
	img_scene_RedChannel.convertTo(red_float, CV_32F);
	img_scene_GreenChannel.convertTo(green_float, CV_32F);
	img_scene_BlueChannel.convertTo(blue_float, CV_32F);


	/*求红色通道和另外两个通道对应位置像素值得比值*/
	Mat red_green;
	Mat red_blue;
	Mat blue_red;
	Mat blue_green;
	divide(red_float, green_float + 0.00001, red_green, 1, -1);  // 注意预防被除数为0
	divide(red_float, blue_float + 0.00001, red_blue, 1, -1);
	divide(blue_float, red_float + 0.00001, blue_red, 1, -1);
	divide(blue_float, green_float + 0.00001, blue_green, 1, -1);

	/*对后续相乘时候其两个乘数符号均为负的时候进行排除*/
	Mat red_green_new;
	Mat red_blue_new;
	Mat blue_red_new;
	Mat blue_green_new;
	threshold(red_green - 1, red_green_new, 0, 0, CV_THRESH_TOZERO);       //  当图像元素值 > 0 时才能取原值，否则为0
	threshold(red_blue - 1, red_blue_new, 0, 0, CV_THRESH_TOZERO);         //  当图像元素值 > 0 时才能取原值，否则为0
	threshold(blue_red - 1, blue_red_new, 0, 0, CV_THRESH_TOZERO);         //  当图像元素值 > 0 时才能取原值，否则为0
	threshold(blue_green - 1, blue_green_new, 0, 0, CV_THRESH_TOZERO);     //  当图像元素值 > 0 时才能取原值，否则为0


	/*将比值结果合并，将红色通道与蓝色通道比值的结果和红色通道与绿色通道比值的结果相乘，蓝色通道作同样的处理*/
	Mat red_multi;
	Mat blue_multi;
	multiply(red_green_new, red_blue_new, red_multi);
	multiply(blue_red_new, blue_green_new, blue_multi);

	/*对乘积结果取阈值，生成二值图像*/
	threshold(red_multi, Seg_img_red, threshold_value_red, 1, THRESH_BINARY);        // 红色通道进行阈值分割（大于阈值时候取 1）
	threshold(blue_multi, Seg_img_blue, threshold_value_blue, 1, THRESH_BINARY);     // 蓝色通道进行阈值分割（小于阈值时候取 1）
	


	///*用于显示图像察看效果*/
	//imshow("img_scene_red_rate", Seg_img_red * 255);
	//imshow("img_scene_blue_rate", Seg_img_blue * 255);


	/////*彩色图像阈值分割后的结果分析代码*/
	//Mat Overlap_img = img_scene.clone();
	//int nr = Overlap_img.rows;  // 行数
	//int nc = Overlap_img.cols;  // 列数

	//for (int j = 0; j<nr; j++) 
	//{
	//	for (int i = 0; i<nc; i++) 
	//	{
	//		if ((Seg_img_red.at<float>(j, i) == 1) && (Seg_img_blue.at<float>(j, i)) == 0)
	//		{
	//			Overlap_img.at<cv::Vec3b>(j, i)[0] = 0;   // 原图中的红色部分用纯红表示
	//			Overlap_img.at<cv::Vec3b>(j, i)[1] = 0;
	//			Overlap_img.at<cv::Vec3b>(j, i)[2] = 255;
	//		}
	//		else if (((Seg_img_red.at<float>(j, i) == 0) && (Seg_img_blue.at<float>(j, i)) == 1))
	//		{
	//			Overlap_img.at<cv::Vec3b>(j, i)[0] = 255;  // 原图中的蓝色部分用纯蓝表示
	//			Overlap_img.at<cv::Vec3b>(j, i)[1] = 0;
	//			Overlap_img.at<cv::Vec3b>(j, i)[2] = 0;
	//		}
	//		else if (((Seg_img_red.at<float>(j, i) == 1) && (Seg_img_blue.at<float>(j, i)) == 1))
	//		{
	//			Overlap_img.at<cv::Vec3b>(j, i)[0] = 0;   // 原图中的交叉部分用纯绿表示
	//			Overlap_img.at<cv::Vec3b>(j, i)[1] = 255;
	//			Overlap_img.at<cv::Vec3b>(j, i)[2] = 0;
	//		}
	//		else if (((Seg_img_red.at<float>(j, i) == 0) && (Seg_img_blue.at<float>(j, i)) == 0))
	//		{
	//			Overlap_img.at<cv::Vec3b>(j, i)[0] = 0;   // 原图中的背景部分用纯黑表示
	//			Overlap_img.at<cv::Vec3b>(j, i)[1] = 0;
	//			Overlap_img.at<cv::Vec3b>(j, i)[2] = 0;
	//		}
	//	}               
	//}

	//imshow("Overlap_img", Overlap_img);




}