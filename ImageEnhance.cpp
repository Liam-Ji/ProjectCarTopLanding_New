/*对原始的视频图像进行增强*/

# pragma warning (disable:4819)

#include "objectDetect.h"

void IntensityEnhance(Mat &img_scene, Mat &new_image);   // 子函数的函数声明
void IntensityEnhance_fast(Mat &img_scene, Mat &new_image);   // 子函数的函数声明

void ImageEnhance(Mat &img_scene, Mat &new_image)
{
	Mat HSV;
	cv::cvtColor(img_scene, HSV, CV_BGR2HSV);    // 将RGB空间转换为HSV空间
	vector<Mat> splitBGR(HSV.channels());        //用来存储各通道图片的向量
	split(HSV, splitBGR);   //分割通道，存储到splitBGR中

	IntensityEnhance_fast(splitBGR[1], splitBGR[1]);        //对饱和度通道分别进行增强

	merge(splitBGR, new_image);  //合并通道
	cv::cvtColor(new_image, new_image, CV_HSV2BGR);    // 将RGB空间转换为HSV空间

	IntensityEnhance_fast(new_image, new_image);    // 整体进行亮度增强


}




// 内部子函数
void IntensityEnhance(Mat &img_scene, Mat &new_image)
{
	double alpha = 1.23;
	double beta = 14;

	if (img_scene.channels() == 3)    // 当输入图像为彩色图像时候
	{
		/// 执行运算 new_image(i,j) = alpha*image(i,j) + beta
		for (int y = 0; y < img_scene.rows; y++)
		{
			for (int x = 0; x < img_scene.cols; x++)
			{
				for (int c = 0; c < 3; c++)
				{
					new_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha*(img_scene.at<Vec3b>(y, x)[c]) + beta);
				}
			}
		}
	}

	else if (img_scene.channels() == 1)    // 当输入图像为灰度图像时候
	{
		for (int y = 0; y < img_scene.rows; y++)
		{
			for (int x = 0; x < img_scene.cols; x++)
			{
				new_image.at<uchar>(y, x) = saturate_cast<uchar>(alpha*(img_scene.at<uchar>(y, x)) + beta);
			}
		}
	}

	
}


// 内部子函数
void IntensityEnhance_fast(Mat &img_scene, Mat &new_image)
{
	double alpha = 1.23;
	double beta = 14;

	new_image = alpha*img_scene + beta;


}