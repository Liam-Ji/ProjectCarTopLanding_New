#include <fstream>
#include <iostream>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define threshold_value_red 0.1				// 红色通道阈值
#define threshold_value_blue 0.3			// 蓝色通道阈值

#define dilate_size 3		                // 图像中蓝色部分的膨胀粘连尺寸

#define threshold_value_sum 0.3				//连通域边界像素总合阈值（(将轮廓按照比例去掉，取值范围0-1）
#define threshold_value_rate 0.5			//外接矩形长宽比阈值（取值范围0-1）

#define threshold_value_sum_minimum 40		//连通域边界像素总合最低值(将太小的轮廓去掉)
#define threshold_value_rate_minimum 0.3	 //外接矩形长宽比阈值最低值(将长宽比 <0.3 太扁的轮廓去掉)
#define threshold_value_rate_maximum 0.6	 //外接矩形长宽比阈值最大值(长宽比 >0.7 的类正方形则全部保留)

#define threshold_value_pixel_rate 0.03		//连通域内红蓝像素比值阈值


extern bool ficoo;
double rate(double, double);														//求两个数的比值(小的数比上大的数)
string intToString(int number);
void preprocess(Mat &img_scene, Mat &Seg_img_red, Mat &Seg_img_blue);				//图像预处理
void cnsr(vector<vector<Point>> &contours);											//连通域按比例大小去噪
void cnsr_mini(vector<vector<Point>> &contours);                                     //连通域按最小阈值去噪
void rnsr(vector<vector<Point>> &contours);						    //外接矩形按长宽比按比例饿去噪
void rnsr_mini(vector<vector<Point>> &contours);					//外接矩形按长宽比按最小值去噪
void pxsnsr(vector<vector<Point>> &contours, Mat &Seg_img_red, Mat &Seg_img_blue);	//兴趣区像素和比值去噪

void ImageEnhance(Mat &img_scene, Mat &new_image);     //  对原始图像进行增强处理

Mat  sign(Mat &img);    // 符号函数

Mat CombineMultiImages(const std::vector<cv::Mat>& Images,
	const int NumberOfRows,
	const int NumberOfCols,
	const int Distance,
	const int ImageWidth,
	const int ImageHeight);

void ExtraCoordinates(vector<vector<Point>> &contours, vector<Point2i> &coordinates);

void NormMatche( vector<vector<Point>> &contours, vector<Point2i> &Matches );