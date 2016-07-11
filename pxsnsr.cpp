/*利用兴趣区红蓝像素总合的比值除去噪声*/
/*输入的参数依次为，连通域，红色通道阈值分割结果，蓝色通道阈值分割结果*/
# pragma warning (disable:4819)

#include "objectDetect.h"

void pxsnsr(vector<vector<Point>> &contours, Mat &Seg_img_red, Mat &Seg_img_blue)
{
	Rect *r = new Rect[contours.size()];   // 定义外接矩形数组
	Mat obj_rec_thr = Mat::zeros(Seg_img_red.size(), CV_8UC3);

	for (unsigned int i = 0; i < contours.size(); i++)
	{
		r[i] = boundingRect(Mat(contours[i]));// boundingRect获取这个外接矩形
	}


	/// 绘出轮廓及其凸包
	Mat Seg_img_blue_hull(Seg_img_blue.size(), CV_8U, Scalar(0));
	vector<vector<Point> >hull(contours.size());
	for (unsigned int i = 0; i< contours.size(); i++)
	{
		convexHull(Mat(contours[i]), hull[i], false);
		fillConvexPoly(Seg_img_blue_hull, &hull[i][0], hull[i].size(), Scalar(1));
	}

	erode(Seg_img_blue_hull, Seg_img_blue_hull, Mat(), Point(-1, -1), dilate_size);   // 对图像进行之前的膨胀恢复


	Seg_img_blue_hull.convertTo(Seg_img_blue_hull, CV_32F);

	imshow("Seg_img_blue _hull", Seg_img_blue_hull);


	double Evaluation = 0;      // 对潜在目标的打分值初始化
	double Evaluation_max = 0;   // 对潜在目标的打分最大值初始化
	int index_best = -1;         // 最优轮廓标记 <注意此处初始值应为-1>
	int index = 0;               // 轮廓标记

	cout << contours.size() << endl;

	vector<vector<Point>>::const_iterator itContours = contours.begin();
	//for (; itContours != contours.end(); ++itContours)
	while (itContours != contours.end())
	{
		Mat imageROI_red = Seg_img_red(cv::Rect(r[index].x, r[index].y, r[index].width, r[index].height));
		Mat imageROI_blue = Seg_img_blue_hull(cv::Rect(r[index].x, r[index].y, r[index].width, r[index].height));

		Mat imageROI_red_blue = imageROI_red + imageROI_blue;
		threshold(imageROI_red_blue, imageROI_red_blue, 1, 1, THRESH_BINARY);   // 进行阈值分割（大于阈值1等于2时候取 1）


		Scalar s1 = sum(imageROI_red_blue);
		Scalar s2 = sum(imageROI_blue);
		double sum_red = s1.val[0];    // 获取图像块包含在蓝色中的红色像素数量
		double sum_blue = s2.val[0];    // 获取图像块中蓝色像素数量

		double pixel_sum_rate = rate(sum_red, sum_blue);  // 计算图像块中红蓝像素的比例
		
		cout << "sum_red:" << sum_red << "\t" << "," << "sum_blue:" << sum_blue << "\t";
		cout << "pixel_sum_rate:" << pixel_sum_rate << endl;


	    // 将红蓝像素比太低的连通域轮廓删除
		if ((pixel_sum_rate < threshold_value_pixel_rate) || (sum_red < 12))  // 如当前轮廓不含一定的红色像素
		{
			itContours = contours.erase(itContours);  // 删除当前轮廓 <特别注意：删除当前轮廓后迭代器自动指向后一个>
			index++;
			continue;
		}
		else     // 如当前轮廓含一定的红色像素
		{
			int aaa = contours.size();  // 用于设置条件断点

			Evaluation = sum_red;       // 目标轮廓简单评价值定义
			if (Evaluation > Evaluation_max)
			{
				Evaluation_max = Evaluation;
				index_best++;

				cout << "index_best - " << index_best  << endl;
			}

			index++;
			itContours++;  // 继续检索下一个轮廓 <此处必须要代码指定迭代器转向后一个轮廓>
		}

	
	}



	int ttt = contours.size();    // 用于设置条件断点


   // 如果仍然有大于一个潜在目标则选取最优的一个
	if (ttt > 1) 
	{
		int index = 0;
		vector<vector<Point>>::const_iterator itContours_2 = contours.begin();
		while(itContours_2 != contours.end())
		{
			if (index != index_best)
				itContours_2 = contours.erase(itContours_2);
			else
				++itContours_2; 

			index++;
		}

	}



	delete[] r;



}