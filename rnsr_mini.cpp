/*利用外界矩形长宽比去除噪声*/
/*第一个参数是连通域，第二个参数是红色通道阈值处理结果（仅用于获取图像大小来画图，最后可以去除）*/
# pragma warning (disable:4819)

#include "objectDetect.h"

void rnsr_mini(vector<vector<Point>> &contours)
{
	Rect *r = new Rect[contours.size()];         // 定义外接矩形数组
	double *ra = new double[contours.size()];	 // 定义外接矩形长宽比数组
	double *rb = new double[contours.size()];


	for (unsigned int i = 0; i < contours.size(); i++)
	{
		r[i] = boundingRect(Mat(contours[i]));          // 获取当前连通域轮廓的外接矩形;
		rb[i] = ra[i] = rate(r[i].width, r[i].height);	// 计算长宽比
	}

	// 绘制通过长宽比阈值限制后的外接矩形
	vector<vector<Point>>::const_iterator itContours = contours.begin();
	for (unsigned int i = 0; i < contours.size(), itContours != contours.end(); ++i)
	{
		if (rb[i] > threshold_value_rate_minimum)
		{
			++itContours;
		}
		else
			itContours = contours.erase(itContours);   // 删除当前太扁的连通域轮廓

	}

	delete[] r;
	delete[] ra;
	delete[] rb;



}