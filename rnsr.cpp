/*利用外界矩形长宽比去除噪声*/
/*第一个参数是连通域，第二个参数是红色通道阈值处理结果（仅用于获取图像大小来画图，最后可以去除）*/
# pragma warning (disable:4819)

#include "objectDetect.h"

void rnsr(vector<vector<Point>> &contours)
{
	Rect *r = new Rect[contours.size()];         // 定义外接矩形数组
	double *ra = new double[contours.size()];	 // 定义外接矩形长宽比数组
	double *rb = new double[contours.size()];


	for (unsigned int i = 0; i < contours.size(); i++)
	{
		r[i] = boundingRect(Mat(contours[i]));          // 获取当前连通域轮廓的外接矩形;
		rb[i] = ra[i] = rate(r[i].width, r[i].height);	// 计算长宽比
		cout << "width:" << r[i].width << "height:" << r[i].height << "rb[i]:" << rb[i] << endl;
	}

	sort(ra, ra + contours.size());		// 将外接矩形长宽比按大小进行排序

	int k = (int)((double)contours.size() * threshold_value_rate);
	if ((k < 2) && (contours.size() > 2))  k = 2;  // 保险处理 <contours.size() 已经是大于等于 2>
	if (k > 6)  k = 3;  // 最多保留 3 个潜在目标

	double threshold_rate = (ra[contours.size() - k - 1] + ra[contours.size() - k])/2;	//定义外接矩形长宽比阈值
	
	if (threshold_rate > threshold_value_rate_maximum)  // 如果轮廓都是类正方形则全部保留
		threshold_rate = threshold_value_rate_maximum;

	// 绘制通过长宽比阈值限制后的外接矩形
	vector<vector<Point>>::const_iterator itContours = contours.begin();
	for (unsigned int i = 0; i < contours.size(), itContours != contours.end(); ++i)
	{
		if ((rb[i] > threshold_rate) && ((rb[i] > threshold_value_rate_minimum)))
		{
			++itContours;
		}	
		else
			itContours = contours.erase(itContours);   // 删除当前连通域轮廓

	}
	
	delete[] r;
	delete[] ra;
	delete[] rb;

}