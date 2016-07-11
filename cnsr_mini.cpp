/*用于去除连通域的部分噪声利用最低阈值*/
# pragma warning (disable:4819)
#include "objectDetect.h"

void cnsr_mini(vector<vector<Point>> &contours)
{
	//int *ca = new int[contours.size()];	 // 定义连通域边界长度排序数组

	//vector<vector<Point>>::const_iterator itContours = contours.begin();
	//for (unsigned int i = 0; i < contours.size(), itContours != contours.end(); ++i, ++itContours)
	//{
	//	ca[i] = itContours->size();     // 读取连通域边界长度信息
	//}
	//sort(ca, ca + contours.size());	    // 按连通域边界像素的多少排序

	//int s = (int)((double)contours.size() * threshold_value_sum);

	//if ((s < 2) && (contours.size() > 2))  s = 3;  // 保险处理 <contours.size() 已经是大于 3>
	//if (s > 6)  s = 6;  // 最多保留 6 个潜在目标

	//unsigned int threshhold_con = (ca[contours.size() - s - 1] + ca[contours.size() - s]) / 2;

	vector<vector<Point>>::const_iterator itc = contours.begin();
	while (itc != contours.end())
	{
		if (itc->size() < threshold_value_sum_minimum)
			itc = contours.erase(itc);   // 删除当前连通域轮廓
		else
			++itc;
	}

	//delete[] ca;
}
