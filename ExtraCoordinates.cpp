# pragma warning (disable:4819)
#include "objectDetect.h"

int distance(Point2i &p, vector<Point>::const_iterator &q)
{
	int s;
	s = (int)sqrt((p.x - q->x)*(p.x - q->x) + (p.y - q->y)*(p.y - q->y));
	return s;
}

void ExtraCoordinates(vector<vector<Point>> &contours, vector<Point2i> &coordinates)
{
	Point2i lastcoo, currentcoo;
	Point2i acoo, bcoo;
	Rect *r = new Rect[contours.size()];   // 定义外接矩形数组

	vector<Point2i> coo;
	for (unsigned int i = 0; i < contours.size(); i++)
	{
		r[i] = boundingRect(Mat(contours[i]));   // boundingRect获取这个外接矩形;
		currentcoo.x = r[i].x + r[i].width / 2;
		currentcoo.y = r[i].y + r[i].height / 2;
		coo.push_back(currentcoo);      // 在coordinates末尾插入currentcoo数据
	}

	if (contours.size() == 1)
	{
		coordinates.push_back(currentcoo);  // 在coordinates末尾插入currentcoo数据
	}


	if (contours.size() == 0)
	{
		currentcoo.x = -1;   // 当前帧没有检测到目标时指定目标位置 （-1，-1）
		currentcoo.y = -1;
		coordinates.push_back(currentcoo);  // 在coordinates末尾插入currentcoo数据
	}
	delete[] r;


}




