/*符号函数，用来判断图像数据元素符号的正负*/

# pragma warning (disable:4819)
#include "objectDetect.h"

Mat sign(Mat &img)
{
	Mat img_result_1;
	Mat img_result_2;
	threshold(img, img_result_1, 0, 1, THRESH_BINARY);    // 通道进行阈值分割（大于阈值0时候取 1，其它为0）
	threshold(img, img_result_2, 0, -1, THRESH_BINARY_INV);    // 通道进行阈值分割（大于阈值0时候取 0，其它为-1）

	Mat img_result;
	img_result = img_result_1 + img_result_2;  // 注意：当前img中值为0的元素操作后最终取值为-1

	return img_result;


}