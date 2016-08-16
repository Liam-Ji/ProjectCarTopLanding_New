/* 定义球两个数的比值的函数(小的数比上大的数) */
/* 传递两个浮点数,返回浮点数的比值结果 */
#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)>(b)?(b):(a))

double rate(double w, double h)
{
	return min(w, h) / max(w, h);
}