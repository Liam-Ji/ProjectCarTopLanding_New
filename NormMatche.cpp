# pragma warning (disable:4819)
#include "objectDetect.h"

void NormMatche( vector<vector<Point>> &contours, vector<Point2i> &Matches )
{

	Rect *r = new Rect[contours.size( )];   // ������Ӿ�������
	for ( unsigned int i = 0; i < contours.size( ); i++ ) {
		r[i] = boundingRect( Mat( contours[i] ) );   // boundingRect��ȡ�����Ӿ���;
		vector<Point2i>::const_iterator it = Matches.begin( );
		while ( it != Matches.end( ) ) {
			if ( it->x > r[i].x && it->x < r[i].x + r[i].width && it->y > r[i].y && it->y < r[i].y + r[i].height )
				++it;
			else
				it = Matches.erase( it );   // ɾ����ǰ��ͨ������
		}
	}
	delete[] r;
}
