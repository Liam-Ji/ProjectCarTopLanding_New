# pragma warning (disable:4819)
/*M///////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014, sheng
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//M*/


#include "objectDetect.h"

#include <vector>

#include <assert.h>

#include <opencv2/opencv.hpp>



/**
* @brief CombineMultiImages  Combine the input images to a single big image.
* @param Images         The input images.
* @param NumberOfRows   The number of the rows to put the input images.
* @param NumberOfCols   The number of the cols to put the input images.
* @param Distance       The distance between each image.
* @param ImageWidth     The width of each image in the big image.
* @param ImageHeight    The height of each image in the big image.
* @return      The big image if the operation is successed;
*              otherwise return empty image.
*
* @author sheng
* @date 2015-03-24
* @version 0.1
*
* @history
*     <author>       <date>         <version>        <description>
*      sheng       2015-03-24          0.1           build the function
*
*/
cv::Mat CombineMultiImages(const std::vector<cv::Mat>& Images,
	const int NumberOfRows,
	const int NumberOfCols,
	const int Distance,
	const int ImageWidth,
	const int ImageHeight)
{
	// return empty mat if the Number of rows or cols is smaller than 1.
	assert((NumberOfRows > 0) && (NumberOfCols > 0));
	if ((NumberOfRows < 1) || (NumberOfCols < 1))
	{
		std::cout << "The number of the rows or the cols is smaller than 1."
			<< std::endl;
		return cv::Mat();
	}


	// return empty mat if the distance, the width or the height of image
	// is smaller than 1.
	assert((Distance > 0) && (ImageWidth > 0) && (ImageHeight > 0));
	if ((Distance < 1) || (ImageWidth < 1) || (ImageHeight < 1))
	{
		std::cout << "The distance, the width or the height of the image is smaller than 1."
			<< std::endl;
		return cv::Mat();
	}


	// Get the number of the input images
	const int NUMBEROFINPUTIMAGES = Images.size();


	// return empty mat if the number of the input images is too big.
	assert(NUMBEROFINPUTIMAGES <= NumberOfRows * NumberOfCols);
	if (NUMBEROFINPUTIMAGES > NumberOfRows * NumberOfCols)
	{
		std::cout << "The number of images is too big." << std::endl;
		return cv::Mat();
	}


	// return empty mat if the number of the input images is too low.
	assert(NUMBEROFINPUTIMAGES > 0);
	if (NUMBEROFINPUTIMAGES < 1)
	{
		std::cout << "The number of images is too low." << std::endl;
		return cv::Mat();
	}


	// create the big image
	const int WIDTH = Distance * (NumberOfCols + 1) + ImageWidth * NumberOfCols;
	const int HEIGHT = Distance * (NumberOfRows + 1) + ImageHeight * NumberOfRows;

	cv::Scalar Color(255, 255, 255);
	if (Images[0].channels() == 1)
	{
		Color = cv::Scalar(255);
	}
	cv::Mat ResultImage(HEIGHT, WIDTH, Images[0].type(), Color);



	// copy the input images to the big image
	for (int Index = 0; Index < NUMBEROFINPUTIMAGES; Index++)
	{

		assert(Images[Index].type() == ResultImage.type());
		if (Images[Index].type() != ResultImage.type())
		{
			std::cout << "The No." << Index << "image has the different type."
				<< std::endl;
			return cv::Mat();
		}


		// Get the row and the col of No.Index image
		int Rows = Index / NumberOfCols;
		int Cols = Index % NumberOfCols;

		// The start point of No.Index image.
		int StartRows = Distance * (Rows + 1) + ImageHeight * Rows;
		int StartCols = Distance * (Cols + 1) + ImageWidth * Cols;

		// copy  No.Index image to the big image
		cv::Mat ROI = ResultImage(cv::Rect(StartCols, StartRows,
			ImageWidth, ImageHeight));
		cv::resize(Images[Index], ROI, cv::Size(ImageWidth, ImageHeight));

	}

	return ResultImage;


}