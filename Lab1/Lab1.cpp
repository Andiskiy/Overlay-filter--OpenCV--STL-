// ConsoleApplication2.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <iostream>
#include <opencv\highgui.h>
#include "opencv\cv.h"

using namespace cv;
using namespace std;


const std::string WINDOW_NAME = "Application";
const int TRACKBAR_MAX = 100;

struct SUserdata {
	cv::Mat image;
	cv::Mat imageH;
};

vector<cv::Vec2f> clear_lines(vector<cv::Vec2f> x)
{
	int space = 3;

	while (x.size() != 4)
	{
		for (size_t i = 0; i < x.size(); i++)
		{
			float rho1 = x[i][0];

			for (size_t j = 0; j < x.size(); j++)
			{
				if (i != j)
				{
					float rho2 = x[j][0];

					if (fabs(rho1 - rho2) <= space)
					{
						x.erase(x.begin() + j);
					}
				}
			}
		}
		space++;
	}

	return x;
}

vector<cv::Point2f> find_points(vector<cv::Vec2f> x, vector<cv::Point2f> srcTri, Mat image)
{
	for (size_t i = 0; i < x.size(); i++)
	{
		for (size_t j = i + 1; j < x.size(); j++)
		{
			if (i != j)
			{
				double rho1 = x[i][0];
				double rho2 = x[j][0];

				double teta1 = x[i][1];
				double teta2 = x[j][1];

				double X = (rho1 / sin(teta1) - rho2 / sin(teta2)) / (cos(teta1) / sin(teta1) - cos(teta2) / sin(teta2));

				if (X > 0 && X < image.cols)
				{
					double Y = rho2 / sin(teta2) - cos(teta2) / sin(teta2) * X;

					if (Y > 0 && Y < image.rows)
					{
						srcTri.push_back(cv::Point2f(X, Y));
					}
				}

			}
		}
	}

	return srcTri;
}

void affine_transforme(){
	Mat image, dst, color_dst;
	vector<cv::Point2f> srcTri, dstTri;

	image = imread("E:/4.jpg");


	Canny(image, dst, 50, 200, 3);
	cvtColor(dst, color_dst, CV_GRAY2BGR);
	//imshow("dst", dst);
	//imshow("color_dst", color_dst);
	//waitKey();
	vector<cv::Vec2f> lines;
	vector<cv::Vec2f> new_lines;

	cv::HoughLines(dst, lines, 1, CV_PI / 180, 100);
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	cout << lines[i].val[0] << "  " << lines[i].val[1]<< endl;
	//	
	//	//if (lines[i].val[0] <= 0)
	//	//	lines[i].val[0] = -lines[i].val[0];
	//	//if (lines[i].val[1] <= 0)
	//	//	lines[i].val[1] = -lines[i].val[1];
	//}
	new_lines = clear_lines(lines);

	for (size_t i = 0; i < new_lines.size(); i++)
	{
		float rho = new_lines[i][0];
		float theta = new_lines[i][1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(color_dst, pt1, pt2, cv::Scalar(0, 0, 255), 3, 8);
	}

	imshow("lines", color_dst);

	srcTri = find_points(new_lines, srcTri, image);

	dstTri.push_back(cv::Point2f(0, 0));
	dstTri.push_back(cv::Point2f(image.cols, 0));
	dstTri.push_back(cv::Point2f(0, image.rows));
	dstTri.push_back(cv::Point2f(image.cols, image.rows));

	//dstTri.push_back(cv::Point2f(image.cols/4, image.rows / 2));
	//dstTri.push_back(cv::Point2f(image.cols, 0));
	//dstTri.push_back(cv::Point2f(0, image.rows));
	//dstTri.push_back(cv::Point2f(image.cols, image.rows));


	Mat Homo = findHomography(srcTri, dstTri);
//	cv::imshow("gg", image);
	cv::warpPerspective(image, image, Homo, cv::Size(image.cols, image.rows));
	cv::imshow("Affine", image);

	waitKey();

}


//GaussianBlur
void trackbar_GaussianBlur(int pos, void *userdata) {
	cv::Mat image = ((SUserdata*)userdata)->image;//ссылка на изображение
	cv::Mat blurred;
	if (pos == 0)
		pos++;
	cv::GaussianBlur(image, blurred, cv::Size(-1, -1), double(pos) / double(TRACKBAR_MAX) * 10.0);
	cv::imshow(WINDOW_NAME, blurred);
}

//medianBlur
void trackbar_medianBlur(int pos, void *userdata) {
	cv::Mat image = ((SUserdata *)userdata)->image;
	cv::Mat blurred;
	if ((pos % 2) == 0)
		pos++;
	cv::medianBlur(image, blurred, pos);
	cv::imshow(WINDOW_NAME, blurred);
}

//bilateralFilter
int Diameter = 0;
int sigmaColor = 0;
int sigmaSpace = 0;
void trackbar_bilateralFilter(int, void *userdata) {
	cv::Mat image = ((SUserdata *)userdata)->image;
	cv::Mat blurred;
	double d;
	d = double(Diameter) / double(TRACKBAR_MAX) * 10.0;
	cv::bilateralFilter(image, blurred, int(d), int(sigmaColor), int(sigmaSpace));
	cv::imshow(WINDOW_NAME, blurred);
}


//Blur
int parametrOne = 0;
int parametrSecond = 0;
void trackbar_Blur(int, void *userdata) {
	cv::Mat image = ((SUserdata *)userdata)->image;
	cv::Mat blurred;
	if (parametrOne == 0)
		parametrOne = 1;
	if (parametrSecond == 0)
		parametrSecond = 1;
	blur(image, blurred, cv::Size(parametrOne, parametrSecond));
	cv::imshow(WINDOW_NAME, blurred);
}

//Morphology Operations
int morph_elem = 0;
int morph_size = 0;
int morph_operator = 0;
int const max_operator = 4;
int const max_elem = 2;
int const max_kernel_size = 21;
void Morphology_Operations(int, void *userdata){
	cv::Mat image = ((SUserdata *)userdata)->image;
	cv::Mat dst;
	// Since MORPH_X : 2,3,4,5,6
	int operation = morph_operator + 2;

	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1),
		Point(morph_size, morph_size));



	/// Apply the specified morphology operation
	morphologyEx(image, dst, operation, element);
	cv::imshow(WINDOW_NAME, dst);
}



//contrast and brightness control
int alpha = 0; // contrast control
int beta = 0; // brightness control
void trackbar_contrastBrightness(int, void *userdata){
	cv::Mat image = ((SUserdata *)userdata)->image;
	cv::Mat dst = Mat::zeros(image.size(), image.type());
	double Alpha = double(alpha) / double(TRACKBAR_MAX) * 2 + 1;
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				dst.at<Vec3b>(y, x)[c] =
					saturate_cast<uchar>(Alpha*(image.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
	cv::imshow(WINDOW_NAME, dst);
}


// hough lines
void Func_Hough(void *userdata)
{
	cv::Mat image = ((SUserdata *)userdata)->imageH;
	cv::Mat dst, color_dst;
	//детектирование линий
	Canny(image, dst, 50, 200, 3);

	//конвертируем в цветное изображение
	cvtColor(dst, color_dst, CV_GRAY2BGR);

	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(color_dst, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
	}

	imshow( WINDOW_NAME, image );
	imshow("HoughLines", color_dst);
	//cv::imshow(WINDOW_NAME, dst);
	
}


void Func_hisograms(void *userdata)
{
	cv::Mat image = ((SUserdata *)userdata)->image;
	cv::Mat dst;

	vector<Mat> bgr_planes;
	split(image, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	// диапазон для B,G,R
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	// вычисление гистограммы
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Рисование гистограммы для B, G и R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	// [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	// рисование для каждого канала
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	// Display
	imshow("calcHist Demo", histImage);
}

void Func_EqualizeHist(void *userdata)
{
	cv::Mat image = ((SUserdata *)userdata)->image;
	cv::Mat dst;

	vector<Mat> bgr_planes;
	split(image, bgr_planes);
	equalizeHist(bgr_planes[0], bgr_planes[0]);
	equalizeHist(bgr_planes[1], bgr_planes[1]);
	equalizeHist(bgr_planes[2], bgr_planes[2]);
	merge(bgr_planes,dst);
	Func_hisograms(&dst);
	imshow("EqualizeHist",dst);
}

void Overlay_image(void *userdata)
{
	cv::Mat image2= ((SUserdata *)userdata)->image;	
	IplImage* image = cvLoadImage("E:/logo.jpg");
	Mat image1;
	cvSetImageROI(image, cvRect(500, 150, 150, 250));
	image1 = image;
	for (int y = image2.rows / 2 - image1.rows / 2; y < image2.rows / 2 + image1.rows / 2; y++)
	{
		for (int x = image2.cols / 2 - image1.cols / 2; x < image2.cols / 2 + image1.cols / 2; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				image2.at<Vec3b>(y, x)[c] = image1.at<Vec3b>(y - image2.rows /2  + image1.rows / 2, x - image2.cols / 2 + image1.cols / 2)[c];
			}
		}
	}

	cvResetImageROI(image);
	imshow("NEW_Image", image2);
}


// MAIN
int main()
{
	setlocale(LC_ALL, "RUS");
	cv::Mat image, image2;
	image = imread("E:/logo.jpg");
	image2 = imread("E:/1.jpg");
	SUserdata userdata, imageFH;
	userdata.image = image;
	imageFH.imageH =image2;
	int num; //для выбора задания в case
	cv::namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);
	moveWindow(WINDOW_NAME, 0, 0);
	bool flag = true;
	while (flag ==true){
		cout << "1 - Фильтр GaussianBlur\n2 - Медианный фильтр\n3 - Билатеральный фильтр\n4 - Фильтр Blur";
		cout << "\n5 - Морфологические операции\n6 - Яркость, контраст изображения\n7 - Вписать в одно изображение другое";
		cout << "\n8 - Преобразование Хау\n9 - Гистрограмма\n10 - Эквализация Гистрограммы\n";
		cout <<"11 - Выровнять изображение\n\nвыберите задание : ";
		cin >> num;

		switch (num)
		{
		case 1:
			cv::createTrackbar("Gaussian", WINDOW_NAME, NULL, TRACKBAR_MAX, trackbar_GaussianBlur, (void *)&userdata);
			flag = false;
			break;

		case 2:
			cv::createTrackbar("Median", WINDOW_NAME, NULL, TRACKBAR_MAX, trackbar_medianBlur, (void *)&userdata);
			flag = false;
			break;

		case 3:
			cv::createTrackbar("Diameter", WINDOW_NAME, &Diameter, TRACKBAR_MAX, trackbar_bilateralFilter, (void *)&userdata);
			cv::createTrackbar("SigmaColor", WINDOW_NAME, &sigmaColor, 1000, trackbar_bilateralFilter, (void *)&userdata);
			cv::createTrackbar("SigmaSpace", WINDOW_NAME, &sigmaSpace, 1000, trackbar_bilateralFilter, (void *)&userdata);
			flag = false;
			break;

		case 4:
			cv::createTrackbar("BlurOne", WINDOW_NAME, &parametrOne, TRACKBAR_MAX, trackbar_Blur, (void *)&userdata);
			cv::createTrackbar("BlurSecond", WINDOW_NAME, &parametrSecond, TRACKBAR_MAX, trackbar_Blur, (void *)&userdata);
			flag = false;
			break;

		case 5:
			createTrackbar("Operator:\n 0: Opening - 1: Closing\n 2: Gradient - 3: Top Hat \n 4: Black Hat",
				WINDOW_NAME, &morph_operator,
				4, Morphology_Operations, (void *)&userdata);

			/// Create Trackbar to select kernel type
			createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", WINDOW_NAME,
				&morph_elem, max_elem,
				Morphology_Operations, (void *)&userdata);

			/// Create Trackbar to choose kernel size
			createTrackbar("Kernel size:\n 2n +1", WINDOW_NAME,
				&morph_size, max_kernel_size,
				Morphology_Operations, (void *)&userdata);

			//Morphology_Operations( 0, 0 );
			flag = false;
			break;

		case 6:
			cv::createTrackbar("Alpha", WINDOW_NAME, &alpha, TRACKBAR_MAX, trackbar_contrastBrightness, (void *)&userdata);
			cv::createTrackbar("Beta", WINDOW_NAME, &beta, TRACKBAR_MAX, trackbar_contrastBrightness, (void *)&userdata);
			flag = false;
			break;

		case 7:
			Overlay_image((void *)&userdata);
			flag = false;
			break;


		case 8:
			Func_Hough((void *)&imageFH);
			flag = false;
			break;


		case 9:
			Func_hisograms((void *)&userdata);
			flag = false;
			break;

		case 10:
			Func_EqualizeHist((void *)&userdata);
			flag = false;
			break;

		case 11:
			affine_transforme();
			flag = false;
			break;

		default:
			cout << "\n\nНеверное значение!\n\n";
			break;
		}

	}

	waitKey(0);
	return 0;
}