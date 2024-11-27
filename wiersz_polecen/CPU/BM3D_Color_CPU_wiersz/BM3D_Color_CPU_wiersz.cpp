/*
© Marcin Wodejko 2024.
marwod@interia.pl
*/
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <stdint.h>
#include <string>
#include <iostream>
#include <cmath>
#include <conio.h>
#include <math.h>
#include <cmath>
#include<vector>
#include <assert.h>
#include <random>
#include <cstdlib>
#include <filesystem> // musi byc C++17 lub wyzej

#define PI 3.141592

using namespace std;
using namespace cv;


inline int NajbizszawielokrotnoscDw(int n)
{
	if ((n && !(n & (n - 1))) == 1)
	{
		return n;
	}
	while (n & n - 1)
	{
		n = n & n - 1;
	}
	return  n;
}

//******************************************************************************************************************
///////////////////////////////FUNKCJE OGÓLNE////////////////////////////////////////////////


inline void dct_1d_2typ(std::vector<cv::Mat>& WektorGrupy3d, int x, int y)//implemtacja użycwająca OPENCV -szybka
{
	int N = WektorGrupy3d.size();
	Mat Tymczasowa(1, N, CV_32F);

	float* wskazniklMacierzy = Tymczasowa.ptr<float>();
	for (int n = 0; n < (N); n++)

	{
		float* wskazniklWektora = WektorGrupy3d[n].ptr<float>(x);
		wskazniklMacierzy[n] = wskazniklWektora[y];
	}
	cv::dct(Tymczasowa, Tymczasowa, DCT_ROWS);
	for (int i = 0; i < (N); i++)

	{
		float* wskazniklWektora = WektorGrupy3d[i].ptr<float>(x);
		wskazniklWektora[y] = wskazniklMacierzy[i];
	}

}
inline void odwrotna_dct_1d_2typ(std::vector<cv::Mat>& WektorGrupy3d, int x, int y)
{
	int N = WektorGrupy3d.size();
	Mat Tymczasowa(1, N, CV_32F);

	float* wskazniklMacierzy = Tymczasowa.ptr<float>();
	for (int n = 0; n < (N); n++)

	{
		float* wskazniklWektora = WektorGrupy3d[n].ptr<float>(x);
		wskazniklMacierzy[n] = wskazniklWektora[y];
	}
	idct(Tymczasowa, Tymczasowa, DCT_ROWS);
	for (int i = 0; i < (N); i++)

	{
		float* wskazniklWektora = WektorGrupy3d[i].ptr<float>(x);
		wskazniklWektora[y] = wskazniklMacierzy[i];
	}
}

inline void walsh_hadamard(std::vector<cv::Mat>& WektorGrupy3d, int x, int y)
{
	if (WektorGrupy3d.size() == 0) return;
	int N = WektorGrupy3d.size();
	float pierwN = sqrt(WektorGrupy3d.size());
	int h = 1;
	Vec3f* wskaznik1;
	Vec3f* wskaznik2;
	while (h < N)
	{
		for (int i = 0; i < N; i += h * 2)
		{
			for (int j = i; j < i + h; j++)
			{
				;
				wskaznik1 = WektorGrupy3d[j].ptr<Vec3f>(x, y);
				wskaznik2 = WektorGrupy3d[j + h].ptr<Vec3f>(x, y);
				Vec3f u = *wskaznik1;
				Vec3f v = *wskaznik2;
				*wskaznik1 = u + v;
				*wskaznik2 = u - v;

			}
		}
		h *= 2;
	}

	for (int i = 0; i < N; i++)
	{
		WektorGrupy3d[i].at<Vec3f>(x, y)[0] = WektorGrupy3d[i].at<Vec3f>(x, y)[0] / (pierwN);
		WektorGrupy3d[i].at<Vec3f>(x, y)[1] = WektorGrupy3d[i].at<Vec3f>(x, y)[1] / (pierwN);
		WektorGrupy3d[i].at<Vec3f>(x, y)[2] = WektorGrupy3d[i].at<Vec3f>(x, y)[2] / (pierwN);
	}

}

void dodanie_szumu(cv::Mat obrazek_zaszumiony, int sigm, int ilosc_kanalow)
{

	double sigma = sigm; // Wartość sigma dla szumu gaussowskiego

	// Generator liczb losowych dla szumu gaussowskiego
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, sigma);
	// Dodaje szum gaussowski do każdego piksela
	for (int y = 0; y < obrazek_zaszumiony.rows; y++)
	{
		for (int x = 0; x < obrazek_zaszumiony.cols; x++)
		{
			if (ilosc_kanalow == 1)
			{
				cv::Vec<uchar, 1>& pixele = obrazek_zaszumiony.at<cv::Vec<uchar, 1>>(y, x);
				for (int c = 0; c < 1; c++)
				{
					double szum = distribution(generator);
					int new_value = cv::saturate_cast<uchar>(pixele[c] + szum);
					pixele[c] = new_value;
				}
			}
			else
			{
				cv::Vec3b& pixele = obrazek_zaszumiony.at<cv::Vec3b>(y, x);
				for (int c = 0; c < 3; c++)
				{
					double szum = distribution(generator);
					int new_value = cv::saturate_cast<uchar>(pixele[c] + szum);
					pixele[c] = new_value;
				}
			}


		}
	}
}


//*******************************************************************************************************************

///////////////////////////////FUNKCJE I KROK//////////////////////////////////////////////////////////////////////

inline void progowanie_DCT_HardTresholding_3D(std::vector<cv::Mat>& WektorGrupy3d_1krok, int wielkosc_Latki, float lambda3Dhard, Vec3f& waga, Vec3f sigma, int kanaly)
{
	waga = 0;
	int N = WektorGrupy3d_1krok.size();

	if (N == 1)
	{
		Mat kanaly_1[3];
		split(WektorGrupy3d_1krok[0], kanaly_1);

		for (int c = 0; c < kanaly; c++)
		{
			for (int i = 0; i < wielkosc_Latki; i++)
			{
				float* wskaznikWektora = kanaly_1[c].ptr<float>(i);

				for (int j = 0; j < wielkosc_Latki; j++)
				{
					if (abs(wskaznikWektora[j]) < lambda3Dhard * sigma[c])
					{
						wskaznikWektora[j] = 0;
					}
					if (wskaznikWektora[j] != 0)
					{
						waga[c] = waga[c] + 1;
					}
				}
			}
		}
		cv::merge(kanaly_1, 3, WektorGrupy3d_1krok[0]);

		for (int c = 0; c < kanaly; c++)
		{
			waga[c] = 1 / waga[c];
		}

		return;
	}
	int s = wielkosc_Latki;
	int  index = 0;
	float lambda = lambda3Dhard;
	for (int n = 0; n < N; n++)
	{
		WektorGrupy3d_1krok[n] = WektorGrupy3d_1krok[n].reshape(1, 0);
	}

	for (int c = 0; c < kanaly; c++)
	{
		for (int i = 0; i < s; i++)
		{
			for (int j = c; j < s * 3; j += 3)
			{
				Mat Tymczasowa(1, N, CV_32FC1);

				float* wskazniklMacierzy = Tymczasowa.ptr<float>();
				for (int n = 0; n < N; n++)

				{
					float* wskazniklWektora = WektorGrupy3d_1krok[n].ptr<float>(i);
					wskazniklMacierzy[n] = wskazniklWektora[j];
				}

				cv::dct(Tymczasowa, Tymczasowa, DCT_ROWS);

				for (int n = 0; n < (N); n++)
				{

					if (abs(wskazniklMacierzy[n]) < lambda * sigma[c])
					{
						wskazniklMacierzy[n] = 0;
					}
					if (wskazniklMacierzy[n] != 0)

					{
						waga[c] += 1;
					}
				}

				cv::idct(Tymczasowa, Tymczasowa, DCT_ROWS);
				for (int n = 0; n < N; n++)
					//
				{
					float* wskazniklWektora = WektorGrupy3d_1krok[n].ptr<float>(i);
					wskazniklWektora[j] = wskazniklMacierzy[n];
				}
			}
		}
	}
	for (int c = 0; c < kanaly; c++)
	{
		//waga[c] = 1 / (waga[c] * sigma[c] * sigma[c]);
		waga[c] = 1 / (waga[c] * sigma[c]);
	}

	for (int n = 0; n < N; n++)
	{
		WektorGrupy3d_1krok[n] = WektorGrupy3d_1krok[n].reshape(3, 0);
	}
	return;
}


inline void progowanie_Walsh_HardTresholding_3D(std::vector<cv::Mat>& WektorGrupy3d_1krok, int wielkosc_Latki, float lambda3Dhard, Vec3f& waga, Vec3f sigma, int kanaly)
{
	waga[0] = 0;
	waga[1] = 0;
	waga[2] = 0;
	int N = WektorGrupy3d_1krok.size();
	int s = wielkosc_Latki;
	int  index = 0;
	float lambda = lambda3Dhard;

	for (int index = 0; index < N; index++)
	{
		for (int i = 0; i < s; i++)
		{
			Vec3f* pixel = WektorGrupy3d_1krok[index].ptr<Vec3f>(i);
			for (int j = 0; j < s; j++)
			{
				for (int c = 0; c < kanaly; c++)
				{
					if (abs(pixel[j][c]) < lambda * sigma[c])
					{
						pixel[j][c] = 0;
					}
					else
					{
						waga[c]++;
					}
				}
			}
		}

	}
	for (int c = 0; c < kanaly; c++)
	{
		waga[c] = 1 / (waga[c]);
	}

}

//******************************************************************************************************************

inline float ObliczanieMAD_szumMaly(cv::Mat referencyjna, cv::Mat porownywana, int latka)
{
	float sum = 0;
	for (int i = 0; i < latka; i++)

	{
		for (int j = 0; j < latka; j++)
		{

			sum += pow(referencyjna.at<float>(i, j) - porownywana.at<float>(i, j), 2);
		}
	}
	float mad = sum / (latka * latka);
	return mad;
}


inline float ObliczanieMAD_szumMalyAlternatywa1(cv::Mat referencyjna, cv::Mat porownywana, int latka)//wolniej niż Alternatywa 3
{
	float suma = 0;
	Mat referencyjna2 = referencyjna - porownywana;
	referencyjna2 = referencyjna2.mul(referencyjna2);
	suma = cv::sum(referencyjna2)[0];
	float mad = suma / (latka * latka);
	return mad;
}

inline float ObliczanieMAD_szumMalyAlternatywa3(cv::Mat referencyjna, cv::Mat porownywana, int latka)//bardzo najszybciej
{
	float sum = 0;
	float dzielnik = latka * latka;
	for (int i = 0; i < latka; i++)

	{
		Vec3f* pixel1 = referencyjna.ptr<Vec3f>(i);
		Vec3f* pixel2 = porownywana.ptr<Vec3f>(i);
		for (int j = 0; j < latka; j++)
		{
			sum += ((pixel1[j][0] - pixel2[j][0]) * (pixel1[j][0] - pixel2[j][0]));
		}
	}
	float mad = sum / dzielnik;
	return mad;
}

//******************************************************************************************************************

inline float ObliczanieMAD_szumDuzy(cv::Mat referencyjna, cv::Mat porownywana, int latka, float progowanie_blockMatchingHard, Vec3f sigma)
{

	float sum = 0;
	float dzielnik = latka * latka;
	for (int i = 0; i < latka; i++)

	{
		Vec3f* pixel1 = referencyjna.ptr<Vec3f>(i);
		Vec3f* pixel2 = porownywana.ptr<Vec3f>(i);
		for (int j = 0; j < latka; j++)
		{
			float a = pixel1[j][0];
			float b = pixel2[j][0];
			if (abs(a) < sigma[0] * progowanie_blockMatchingHard) a = 0;
			if (abs(b) < sigma[0] * progowanie_blockMatchingHard) b = 0;
			//sum += ((pixel1[j][0] - pixel2[j][0]) * (pixel1[j][0] - pixel2[j][0]));
			sum += ((a - b) * (a - b));
		}
	}
	float mad = sum / dzielnik;
	return mad;
}

//******************************************************************************************************************

void wyszukiwacz1krok(int i, int j, int latka, int obszar_przeszukania, Mat** TablicaReferencyjna, cv::Mat MacierzWyjsciowa1krok, cv::Mat MacierzWag1krok, int SzerokoscTablicy, int WysokoscTablicy, vector<cv::Mat>& WektorGrupy3d_1krok, vector<cv::Mat>& VectorWskaznikowMaciezyWyjsc, vector < cv::Mat>& VectorWskaznikowMaciezyWag, bool szumDuzy, float progowanie_blockMatchingHard, float MaxOdlegloscSzumDuzy, float MaxOdlegloscSzumMaly, int N_wielkosc_Grupy3D1krok, int transformata, Vec3f sigmaKanaly)
{

	int pol_latki = latka / 2;
	int pol_obszaru_wyszukiwania = obszar_przeszukania / 2;
	int x1 = i + pol_latki - pol_obszaru_wyszukiwania;
	int x2 = i + pol_latki + pol_obszaru_wyszukiwania;
	int y1 = j + pol_latki - pol_obszaru_wyszukiwania;
	int y2 = j + pol_latki + pol_obszaru_wyszukiwania;
	float MAD;
	vector<float>VectorMAD;
	VectorMAD.reserve(16);
	VectorWskaznikowMaciezyWag.clear();
	VectorWskaznikowMaciezyWyjsc.clear();
	WektorGrupy3d_1krok.clear();

	vector<int>Vector_i;
	Vector_i.reserve(N_wielkosc_Grupy3D1krok);
	Vector_i.clear();
	vector<int>Vector_j;
	Vector_j.reserve(N_wielkosc_Grupy3D1krok);
	Vector_j.clear();


	if (x1 <= 0)
	{
		x1 = 0;
		x2 = obszar_przeszukania;
	}
	if (x2 > SzerokoscTablicy - 1)
	{
		x1 = SzerokoscTablicy - obszar_przeszukania;
		x2 = (SzerokoscTablicy);
	}
	if (y1 <= 0)
	{
		y1 = 0;
		y2 = obszar_przeszukania;
	}

	if (y2 > WysokoscTablicy - 1)
	{
		y1 = WysokoscTablicy - obszar_przeszukania;
		y2 = (WysokoscTablicy);
	}
	bool wstawiono = 0;
	if (szumDuzy == 0)
	{
		int ilosc = 0;
		float wartosc_ostatniego = 0;
		for (int x = x1; x < x2; x++)
		{
			for (int y = y1; y < y2; y++)
			{
				MAD = ObliczanieMAD_szumMalyAlternatywa3(TablicaReferencyjna[i][j], TablicaReferencyjna[x][y], latka);
				if (MAD < MaxOdlegloscSzumMaly)
				{
					if (ilosc == 0)
					{
						VectorMAD.push_back(MAD);
						Vector_i.push_back(x);
						Vector_j.push_back(y);
						wartosc_ostatniego = MAD;
						ilosc = 1;
					}
					else if (ilosc < N_wielkosc_Grupy3D1krok)
					{
						for (int n = 0; n < ilosc; n++)
						{
							if (MAD < VectorMAD[n])
							{
								VectorMAD.insert(VectorMAD.begin() + n, MAD);
								Vector_j.insert(Vector_j.begin() + n, y);
								Vector_i.insert(Vector_i.begin() + n, x);
								ilosc += 1;
								wartosc_ostatniego = VectorMAD[ilosc - 1];
								break;
							}

						}
					}
					else if (wartosc_ostatniego > MAD)

					{
						for (int n = 0; n < VectorMAD.size(); n++)
						{
							if (MAD < VectorMAD[n])
							{
								VectorMAD.insert(VectorMAD.begin() + n, MAD);
								Vector_j.insert(Vector_j.begin() + n, y);
								Vector_i.insert(Vector_i.begin() + n, x);
								VectorMAD.pop_back();
								Vector_j.pop_back();
								Vector_i.pop_back();
								wartosc_ostatniego = VectorMAD[ilosc - 1];
								break;
							}
						}
					}
				}
			}
		}

	}

	if (szumDuzy == 1)
	{
		int ilosc = 0;
		float wartosc_ostatniego = 0;
		for (int x = x1; x < x2; x++)
		{
			for (int y = y1; y < y2; y++)
			{
				MAD = ObliczanieMAD_szumDuzy(TablicaReferencyjna[i][j], TablicaReferencyjna[x][y], latka, progowanie_blockMatchingHard, sigmaKanaly);
				if (MAD < MaxOdlegloscSzumDuzy)
				{
					if (ilosc == 0)
					{
						VectorMAD.push_back(MAD);
						Vector_i.push_back(x);
						Vector_j.push_back(y);
						wartosc_ostatniego = MAD;
						ilosc = 1;
					}
					else if (ilosc < N_wielkosc_Grupy3D1krok)
					{
						for (int n = 0; n < ilosc; n++)
						{
							if (MAD < VectorMAD[n])
							{
								VectorMAD.insert(VectorMAD.begin() + n, MAD);
								Vector_j.insert(Vector_j.begin() + n, y);
								Vector_i.insert(Vector_i.begin() + n, x);
								ilosc += 1;
								wartosc_ostatniego = VectorMAD[ilosc - 1];
								break;
							}

						}
					}
					else if (wartosc_ostatniego > MAD)

					{
						for (int n = 0; n < VectorMAD.size(); n++)
						{
							if (MAD < VectorMAD[n])
							{
								VectorMAD.insert(VectorMAD.begin() + n, MAD);
								Vector_j.insert(Vector_j.begin() + n, y);
								Vector_i.insert(Vector_i.begin() + n, x);
								VectorMAD.pop_back();
								Vector_j.pop_back();
								Vector_i.pop_back();
								wartosc_ostatniego = VectorMAD[ilosc - 1];
								break;
							}
						}
					}
				}
			}
		}
	}


	int wielkosc_MAD = VectorMAD.size();
	if (transformata == 2)
	{
		wielkosc_MAD = NajbizszawielokrotnoscDw(wielkosc_MAD);
	}
	if (transformata == 1)
	{
		if (wielkosc_MAD % 2 != 0 && wielkosc_MAD > 1)
		{
			VectorMAD.pop_back();
			wielkosc_MAD = wielkosc_MAD - 1;
		}
	}



	for (int i = 0; i < wielkosc_MAD; i++)
	{
		WektorGrupy3d_1krok.push_back(TablicaReferencyjna[Vector_i[i]][Vector_j[i]].clone());

		VectorWskaznikowMaciezyWyjsc.push_back(cv::Mat(MacierzWyjsciowa1krok, cv::Rect(Vector_i[i], Vector_j[i], latka, latka)));
		VectorWskaznikowMaciezyWag.push_back(cv::Mat(MacierzWag1krok, cv::Rect(Vector_i[i], Vector_j[i], latka, latka)));
	}
}
//******************************************************************************************************************

inline void HardTresholding_3D(std::vector<cv::Mat>& WektorGrupy3d_1krok, int wielkosc_Latki, float lambda3Dhard, Vec3f& waga, Vec3f sigma, int transformata, int kanaly)
{
	if (transformata == 1)
	{
		progowanie_DCT_HardTresholding_3D(WektorGrupy3d_1krok, wielkosc_Latki, lambda3Dhard, waga, sigma, kanaly);
	}
	else if (transformata == 2)
	{
		int s = wielkosc_Latki;
		for (int i = 0; i < s; i++)
		{
			for (int j = 0; j < s; j++)
			{
				walsh_hadamard(WektorGrupy3d_1krok, i, j);
			}
		}
		progowanie_Walsh_HardTresholding_3D(WektorGrupy3d_1krok, wielkosc_Latki, lambda3Dhard, waga, sigma, kanaly);
		for (int i = 0; i < s; i++)
		{
			for (int j = 0; j < s; j++)
			{
				walsh_hadamard(WektorGrupy3d_1krok, i, j);
			}
		}
	}
	int v = WektorGrupy3d_1krok.size();
	int g = 0;
	while (g < v)
	{
		Mat kanaly_1[3];
		split(WektorGrupy3d_1krok[g], kanaly_1);

		for (int c = 0; c < kanaly; c++)

		{
			cv::idct(kanaly_1[c], kanaly_1[c]);
		}

		cv::merge(kanaly_1, 3, WektorGrupy3d_1krok[g]);
		g++;
	}

}
//******************************************************************************************************************

///////////////////////////////FUNKCJE II KROK//////////////////////////////////////////////////////////////////////


inline float ObliczanieMAD(cv::Mat referencyjna, cv::Mat porownywana, int latka)//bardzo najszybciej
{
	float sum = 0;
	float dzielnik = latka * latka;
	for (int i = 0; i < latka; i++)

	{
		Vec3f* pixel1 = referencyjna.ptr<Vec3f>(i);
		Vec3f* pixel2 = porownywana.ptr<Vec3f>(i);
		for (int j = 0; j < latka; j++)
		{
			sum += ((pixel1[j][0] - pixel2[j][0]) * (pixel1[j][0] - pixel2[j][0]));
		}
	}
	float mad = sum / dzielnik;
	return mad;
}

void wyszukiwacz2krok(int i, int j, int latka2krok, int obszar_przeszukania, Mat** TablicaReferencyjna2, Mat** TablicaWyjscpo1Kroku, cv::Mat MacierzWyjsciowa2krok, cv::Mat MacierzWag2krok, int SzerokoscTablicy, int WysokoscTablicy, vector<cv::Mat>& WektorGrupy3d_2krok, vector<cv::Mat>& WektorGrupy3dpo1kroku, vector<cv::Mat>& VectorWskaznikowMaciezyWyjsc2krok, vector < cv::Mat>& VectorWskaznikowMaciezyWag2krok, bool szum_Duzy, float MaxOdlegloscSzumWysoki2krok, float MaxOdlegloscSzumNiski2krok, int N_wielkosc_Grupy3D2krok, int transformata)
{

	int pol_latki = latka2krok / 2;
	int pol_obszaru_wyszukiwania = obszar_przeszukania / 2;
	int x1 = i + pol_latki - pol_obszaru_wyszukiwania;
	int x2 = i + pol_latki + pol_obszaru_wyszukiwania;
	int y1 = j + pol_latki - pol_obszaru_wyszukiwania;
	int y2 = j + pol_latki + pol_obszaru_wyszukiwania;
	float MAD;
	float MaxOdleglosc2krok;
	float wartosc_ostatniego = 0;
	vector<float>VectorMAD;
	vector<int>Vector_i;
	Vector_i.reserve(N_wielkosc_Grupy3D2krok);
	Vector_i.clear();
	vector<int>Vector_j;
	Vector_j.reserve(N_wielkosc_Grupy3D2krok);
	Vector_j.clear();
	VectorMAD.reserve(N_wielkosc_Grupy3D2krok);
	VectorWskaznikowMaciezyWag2krok.clear();
	VectorWskaznikowMaciezyWyjsc2krok.clear();
	WektorGrupy3d_2krok.clear();
	WektorGrupy3dpo1kroku.clear();


	if (szum_Duzy == true) MaxOdleglosc2krok = MaxOdlegloscSzumWysoki2krok;
	else MaxOdleglosc2krok = MaxOdlegloscSzumNiski2krok;

	if (x1 < 0)
	{
		x1 = 0;
		x2 = obszar_przeszukania;
	}
	if (x2 > SzerokoscTablicy)
	{
		x1 = SzerokoscTablicy - obszar_przeszukania;
		x2 = (SzerokoscTablicy);
	}
	if (y1 < 0)
	{
		y1 = 0;
		y2 = obszar_przeszukania;
	}

	if (y2 > WysokoscTablicy)
	{
		y1 = WysokoscTablicy - obszar_przeszukania;
		y2 = (WysokoscTablicy);
	}
	int ilosc = 0;
	for (int x = x1; x < x2; x++)
	{
		for (int y = y1; y < y2; y++)
		{
			MAD = ObliczanieMAD(TablicaWyjscpo1Kroku[i][j], TablicaWyjscpo1Kroku[x][y], latka2krok);
			if (MAD < MaxOdleglosc2krok)
			{
				if (ilosc == 0)
				{
					VectorMAD.push_back(MAD);
					Vector_i.push_back(x);
					Vector_j.push_back(y);
					wartosc_ostatniego = MAD;
					ilosc = 1;
				}
				else if (ilosc < N_wielkosc_Grupy3D2krok)
				{
					for (int n = 0; n < ilosc; n++)
					{
						if (MAD < VectorMAD[n])
						{
							VectorMAD.insert(VectorMAD.begin() + n, MAD);
							Vector_j.insert(Vector_j.begin() + n, y);
							Vector_i.insert(Vector_i.begin() + n, x);
							ilosc += 1;
							wartosc_ostatniego = VectorMAD[ilosc - 1];
							break;
						}

					}
				}
				else if (wartosc_ostatniego > MAD)

				{
					for (int n = 0; n < VectorMAD.size(); n++)
					{
						if (MAD < VectorMAD[n])
						{
							VectorMAD.insert(VectorMAD.begin() + n, MAD);
							Vector_j.insert(Vector_j.begin() + n, y);
							Vector_i.insert(Vector_i.begin() + n, x);
							VectorMAD.pop_back();
							Vector_j.pop_back();
							Vector_i.pop_back();
							wartosc_ostatniego = VectorMAD[ilosc - 1];
							break;
						}
					}
				}
			}
		}
	}
	int wielkosc_MAD = VectorMAD.size();
	if (transformata == 2)
	{
		wielkosc_MAD = NajbizszawielokrotnoscDw(wielkosc_MAD);
	}
	else if (transformata == 1)
	{
		if (wielkosc_MAD % 2 != 0 && wielkosc_MAD > 2)
		{
			VectorMAD.pop_back();
			wielkosc_MAD = wielkosc_MAD - 1;
		}
	}
	for (int i = 0; i < wielkosc_MAD; i++)
	{
		WektorGrupy3d_2krok.push_back(TablicaReferencyjna2[Vector_i[i]][Vector_j[i]].clone());
		WektorGrupy3dpo1kroku.push_back(TablicaWyjscpo1Kroku[Vector_i[i]][Vector_j[i]].clone());
		VectorWskaznikowMaciezyWyjsc2krok.push_back(cv::Mat(MacierzWyjsciowa2krok, cv::Rect(Vector_i[i], Vector_j[i], 8, 8)));
		VectorWskaznikowMaciezyWag2krok.push_back(cv::Mat(MacierzWag2krok, cv::Rect(Vector_i[i], Vector_j[i], 8, 8)));
	}
}
//******************************************************************************************************************



void FiltrowanieFiltremWienaDCT(vector<cv::Mat>& WektorGrupy3d_2krok, vector<cv::Mat>& WektorGrupy3dpo1kroku, int k_Wien, Vec3f& wagaWien, Vec3f sigma, int kanaly)
{

	int s = k_Wien;

	Vec3f sigma1;
	sigma1[0] = sigma[0] * sigma[0];
	sigma1[1] = sigma[1] * sigma[1];
	sigma1[2] = sigma[2] * sigma[2];
	Vec3f suma;
	suma[0] = 0;
	suma[1] = 0;
	suma[2] = 0;

	wagaWien[0] = 0;
	wagaWien[1] = 0;
	wagaWien[2] = 0;

	int N = WektorGrupy3d_2krok.size();
	/*
	if (N == 1)
	{
		wagaWien[0] = 1 / 10000000 * sigma[0];
		wagaWien[1] = 1 / 10000000 * sigma[1];
		wagaWien[2] = 1 / 10000000 * sigma[2];
		int g = 0;
		Mat kanaly_1[3];
		split(WektorGrupy3d_2krok[0], kanaly_1);
		idct(kanaly_1[0], kanaly_1[0]);
		idct(kanaly_1[1], kanaly_1[1]);
		idct(kanaly_1[2], kanaly_1[2]);
		cv::merge(kanaly_1, 3, WektorGrupy3d_2krok[0]);
		return;
	}
	*/
	if (N == 0) return;
	for (int n = 0; n < N; n++)
	{
		WektorGrupy3d_2krok[n] = WektorGrupy3d_2krok[n].reshape(1, 0);
		WektorGrupy3dpo1kroku[n] = WektorGrupy3dpo1kroku[n].reshape(1, 0);
	}
	float coef = 1 / N;
	for (int c = 0; c < kanaly; c++)
	{
		for (int i = 0; i < s; i++)
		{
			for (int j = c; j < s * 3; j += 3)
			{
				Mat Tymczasowa_po1kroku(1, N, CV_32FC1);
				Mat Tymczasowa_2krok(1, N, CV_32FC1);

				float* wskazniklMacierzyTymczasowa_po1kroku = Tymczasowa_po1kroku.ptr<float>();
				float* wskazniklMacierzyTymczasowa_2krok = Tymczasowa_2krok.ptr<float>();
				for (int n = 0; n < N; n++)

				{
					float* wskazniklWektoraPo1Kroku = WektorGrupy3dpo1kroku[n].ptr<float>(i);
					float* wskazniklWektora2krok = WektorGrupy3d_2krok[n].ptr<float>(i);
					wskazniklMacierzyTymczasowa_po1kroku[n] = wskazniklWektoraPo1Kroku[j];
					wskazniklMacierzyTymczasowa_2krok[n] = wskazniklWektora2krok[j];
				}
				cv::dct(Tymczasowa_po1kroku, Tymczasowa_po1kroku, DCT_ROWS);
				cv::dct(Tymczasowa_2krok, Tymczasowa_2krok, DCT_ROWS);
				for (int n = 0; n < (N); n++)
				{

					float tmp = wskazniklMacierzyTymczasowa_po1kroku[n] * wskazniklMacierzyTymczasowa_po1kroku[n];
					tmp = tmp / (tmp + sigma1[c]);
					wskazniklMacierzyTymczasowa_2krok[n] = tmp * wskazniklMacierzyTymczasowa_2krok[n];
					suma[c] += tmp;
				}
				idct(Tymczasowa_2krok, Tymczasowa_2krok, DCT_ROWS);
				for (int n = 0; n < (N); n++)

				{
					float* wskazniklWektora2krok = WektorGrupy3d_2krok[n].ptr<float>(i);
					wskazniklWektora2krok[j] = wskazniklMacierzyTymczasowa_2krok[n];
				}

			}
		}
	}
	for (int c = 0; c < kanaly; c++)
	{
		wagaWien[c] = 1 / (suma[c] * sigma[c] * sigma[c]);
	}
	for (int n = 0; n < N; n++)
	{
		WektorGrupy3d_2krok[n] = WektorGrupy3d_2krok[n].reshape(3, 0);
	}
	for (int n = 0; n < N; n++)
	{
		Mat kanaly_1[3];
		split(WektorGrupy3d_2krok[n], kanaly_1);
		idct(kanaly_1[0], kanaly_1[0]);
		idct(kanaly_1[1], kanaly_1[1]);
		idct(kanaly_1[2], kanaly_1[2]);
		cv::merge(kanaly_1, 3, WektorGrupy3d_2krok[n]);
	}
}


void FiltrowanieFiltremWienaWalsh(vector<cv::Mat>& WektorGrupy3d_2krok, vector<cv::Mat>& WektorGrupy3dpo1kroku, int k_Wien, Vec3f& wagaWien, Vec3f sigma, int kanaly)
{

	int s = k_Wien;

	for (int i = 0; i < s; i++)
	{
		for (int j = 0; j < s; j++)

		{
			walsh_hadamard(WektorGrupy3d_2krok, i, j);
			walsh_hadamard(WektorGrupy3dpo1kroku, i, j);
		}
	}
	int N = WektorGrupy3dpo1kroku.size();


	Vec3f sigma1;
	sigma1[0] = sigma[0] * sigma[0];
	sigma1[1] = sigma[1] * sigma[1];
	sigma1[2] = sigma[2] * sigma[2];
	Vec3f suma;
	suma[0] = 0;
	suma[1] = 0;
	suma[2] = 0;

	wagaWien[0] = 0;
	wagaWien[1] = 0;
	wagaWien[2] = 0;

	for (int i = 0; i < N; i++)
	{
		Mat kanaly_po1kroku[3];
		Mat kanaly_2krok[3];
		split(WektorGrupy3dpo1kroku[i], kanaly_po1kroku);
		split(WektorGrupy3d_2krok[i], kanaly_2krok);
		for (int c = 0; c < kanaly; c++)
		{
			Mat tmp = kanaly_po1kroku[c].mul(kanaly_po1kroku[c]);
			tmp = tmp / (tmp + sigma1[c]);
			kanaly_2krok[c] = tmp.mul(kanaly_2krok[c]);
			suma[c] += cv::sum(tmp)[0];
		}
		cv::merge(kanaly_po1kroku, 3, WektorGrupy3dpo1kroku[i]);
		cv::merge(kanaly_2krok, 3, WektorGrupy3d_2krok[i]);
	}


	for (int c = 0; c < kanaly; c++)

	{
		wagaWien[c] = 1 / sigma1[c] * suma[c];
	}


	for (int i = 0; i < s; i++)
	{
		for (int j = 0; j < s; j++)

		{
			walsh_hadamard(WektorGrupy3d_2krok, i, j);
		}
	}
	for (int i = 0; i < N; i++)
	{
		Mat kanaly_2krok[3];
		split(WektorGrupy3d_2krok[i], kanaly_2krok);
		idct(kanaly_2krok[0], kanaly_2krok[0]);
		idct(kanaly_2krok[1], kanaly_2krok[1]);
		idct(kanaly_2krok[2], kanaly_2krok[2]);
		merge(kanaly_2krok, 3, WektorGrupy3d_2krok[i]);
	}
}
void funkcja_glowna(cv::Mat Obrazek, cv::Mat& MacierzWyjsciowa1krok, cv::Mat& MacierzWyjsciowa2krok, int sigma, int tryb_szybkosci, int transformata, cv::Mat Macierz_Kaisera)
{
	int p_Hard = 3; //p_Hard krok tworzenia latek, w oryginale 1,2 lub 3, u Lebruna 3
	int p_Wien = 3; //krok tworzenia latek, w oryginale 1, 2 lub 3, u Lebruna 3
	float LambdaHard2d = 2.0; //Lambda_hard2d progowanie(trasholding) przy block matchingu, u Lebruna 2.0
	float LambdaHard3d = 2.7; //LambdaHard3d	progowanie(trasholding) Grupy3d w pierwszym kroku filtra, u Lebruna 2,7
	int k_Hard = 8; //k_Hard Wielkosc Latki w 1 kroku, dla DTC 8
	int k_Wien = 8;// Wielkosc Latki w 1 kroku, dla DTC 8
	int N_Hard = 16;//N_Hard Maksymalna wielkość grupy 3d w 1kroku, dla DCT musi być parzysta, u Lebruna 16
	int N_Wien = 32; // Maksymalna wielkość grupy 3d w 2kroku, dla DCT musi być parzysta u Lebruna 32
	int n_Hard = 39;//n_Hard Wielkosc okna szukania podobych latek w pierwszym kroku
	int n_Wien = 39; // Wielkosc okna szukania podobych latek w drugim kroku
	float tau_Hard_niski = 2500;//tau_Hard_niski -maksymalna odleglosc latek przy szumie małym (ponizej 40) - u Lebruna 2500
	float tau_Hard_wysoki = 10000.0;//tau_Hard_wysoki -maksymalna odleglosc latek przy szumie duzym (powyzej (40) - u Lebruna 5000
	float tau_Wien_niski = 400;// -maksymalna odleglosc latek przy szumie małym(ponizej 40) - u Lebruna 400
	float tau_Wien_wysoki = 1500;// -maksymalna odleglosc latek przy szumie duzym(powyzej(40) - u Lebruna 800
	std::string nazwa_pilku_zaszumionego;
	//std::string nazwa_pilku_referencyjnego;
	int opcja_obrazka = 1;
	//dodanie_szumu(Obrazek, sigma, 3);
	if (tryb_szybkosci == 1)
	{
		p_Hard = 4;
		p_Wien = 4;
	}
	Vec3f waga = 0;
	cv::Mat ObrazekKolorowy;
	cv::cvtColor(Obrazek, ObrazekKolorowy, COLOR_BGR2YCrCb);
	//cv::cvtColor(Obrazek, ObrazekKolorowy, COLOR_BGR2YUV);
	ObrazekKolorowy.convertTo(ObrazekKolorowy, CV_32FC3);

	int width = Obrazek.cols;
	int height = Obrazek.rows;
	int w = width / p_Hard;
	int h = height / p_Hard;
	int kanaly = 3;
	Vec3f sigmaKanaly;

	sigmaKanaly[0] = sqrtf(0.299 * 0.299 + 0.587 * 0.587 + 0.114 * 0.114) * sigma;
	sigmaKanaly[1] = sqrtf(0.169 * 0.169 + 0.331 * 0.331 + 0.500 * 0.500) * sigma;
	sigmaKanaly[2] = sqrtf(0.500 * 0.500 + 0.419 * 0.419 + 0.081 * 0.081) * sigma;

	MacierzWyjsciowa1krok = cv::Mat::zeros(height, width, CV_32FC3);
	vector<cv::Mat>VectorWskaznikowMacierzyWyjsc;
	VectorWskaznikowMacierzyWyjsc.reserve(16);
	cv::Mat MacierzWag1krok = cv::Mat::zeros(height, width, CV_32FC3);
	vector<cv::Mat>VectorWskaznikowMacierzyWag;
	VectorWskaznikowMacierzyWag.reserve(16);
	std::vector<Mat> WektorGrupy3d_1krok;
	WektorGrupy3d_1krok.reserve(16);
	bool szumDuzy = false;
	if (sigmaKanaly[0] > 40) szumDuzy = true;
	//Najpierw tworzymy tablice zawierającą łatki pokrywające obrazek i od razu łatki te poddajemy DCT2D//
	int szerokosc_Tablicy = Obrazek.cols - k_Hard + 1;
	int wysokosc_Tablicy = Obrazek.rows - k_Hard + 1;
	cv::Mat** tablica_referencyjna1 = new Mat * [szerokosc_Tablicy]; //alokacja pamieci
	for (int i = 0; i < szerokosc_Tablicy; ++i)
	{
		tablica_referencyjna1[i] = new Mat[wysokosc_Tablicy]; //alokacja pamieci
	}

	cv::Mat** tablica_referencyjna2 = new Mat * [szerokosc_Tablicy]; //alokacja pamieci
	for (int i = 0; i < szerokosc_Tablicy; ++i)
	{
		tablica_referencyjna2[i] = new Mat[wysokosc_Tablicy]; //alokacja pamieci
	}

	for (int i = 0; i < szerokosc_Tablicy; i++)
	{
		for (int j = 0; j < wysokosc_Tablicy; j++)

		{
			tablica_referencyjna1[i][j] = cv::Mat(ObrazekKolorowy(Rect(i, j, k_Hard, k_Hard)).clone()); // tak powinno być, żeby stworzyć głęboką kopię
			Mat kanaly_1[3];
			split(tablica_referencyjna1[i][j], kanaly_1);

			cv::dct(kanaly_1[0], kanaly_1[0]);
			cv::dct(kanaly_1[1], kanaly_1[1]);
			cv::dct(kanaly_1[2], kanaly_1[2]);
			cv::merge(kanaly_1, 3, tablica_referencyjna1[i][j]);

			tablica_referencyjna2[i][j] = cv::Mat(tablica_referencyjna1[i][j].clone());
		}
	}
	for (int i = 0; i < szerokosc_Tablicy + p_Hard; i += p_Hard)
	{
		for (int j = 0; j < wysokosc_Tablicy + p_Hard; j += p_Hard)

		{
			int x = i;;
			int y = j;
			if (i >= szerokosc_Tablicy) x = szerokosc_Tablicy - 1;

			if (j >= wysokosc_Tablicy) y = wysokosc_Tablicy - 1;

			wyszukiwacz1krok(x, y, k_Hard, n_Hard, tablica_referencyjna1, MacierzWyjsciowa1krok, MacierzWag1krok, szerokosc_Tablicy, wysokosc_Tablicy, WektorGrupy3d_1krok, VectorWskaznikowMacierzyWyjsc, VectorWskaznikowMacierzyWag, szumDuzy, LambdaHard2d, tau_Hard_wysoki, tau_Hard_niski, N_Hard, transformata, sigmaKanaly);

			HardTresholding_3D(WektorGrupy3d_1krok, k_Hard, LambdaHard3d, waga, sigmaKanaly, transformata, kanaly);

			int index = 0;
			int index_max = WektorGrupy3d_1krok.size();
			while (index < index_max)
			{

				Scalar wartosci_wag(waga[0], waga[1], waga[2]);
				Mat tmp;
				multiply(Macierz_Kaisera, wartosci_wag, tmp);
				VectorWskaznikowMacierzyWyjsc[index] = (VectorWskaznikowMacierzyWyjsc[index]) + (WektorGrupy3d_1krok[index].mul(tmp));
				VectorWskaznikowMacierzyWag[index] = VectorWskaznikowMacierzyWag[index] + (tmp);
				index++;
			}

		}
	}
	MacierzWyjsciowa1krok = MacierzWyjsciowa1krok / MacierzWag1krok;

	for (int i = 0; i < szerokosc_Tablicy; ++i)
	{
		delete[] tablica_referencyjna1[i]; //uwolnienie pamieci
	}

	delete[]tablica_referencyjna1; //uwolnienie pamieci
	tablica_referencyjna1 = NULL;
	///////////////////////////////////////////////////////////////////////Drugi KROK/////////////////////////////////////////////

	MacierzWyjsciowa2krok = cv::Mat::zeros(height, width, CV_32FC3);
	vector<cv::Mat>VectorWskaznikowMaciezyWyjsc2krok;
	VectorWskaznikowMaciezyWyjsc2krok.reserve(N_Wien);
	cv::Mat MacierzWag2krok = cv::Mat::zeros(height, width, CV_32FC3);
	vector<cv::Mat>VectorWskaznikowMaciezyWag2krok;
	VectorWskaznikowMacierzyWag.reserve(N_Wien);
	std::vector<Mat> WektorGrupy3d_2krok;
	WektorGrupy3d_2krok.reserve(N_Wien);
	std::vector<Mat> WektorGrupy3dpo1kroku;
	WektorGrupy3dpo1kroku.reserve(N_Wien);

	Vec3f wagaWien = 0;
	szerokosc_Tablicy = Obrazek.cols - k_Wien + 1;
	wysokosc_Tablicy = Obrazek.rows - k_Wien + 1;
	cv::Mat** TablicaWyjscpo1Kroku = new Mat * [szerokosc_Tablicy]; //alokacja pamieci
	for (int i = 0; i < szerokosc_Tablicy; ++i)
	{
		TablicaWyjscpo1Kroku[i] = new Mat[wysokosc_Tablicy]; //alokacja pamieci
	}

	for (int i = 0; i < szerokosc_Tablicy; i++)
	{
		for (int j = 0; j < wysokosc_Tablicy; j++)

		{
			TablicaWyjscpo1Kroku[i][j] = cv::Mat(MacierzWyjsciowa1krok(Rect(i, j, k_Wien, k_Wien)).clone()); // tak powinno być, żeby stworzyć głęboką kopię
			Mat kanaly_1[3];
			split(TablicaWyjscpo1Kroku[i][j], kanaly_1);
			cv::dct(kanaly_1[0], kanaly_1[0]);
			cv::dct(kanaly_1[1], kanaly_1[1]);
			cv::dct(kanaly_1[2], kanaly_1[2]);
			cv::merge(kanaly_1, 3, TablicaWyjscpo1Kroku[i][j]);
		}
	}
	for (int i = 0; i < szerokosc_Tablicy + p_Wien; i += p_Wien)
	{
		for (int j = 0; j < wysokosc_Tablicy + p_Wien; j += p_Wien)

		{
			int x = i;;
			int y = j;
			if (i >= szerokosc_Tablicy) x = szerokosc_Tablicy - 1;

			if (j >= wysokosc_Tablicy) y = wysokosc_Tablicy - 1;

			wyszukiwacz2krok(x, y, k_Wien, n_Wien, tablica_referencyjna2, TablicaWyjscpo1Kroku, MacierzWyjsciowa2krok, MacierzWag2krok, szerokosc_Tablicy, wysokosc_Tablicy, WektorGrupy3d_2krok, WektorGrupy3dpo1kroku, VectorWskaznikowMaciezyWyjsc2krok, VectorWskaznikowMaciezyWag2krok, szumDuzy, tau_Wien_wysoki, tau_Wien_niski, N_Wien, transformata);

			if (transformata == 1)
			{
				FiltrowanieFiltremWienaDCT(WektorGrupy3d_2krok, WektorGrupy3dpo1kroku, k_Wien, wagaWien, sigmaKanaly, kanaly);
			}
			else if (transformata == 2)
			{
				FiltrowanieFiltremWienaWalsh(WektorGrupy3d_2krok, WektorGrupy3dpo1kroku, k_Wien, wagaWien, sigmaKanaly, kanaly);
			}

			int index = 0;
			int index_max = WektorGrupy3d_2krok.size();
			while (index < index_max)

			{
				Scalar wartosci_wag(wagaWien[0], wagaWien[1], wagaWien[2]);
				Mat tmp;
				multiply(Macierz_Kaisera, wartosci_wag, tmp);
				VectorWskaznikowMaciezyWyjsc2krok[index] = (VectorWskaznikowMaciezyWyjsc2krok[index]) + (WektorGrupy3d_2krok[index].mul(tmp));
				VectorWskaznikowMaciezyWag2krok[index] = VectorWskaznikowMaciezyWag2krok[index] + tmp;
				index++;
			}

		}
	}
	MacierzWyjsciowa2krok = MacierzWyjsciowa2krok / MacierzWag2krok;
	MacierzWyjsciowa1krok.convertTo(MacierzWyjsciowa1krok, CV_8UC3);
	MacierzWyjsciowa2krok.convertTo(MacierzWyjsciowa2krok, CV_8UC3);
	cv::cvtColor(MacierzWyjsciowa1krok, MacierzWyjsciowa1krok, COLOR_YCrCb2BGR);
	cv::cvtColor(MacierzWyjsciowa2krok, MacierzWyjsciowa2krok, COLOR_YCrCb2BGR);
	imshow("Obrazek po 1 kroku", MacierzWyjsciowa1krok);
	for (int i = 0; i < szerokosc_Tablicy; ++i)
	{
		delete[] tablica_referencyjna2[i];
		delete[] TablicaWyjscpo1Kroku[i];
	}

	delete[] tablica_referencyjna2;
	delete[] TablicaWyjscpo1Kroku;
	tablica_referencyjna2 = NULL;
	TablicaWyjscpo1Kroku = NULL;
}

int main(int argc, char* argv[])
{
	int sigma; // sigma - poziom szumu
	cv::Mat Obrazek;
	//cv::Mat ObrazekReferencyjny;
	std::string nazwa_pilku_zaszumionego;
	//std::string nazwa_pilku_referencyjnego;
	cv::Mat MacierzWyjsciowa1krok;
	cv::Mat MacierzWyjsciowa2krok;
	std::string wpisana_nazwa;
	int transformata = 2;
	int tryb_szybkosci = 0;
	int licznik = 0;


	if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
	{
		std::cout << "Filtr CBM3D CPU, filtruje z szumu obrazy kolorowe.\n";
		std::cout << "Uzycie: BM3D_CPU.exe <nazwa pliku> <liczba całkowita> <liczba zmiennoprzecinkowa> <bool>\n";
		std::cout << "Argumenty:\n";
		std::cout << "  <nazwa pliku>       Nazwa pliku (string). Mozna podac nazwe i sciezke folderu lub sama nazwe \n"; 
		std::cout << "						jezeli znajduje sie w jednym folderze z programem - zostanąa przetworzone \n";
		std::cout <<"						wszystkie pliki graficzne w folderze\n";
		std::cout << "  <poziom szumu>      Liczba calkowita (int)\n";
		std::cout << "  <tryb szybkosci>	0 - normalny, 1 - szybki\n";
		return 0;
	}

	if (argc != 4) 
	{
		std::cerr << "Użycie: " << argv[0] << " <nazwa pliku> <poziom szumu> <normalny czy szybki?>\n pomoc: --help lub -h";
		cv::waitKey(0);
		return 1;
	}

	wpisana_nazwa = argv[1];       //nazwa wczytywanego pliku argv[0] to nazwa programu
	sigma = std::atoi(argv[2]);  // drugi arg - poziom szumu
	tryb_szybkosci = std::atoi(argv[3]); // Argument float


	float Tablica_Kaisera[8][8] = { 0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924,
									0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
									0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
									0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
									0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
									0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
									0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
									0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924 };
	cv::Mat Macierz_Kaisera1 = cv::Mat(8, 8, CV_32FC1, (Tablica_Kaisera));
	cv::Mat Macierz_Kaisera = cv::Mat(8, 8, CV_32FC3);
	cv::Mat planes[3];
	planes[0] = Macierz_Kaisera1;
	planes[1] = Macierz_Kaisera1;
	planes[2] = Macierz_Kaisera1;
	cv::merge(planes, 3, Macierz_Kaisera);

	
	time_t czasStart = time(NULL);
	if (std::filesystem::is_regular_file(wpisana_nazwa))
	{
		Obrazek = cv::imread(wpisana_nazwa, cv::IMREAD_COLOR);

		if (Obrazek.empty())
		{
			std::cerr << "Nie mozna wczytac obrazka do odszumienia." << std::endl;
			cv::waitKey(0);
			return -1;
		}
		funkcja_glowna(Obrazek, MacierzWyjsciowa1krok, MacierzWyjsciowa2krok, sigma, tryb_szybkosci, transformata, Macierz_Kaisera);
		for (std::filesystem::path plik : {std::filesystem::absolute(wpisana_nazwa)})
		{
			std::string nowa_nazwa = plik.stem().string() + "_filtered" + plik.extension().string();
			std::string nowa_nazwa_i_sciezka = plik.parent_path().string() + "/" + nowa_nazwa;
			std::cout << plik.parent_path().string() << endl;
			std::cout << nowa_nazwa_i_sciezka << endl;
			// Zapis przetworzonego obrazu
			imshow("Obrazek po 2 kroku", MacierzWyjsciowa2krok);
			cv::imwrite(nowa_nazwa_i_sciezka, MacierzWyjsciowa2krok);
		}
		licznik++;
	}
	else if (std::filesystem::is_directory(wpisana_nazwa)) 
	{
		for (const auto& entry : std::filesystem::directory_iterator(wpisana_nazwa)) 
		{
			// Sprawdzenie, czy plik ma odpowiednie rozszerzenie
			if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg" || entry.path().extension() == ".png"
				|| entry.path().extension() == ".bmp" || entry.path().extension() == ".tiff" || entry.path().extension() == ".tif" || entry.path().extension() == ".webp"
				|| entry.path().extension() == ".hdr" || entry.path().extension() == ".jp2"))
			{
				Obrazek = cv::imread(entry.path().string(), cv::IMREAD_COLOR);

				if (Obrazek.empty())
				{
					std::cerr << "Nie mozna wczytac obrazka do odszumienia." << std::endl;
					cv::waitKey(0);
					return -1;
				}
				funkcja_glowna(Obrazek, MacierzWyjsciowa1krok, MacierzWyjsciowa2krok, sigma, tryb_szybkosci, transformata, Macierz_Kaisera);
				for (std::filesystem::path plik : {std::filesystem::absolute(std::filesystem::path(entry))})
				{
					std::string nowa_nazwa = plik.stem().string() + "_filtered" + plik.extension().string();
					std::string nowa_sciezka = plik.parent_path().string() + "/" + "filtered";
					std::string nowa_nazwa_i_sciezka = nowa_sciezka + "/" + nowa_nazwa;
					if (!std::filesystem::exists(nowa_sciezka)) //sprawdza czy istnieje folder do apisania wynikowych obrazow
					{
						if (std::filesystem::create_directories(nowa_sciezka)) {
							std::cout << "Utworzono folder: " << nowa_sciezka << std::endl;
						}
						else {
							std::cerr << "Nie udało się utworzyć folderu: " << nowa_sciezka << std::endl;
							return 1; // Zakończenie programu z błędem
						}
					}
					cv::imwrite(nowa_nazwa_i_sciezka, MacierzWyjsciowa2krok);
					std::cerr << "przefiltrowano i zapisano:" << nowa_nazwa << std::endl;
					licznik++;
				}
			}
		}
	}
	else 
	{
		std::cerr << "Podana ścieżka lub nazwa pliku  jest bledna" << std::endl;
		return 1;
	}

	time_t czasStop = time(NULL);
	std::cout << "Przefiltrowano " << licznik << " obrazow w czasie : " << czasStop - czasStart << " s." << std::endl;

	cv::waitKey(0);

	return 0;
}
