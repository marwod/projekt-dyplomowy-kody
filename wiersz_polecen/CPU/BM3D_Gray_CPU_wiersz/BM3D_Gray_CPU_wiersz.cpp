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


//******************************************************************************************************************
///////////////////////////////FUNKCJE OGÓLNE////////////////////////////////////////////////
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

inline void Walsh_Hadamard(std::vector<cv::Mat>& WektorGrupy3d, int x, int y)
{
	if (WektorGrupy3d.size() == 0) return;
	int N = WektorGrupy3d.size();
	float pierwN = sqrt(WektorGrupy3d.size());
	int h = 1;
	float* wskaznik1;
	float* wskaznik2;
	while (h < N)
	{
		for (int i = 0; i < N; i += h * 2)
		{			
			for (int j = i; j < i + h; j++)
			{
				wskaznik1 = WektorGrupy3d[j].ptr<float>(x, y);
				wskaznik2 = WektorGrupy3d[j+h].ptr<float>(x, y);
				float u = *wskaznik1;
				float v = *wskaznik2;
				*wskaznik1 = u + v;
				*wskaznik2 = u - v;
			}
		}
		h *= 2;
	}

	for (int i = 0; i < N; i++)
	{
		WektorGrupy3d[i].at<float>(x, y) = WektorGrupy3d[i].at<float>(x, y) / (pierwN);
	}

}



///////////////////////////////FUNKCJE I KROK//////////////////////////////////////////////////////////////////////

inline void progowanie_DCT_HardTresholding_3D(std::vector<cv::Mat> &WektorGrupy3d_1krok, int wielkosc_Latki, float lambda3Dhard, float& waga_ht, int sigma)
{
	waga_ht= 0;
	int N = WektorGrupy3d_1krok.size();
	if (N == 1)
	{
		for (int i = 0; i < wielkosc_Latki; i++)
		{
			float* wskaznikWektora = WektorGrupy3d_1krok[0].ptr<float>(i);
			for (int j = 0; j < wielkosc_Latki; j++)
			{
				if (abs(wskaznikWektora[j]) < lambda3Dhard * sigma)
				{
					wskaznikWektora[j] = 0;
				}
				if (wskaznikWektora[j] != 0)
				{
					waga_ht++;
				}
			}
		}
		return;
	}
	else
	{
		for (int i = 0; i < wielkosc_Latki; i++)
		{
			for (int j = 0; j < wielkosc_Latki; j++)
			{
				Mat Tymczasowa(1, N, CV_32F);
				float* wskazniklMacierzy = Tymczasowa.ptr<float>();
				for (int n = 0; n < N; n++)
				{
					float* wskazniklWektora = WektorGrupy3d_1krok[n].ptr<float>(i);
					wskazniklMacierzy[n] = wskazniklWektora[j];
				}
				cv::dct(Tymczasowa, Tymczasowa, DCT_ROWS);
				for (int n = 0; n < (N); n++)
				{
					if (abs(wskazniklMacierzy[n]) < lambda3Dhard * sigma)
					{
						wskazniklMacierzy[n] = 0;
					}
					if (wskazniklMacierzy[n] != 0)
					{
						waga_ht++;
					}
				}
				idct(Tymczasowa, Tymczasowa, DCT_ROWS);
				for (int n = 0; n < (N); n++)

				{
					float* wskazniklWektora = WektorGrupy3d_1krok[n].ptr<float>(i);
					wskazniklWektora[j] = wskazniklMacierzy[n];
				}
			}
		}
	}
}

inline void progowanie_Walsh_HardTresholding_3D(std::vector<cv::Mat>& WektorGrupy3d_1krok, int wielkosc_Latki, float lambda3Dhard, float& waga_ht, int sigma)
{
	waga_ht= 0.0f;
	int N = WektorGrupy3d_1krok.size();
	int s = wielkosc_Latki;
	int  index = 0;
	
	for (int i = 0; i < wielkosc_Latki; i++)
	{
		for (int j = 0; j < wielkosc_Latki; j++)
		{
			Walsh_Hadamard(WektorGrupy3d_1krok, i, j);
		}
	}
	
	while (index < N)
	{
		for (int i = 0; i < s; i++)
		{
			float* pixel = WektorGrupy3d_1krok[index].ptr<float>(i);
			for (int j = 0; j < s; j++)
			{
				if ((abs(pixel[j])) < lambda3Dhard * sigma)
				{
					pixel[j] = 0;
				}
				else
				{
					waga_ht++;
				}
			}
		}

		index++;
	}
	
	for (int i = 0; i < wielkosc_Latki; i++)
	{
		for (int j = 0; j < wielkosc_Latki; j++)
		{
			Walsh_Hadamard(WektorGrupy3d_1krok,i, j);
		}
	}
	
}

//******************************************************************************************************************



inline float ObliczanieMSE_szumMaly(cv::Mat referencyjna, cv::Mat porownywana, int latka)
{
	float suma = 0.0f;
	float dzielnik = latka * latka;
	for (int i = 0; i < latka; i++)
	{
		float* pixel1 = referencyjna.ptr<float>(i);
		float* pixel2 = porownywana.ptr<float>(i);
		for (int j = 0; j < latka; j++)
		{
			suma += (pixel1[j] - pixel2[j]) * (pixel1[j] - pixel2[j]);
		}
	}
	float MSE = suma / (dzielnik);	
	return MSE;
}

//******************************************************************************************************************

inline float ObliczanieMSE_szumDuzy(cv::Mat referencyjna, cv::Mat porownywana, int latka, float progowanie_blockMatchingHard, int sigma)
{

	float suma = 0.0f;
	float dzielnik = latka*latka;
	float wspolczynnik_progowania = (progowanie_blockMatchingHard * sigma) * (progowanie_blockMatchingHard * sigma);
	for (int i = 0; i < latka; i++)

	{
		float* pixel1 = referencyjna.ptr<float>(i);
		float* pixel2 = porownywana.ptr<float>(i);
		for (int j = 0; j < latka; j++)
		{
			float p1 = pixel1[j];
			float p2 = pixel2[j];
			
			if ((p1*p1) < wspolczynnik_progowania)
			{
				p1 = 0;
			}
			if ((p2* p2) < wspolczynnik_progowania)
			{
				p2 = 0;
			}
			

			suma += (p1 - p2) * (p1 - p2);
		}
	}
	float MSE = suma / dzielnik;
	return MSE;
}

//D******************************************************************************************************************

void wyszukiwacz1krok(int i, int j, int wielkosc_latki, int obszar_przeszukania, Mat **TablicaLatek, cv::Mat &MacierzWyjsciowa1krok, cv::Mat &MacierzWag1krok, int SzerokoscTablicy, int WysokoscTablicy, vector<cv::Mat>& WektorGrupy3d_1krok, vector<cv::Mat>& WektorWskaznikowMacierzyWyjsc, 
	vector <cv::Mat>& WektorWskaznikowMacierzyWag, bool szumDuzy, 
	float progowanie_blockMatchingHard, float MaxOdlegloscSzumDuzy, 
	float MaxOdlegloscSzumMaly, int N_wielkosc_Grupy3D1krok, int transformata, int sigma)	
{
	int pol_latki = wielkosc_latki / 2;
	int pol_obszaru_wyszukiwania = obszar_przeszukania / 2;
	int x1 = i + pol_latki - pol_obszaru_wyszukiwania;
	int x2 = i + pol_latki + pol_obszaru_wyszukiwania;
	int y1 = j + pol_latki - pol_obszaru_wyszukiwania;
	int y2 = j + pol_latki + pol_obszaru_wyszukiwania;
	float MaxOdleglosc;
	float MSE;
	int ilosc = 0;
	float wartosc_ostatniego = 0;

	vector<float>VectorMSE;
	VectorMSE.reserve(16);
	vector<int>Wektor_kordynat_x;
	Wektor_kordynat_x.reserve(N_wielkosc_Grupy3D1krok);
	vector<int>Wektor_kordynat_y;
	Wektor_kordynat_y.reserve(N_wielkosc_Grupy3D1krok);

	WektorWskaznikowMacierzyWag.clear();
	WektorWskaznikowMacierzyWyjsc.clear();
	WektorGrupy3d_1krok.clear();
	Wektor_kordynat_x.clear();
	Wektor_kordynat_y.clear();
	VectorMSE.clear();

	if (szumDuzy == 0)
	{
		MaxOdleglosc = MaxOdlegloscSzumMaly;
	}
	else
	{
		MaxOdleglosc = MaxOdlegloscSzumDuzy;
	}
	if (x1 <= 0)
	{
		x1 = 0;
		x2 = obszar_przeszukania;
	}
	if (x2 > SzerokoscTablicy)
	{
		x1 = SzerokoscTablicy - obszar_przeszukania;
		x2 = (SzerokoscTablicy);
	}
	if (y1 <= 0)
	{
		y1 = 0;
		y2 = obszar_przeszukania;
	}

	if (y2 > WysokoscTablicy)
	{
		y1 = WysokoscTablicy - obszar_przeszukania;
		y2 = (WysokoscTablicy);
	}
		for (int x = x1; x < x2; x++)
		{
			for (int y = y1; y < y2; y++)
			{
				if (szumDuzy == 0)
				{
					MSE = ObliczanieMSE_szumMaly(TablicaLatek[i][j], TablicaLatek[x][y], wielkosc_latki);
				}
				else
				{
					MSE = ObliczanieMSE_szumDuzy(TablicaLatek[i][j], TablicaLatek[x][y], wielkosc_latki, progowanie_blockMatchingHard, sigma);
				}
				if (MSE < MaxOdleglosc)
				{
					if (ilosc == 0)
					{
						VectorMSE.push_back(MSE);
						Wektor_kordynat_x.push_back(x);
						Wektor_kordynat_y.push_back(y);
						wartosc_ostatniego = MSE;
						ilosc = 1;
					}
					else if (ilosc < N_wielkosc_Grupy3D1krok)
					{
						if(MSE<wartosc_ostatniego)
						{
							for (int n = 0; n < ilosc; n++)
							{
								if (MSE < VectorMSE[n])
								{
									VectorMSE.insert(VectorMSE.begin() + n, MSE);
									Wektor_kordynat_y.insert(Wektor_kordynat_y.begin() + n, y);
									Wektor_kordynat_x.insert(Wektor_kordynat_x.begin() + n, x);
									wartosc_ostatniego = VectorMSE[ilosc];
									ilosc += 1;
									break;
								}

							}
						}
						else
						{
							VectorMSE.push_back(MSE);
							Wektor_kordynat_x.push_back(x);
							Wektor_kordynat_y.push_back(y);
							wartosc_ostatniego = MSE;
							ilosc += 1;
						}
					}
					else if (wartosc_ostatniego > MSE)
					{
						for (int n = 0; n < VectorMSE.size(); n++)
						{
							if (MSE < VectorMSE[n])
							{
								VectorMSE.insert(VectorMSE.begin() + n, MSE);
								Wektor_kordynat_y.insert(Wektor_kordynat_y.begin() + n, y);
								Wektor_kordynat_x.insert(Wektor_kordynat_x.begin() + n, x);
								VectorMSE.pop_back();
								Wektor_kordynat_y.pop_back();
								Wektor_kordynat_x.pop_back();	
								wartosc_ostatniego = VectorMSE[ilosc];								
								break;
							}
						}
					}
				}
			}
		}

	int wielkosc_grupy = VectorMSE.size();
	if (transformata == 1)
	{
		if (wielkosc_grupy % 2 != 0)
		{
			VectorMSE.pop_back();
			wielkosc_grupy = wielkosc_grupy - 1;
		}
	}
	if (transformata == 2)
	{
		wielkosc_grupy = NajbizszawielokrotnoscDw(wielkosc_grupy);
	}



	for (int i = 0; i < wielkosc_grupy; i++)
	{
		WektorGrupy3d_1krok.push_back(TablicaLatek[Wektor_kordynat_x[i]][Wektor_kordynat_y[i]].clone());
		WektorWskaznikowMacierzyWyjsc.push_back(cv::Mat(MacierzWyjsciowa1krok, cv::Rect(Wektor_kordynat_x[i], Wektor_kordynat_y[i], 8, 8)));
		WektorWskaznikowMacierzyWag.push_back(cv::Mat(MacierzWag1krok, 
			cv::Rect(Wektor_kordynat_x[i], Wektor_kordynat_y[i], 8, 8)));
	}
}
//******************************************************************************************************************

inline void HardTresholding_3D(std::vector<cv::Mat>& WektorGrupy3d_1krok, int wielkosc_Latki, float lambda3Dhard, float& waga, int sigma, 
	int transformata)
{
	if (transformata == 1)
	{
		progowanie_DCT_HardTresholding_3D(WektorGrupy3d_1krok, wielkosc_Latki, lambda3Dhard, waga, sigma);
	}
	else if (transformata == 2)
	{
		progowanie_Walsh_HardTresholding_3D(WektorGrupy3d_1krok, wielkosc_Latki, lambda3Dhard, waga, sigma);
	}
	int v = WektorGrupy3d_1krok.size();
	int g = 0;
	while (g < v)
	{
		idct(WektorGrupy3d_1krok[g], WektorGrupy3d_1krok[g]);
		g++;
	}

}
//******************************************************************************************************************

///////////////////////////////FUNKCJE II KROK//////////////////////////////////////////////////////////////////////


void wyszukiwacz2krok(int i, int j, int latka2krok, int obszar_przeszukania, Mat** TablicaReferencyjna2, Mat** tablica_latek_po_1_kroku, cv::Mat MacierzWyjsciowa2krok, cv::Mat MacierzWag2krok, int SzerokoscTablicy, int WysokoscTablicy, 
	vector<cv::Mat>& WektorGrupy3d_2krok, vector<cv::Mat>& WektorGrupy3dpo1kroku, vector<cv::Mat>& WektorWskaznikowMacierzyWyjsc2krok, 
	  vector < cv::Mat>& WektorWskaznikowMacierzyWag2krok, bool szum_Duzy, 
	  float MaxOdlegloscSzumWysoki2krok, float MaxOdlegloscSzumNiski2krok, 
	  int N_wielkosc_Grupy3D2krok, int transformata)
{

	int pol_latki = latka2krok / 2;
	int pol_obszaru_wyszukiwania = obszar_przeszukania / 2;
	int x1 = i + pol_latki - pol_obszaru_wyszukiwania;
	int x2 = i + pol_latki + pol_obszaru_wyszukiwania;
	int y1 = j + pol_latki - pol_obszaru_wyszukiwania;
	int y2 = j + pol_latki + pol_obszaru_wyszukiwania;
	float MSE;
	float MaxOdleglosc2krok;
	float wartosc_ostatniego = 0;
	vector<float>VectorMSE;
	vector<int>Wektor_kordynat_x;
	vector<int>Wektor_kordynat_y;
	Wektor_kordynat_x.reserve(N_wielkosc_Grupy3D2krok);
	Wektor_kordynat_x.clear();
	Wektor_kordynat_y.reserve(N_wielkosc_Grupy3D2krok);
	Wektor_kordynat_y.clear();
	VectorMSE.reserve(N_wielkosc_Grupy3D2krok);
	WektorWskaznikowMacierzyWag2krok.clear();
	WektorWskaznikowMacierzyWyjsc2krok.clear();
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
			MSE = ObliczanieMSE_szumMaly(tablica_latek_po_1_kroku[i][j], tablica_latek_po_1_kroku[x][y], latka2krok);
			if (MSE < MaxOdleglosc2krok)
			{
				if (ilosc == 0)
				{
					VectorMSE.push_back(MSE);
					Wektor_kordynat_x.push_back(x);
					Wektor_kordynat_y.push_back(y);
					wartosc_ostatniego = MSE;
					ilosc = 1;
				}
				else if (ilosc < N_wielkosc_Grupy3D2krok)
				{
					if (MSE < wartosc_ostatniego)
					{
						for (int n = 0; n < ilosc; n++)
						{
							if (MSE < VectorMSE[n])
							{
								VectorMSE.insert(VectorMSE.begin() + n, MSE);
								Wektor_kordynat_y.insert(Wektor_kordynat_y.begin() + n, y);
								Wektor_kordynat_x.insert(Wektor_kordynat_x.begin() + n, x);
								wartosc_ostatniego = VectorMSE[ilosc];
								ilosc += 1;
								break;
							}

						}
					}
					else
					{
						VectorMSE.push_back(MSE);
						Wektor_kordynat_x.push_back(x);
						Wektor_kordynat_y.push_back(y);
						wartosc_ostatniego = MSE;
						ilosc += 1;
					}
				}
				else if (wartosc_ostatniego > MSE)
				{
					for (int n = 0; n < VectorMSE.size(); n++)
					{
						if (MSE < VectorMSE[n])
						{
							VectorMSE.insert(VectorMSE.begin() + n, MSE);
							Wektor_kordynat_y.insert(Wektor_kordynat_y.begin() + n, y);
							Wektor_kordynat_x.insert(Wektor_kordynat_x.begin() + n, x);
							VectorMSE.pop_back();
							Wektor_kordynat_y.pop_back();
							Wektor_kordynat_x.pop_back();
							wartosc_ostatniego = VectorMSE[ilosc];
							break;
						}
					}
				}
			}
		}
	}
	int wielkosc_grupy = VectorMSE.size();
	if (transformata == 2)
	{
		wielkosc_grupy = NajbizszawielokrotnoscDw(wielkosc_grupy);
	}
	else if (transformata == 1)
	{
		if (wielkosc_grupy % 2 != 0 && wielkosc_grupy > 2)
		{
			VectorMSE.pop_back();
			wielkosc_grupy = wielkosc_grupy - 1;
		}
	}
	for (int i = 0; i < wielkosc_grupy; i++)
	{
		WektorGrupy3d_2krok.push_back(TablicaReferencyjna2[Wektor_kordynat_x[i]][Wektor_kordynat_y[i]].clone());
		WektorGrupy3dpo1kroku.push_back(tablica_latek_po_1_kroku[Wektor_kordynat_x[i]][Wektor_kordynat_y[i]].clone());
		WektorWskaznikowMacierzyWyjsc2krok.push_back(cv::Mat(MacierzWyjsciowa2krok, cv::Rect(Wektor_kordynat_x[i], Wektor_kordynat_y[i], 8, 8)));
		WektorWskaznikowMacierzyWag2krok.push_back(cv::Mat(MacierzWag2krok, cv::Rect(Wektor_kordynat_x[i], Wektor_kordynat_y[i], 8, 8)));
	}
}

//******************************************************************************************************************

void FiltrowanieFiltremWienaDCT(vector<cv::Mat>& WektorGrupy3d_2krok, vector<cv::Mat>& WektorGrupy3dpo1kroku, int k_Wien, float& wagaWien, int sigma)
{

	int s = k_Wien;

	float sigma1 = sigma * sigma;
	float suma = 0.0f;

	int N = WektorGrupy3d_2krok.size();
	/*
	if (N == 1)
	{
		wagaWien = 10000000 * sigma;
		int g = 0;
		while (g < N)
		{
			idct(WektorGrupy3d_2krok[g], WektorGrupy3d_2krok[g]);
			g++;
		}

		return;
	}
	*/
	wagaWien = 0;
	for (int i = 0; i < s; i++)
	{
		for (int j = 0; j < s; j++)
		{
			Mat Tymczasowa_po1kroku(1, N, CV_32F);
			Mat Tymczasowa_2krok(1, N, CV_32F);

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
				tmp = tmp / (tmp + sigma1);
			
				wskazniklMacierzyTymczasowa_2krok[n] = tmp * wskazniklMacierzyTymczasowa_2krok[n];
				wagaWien += tmp;
			}

			idct(Tymczasowa_2krok, Tymczasowa_2krok, DCT_ROWS);
			for (int n = 0; n < (N); n++)

			{
				float* wskazniklWektora2krok = WektorGrupy3d_2krok[n].ptr<float>(i);
				wskazniklWektora2krok[j] = wskazniklMacierzyTymczasowa_2krok[n];
			}

		}
	}
	int g = 0;
	while (g < N)
	{
		idct(WektorGrupy3d_2krok[g], WektorGrupy3d_2krok[g]);
		g++;
	}
}

void FiltrowanieFiltremWienaWalsh(vector<cv::Mat>& WektorGrupy3d_2krok, vector<cv::Mat>& WektorGrupy3dpo1kroku, int k_Wien, float& wagaWien, int sigma)
{
	int s = k_Wien;
	for (int i = 0; i < s; i++)
	{
		for (int j = 0; j < s; j++)

		{
			Walsh_Hadamard(WektorGrupy3d_2krok, i, j);
			Walsh_Hadamard(WektorGrupy3dpo1kroku, i, j);
		}
	}

	float sigma1 = sigma * sigma;
	wagaWien = 0;
	float wspolczynnik = 0;
	for (int i = 0; i < WektorGrupy3dpo1kroku.size(); i++)
	{	
		for (int x = 0; x < k_Wien; x++)
		{
			float* pixel1 = WektorGrupy3dpo1kroku[i].ptr<float>(x);
			float* pixel2 = WektorGrupy3d_2krok[i].ptr<float>(x);
		
			for (int y = 0; y < k_Wien; y++)
			{
				wspolczynnik = pixel1[y] * pixel1[y];
				wspolczynnik = wspolczynnik / (wspolczynnik + sigma1);
				pixel2[y] = pixel2[y]* wspolczynnik;
				wagaWien += wspolczynnik;
			}
		}
	}
	
	for (int i = 0; i < s; i++)
	{
		for (int j = 0; j < s; j++)

		{
			Walsh_Hadamard(WektorGrupy3d_2krok, i, j);
		}
	}
	
	int v = WektorGrupy3d_2krok.size();
	int g = 0;
	while (g < v)
	{
		idct(WektorGrupy3d_2krok[g], WektorGrupy3d_2krok[g]);
		g++;
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
					int nowa_wartosc = cv::saturate_cast<uchar>(pixele[c] + szum);
					pixele[c] = nowa_wartosc;
				}
			}
			else
			{
				cv::Vec3b& pixele = obrazek_zaszumiony.at<cv::Vec3b>(y, x);
				for (int c = 0; c < 3; c++)
				{
					double szum = distribution(generator);
					int nowa_wartosc = cv::saturate_cast<uchar>(pixele[c] + szum);
					pixele[c] = nowa_wartosc;
				}
			}


		}
	}
}

void funkcja_glowna(cv::Mat Obrazek, cv::Mat &MacierzWyjsciowa1krok, cv::Mat &MacierzWyjsciowa2krok, int sigma, int tryb_szybkosci, int transformata, cv::Mat Macierz_Kaisera)
{

	int p_Hard = 3; //p_Hard krok tworzenia latek, w oryginale 1,2 lub 3, u Lebruna 3
	int p_Wien = 3; //krok tworzenia latek, w oryginale 1, 2 lub 3, u Lebruna 3
	float LambdaHard2d = 2.0; //Lambda_hard2d progowanie(trasholding) przy block matchingu, u Lebruna 2.0
	float LambdaHard3d = 2.7; //LambdaHard3d	progowanie(trasholding) Grupy3d w pierwszym kroku filtra, u Lebruna 2,7
	int k_Hard = 8; //k_Hard Wielkosc Latki w 1 kroku, dla DTC 8
	int k_Wien = 8;// Wielkosc Latki w 1 kroku, dla DTC 8
	int N_Hard = 16;//N_Hard Maksymalna wielkość grupy 3d w 1kroku, dla DCT musi być parzysta, u Lebruna 16
	int N_Wien = 32; // Maksymalna wielkość grupy 3d w 2kroku, dla DCT musi być parzysta u Lebruna 32
	int n_Hard = 38;//n_Hard Wielkosc okna szukania podobych latek w pierwszym kroku
	int n_Wien = 38; // Wielkosc okna szukania podobych latek w drugim kroku
	float tau_Hard_niski = 25000;//tau_Hard_niski -maksymalna odleglosc latek przy szumie małym (ponizej 40) - u Lebruna 2500
	float tau_Hard_wysoki = 15000.0;//tau_Hard_wysoki -maksymalna odleglosc latek przy szumie duzym (powyzej (40) - u Lebruna 5000
	float tau_Wien_niski = 400;// -maksymalna odleglosc latek przy szumie małym(ponizej 40) - u Lebruna 400
	float tau_Wien_wysoki = 1500;// -maksymalna odleglosc latek przy szumie duzym(powyzej(40) - u Lebruna 800
	cv::Mat ObrazekReferencyjny;
	cv::Mat** tablica_latek_1;
	cv::Mat** tablica_latek_po_1_kroku;
	cv::Mat MacierzWag1krok;
	cv::Mat MacierzWag2krok;
	std::vector<cv::Mat>WektorWskaznikowMacierzyWyjsc;
	std::vector<cv::Mat>WektorWskaznikowMacierzyWag;
	std::vector<cv::Mat>WektorWskaznikowMacierzyWyjsc2krok;
	std::vector<cv::Mat>WektorWskaznikowMacierzyWag2krok;
	std::vector<cv::Mat> WektorGrupy3d_2krok;
	std::vector<cv::Mat> WektorGrupy3dpo1kroku;
	std::vector<cv::Mat> WektorGrupy3d_1krok;
	int szerokosc_obrazka;
	int wysokosc_obrazka;
	int szerokosc_Tablicy;
	int wysokosc_Tablicy;
	int opcja_obrazka;
	float waga_ht = 0;
	float wagaWien = 0;
	bool szumDuzy;


	Obrazek.convertTo(Obrazek, CV_32F);

	szerokosc_obrazka = Obrazek.cols;
	wysokosc_obrazka = Obrazek.rows;
	szerokosc_Tablicy = Obrazek.cols - k_Hard + 1;
	wysokosc_Tablicy = Obrazek.rows - k_Hard + 1;

	MacierzWyjsciowa1krok = cv::Mat::zeros(wysokosc_obrazka, szerokosc_obrazka, CV_32F);
	MacierzWyjsciowa2krok = cv::Mat::zeros(wysokosc_obrazka, szerokosc_obrazka, CV_32F);

	MacierzWyjsciowa1krok = cv::Mat::zeros(wysokosc_obrazka, szerokosc_obrazka, CV_32F);
	MacierzWag1krok = cv::Mat::zeros(wysokosc_obrazka, szerokosc_obrazka, CV_32F);
	MacierzWyjsciowa2krok = cv::Mat::zeros(wysokosc_obrazka, szerokosc_obrazka, CV_32F);
	MacierzWag2krok = cv::Mat::zeros(wysokosc_obrazka, szerokosc_obrazka, CV_32F);
	WektorGrupy3d_1krok.reserve(N_Hard);
	WektorWskaznikowMacierzyWyjsc.reserve(N_Hard);
	WektorWskaznikowMacierzyWag.reserve(N_Wien);
	WektorWskaznikowMacierzyWyjsc2krok.reserve(N_Wien);
	WektorGrupy3d_2krok.reserve(N_Wien);
	WektorGrupy3dpo1kroku.reserve(N_Wien);

	tablica_latek_1 = new Mat * [szerokosc_Tablicy];
	for (int i = 0; i < szerokosc_Tablicy; ++i)
	{
		tablica_latek_1[i] = new Mat[wysokosc_Tablicy];
	}

	tablica_latek_po_1_kroku = new Mat * [szerokosc_Tablicy];
	for (int i = 0; i < szerokosc_Tablicy; ++i)
	{
		tablica_latek_po_1_kroku[i] = new Mat[wysokosc_Tablicy];
	}

	//Najpierw tworzymy tablice zawierającą łatki pokrywające obrazek i od razu łatki te poddajemy DCT2D//


	for (int i = 0; i < szerokosc_Tablicy; i++)
	{
		for (int j = 0; j < wysokosc_Tablicy; j++)
		{
			tablica_latek_1[i][j] = cv::Mat(Obrazek(Rect(i, j, k_Hard, k_Hard)).clone());
			cv::dct(tablica_latek_1[i][j], tablica_latek_1[i][j]);
		}
	}
	szumDuzy = false;
	if (sigma > 40) szumDuzy = true;
	if (tryb_szybkosci == 1)
	{
		p_Hard = 4;
		p_Wien = 4;
	}

	for (int i = 0; i < szerokosc_Tablicy + p_Hard; i += p_Hard)
	{
		for (int j = 0; j < wysokosc_Tablicy + p_Hard; j += p_Hard)

		{
			//zabezpieczenie na wypadek gsyny przy p_Hard =3 iteracja przeskaiwała poza ostatnie piksele
			int x = i;
			int y = j;
			if (i >= szerokosc_Tablicy) x = szerokosc_Tablicy - 1;
			if (j >= wysokosc_Tablicy) y = wysokosc_Tablicy - 1;

			wyszukiwacz1krok(x, y, k_Hard, n_Hard, tablica_latek_1, MacierzWyjsciowa1krok, MacierzWag1krok, szerokosc_Tablicy, wysokosc_Tablicy, WektorGrupy3d_1krok, WektorWskaznikowMacierzyWyjsc, WektorWskaznikowMacierzyWag, szumDuzy, LambdaHard2d, tau_Hard_wysoki, tau_Hard_niski, N_Hard, transformata, sigma);

			HardTresholding_3D(WektorGrupy3d_1krok, k_Hard, LambdaHard3d, waga_ht, sigma, transformata);

			int index = 0;
			int index_max = WektorGrupy3d_1krok.size();
			while (index < index_max)
			{
				WektorWskaznikowMacierzyWyjsc[index] = (WektorWskaznikowMacierzyWyjsc[index]) + (WektorGrupy3d_1krok[index].mul(Macierz_Kaisera) * (1 / waga_ht));
				WektorWskaznikowMacierzyWag[index] = WektorWskaznikowMacierzyWag[index] + ((1 / waga_ht) * Macierz_Kaisera);
				index++;
			}

		}
	}

	MacierzWyjsciowa1krok = MacierzWyjsciowa1krok / MacierzWag1krok;

	///////////////////////////////////////////////////////////////////////Drugi KROK/////////////////////////////////////////////


	for (int i = 0; i < szerokosc_Tablicy; i++)
	{
		for (int j = 0; j < wysokosc_Tablicy; j++)

		{
			tablica_latek_po_1_kroku[i][j] = cv::Mat(MacierzWyjsciowa1krok(Rect(i, j, k_Wien, k_Wien)).clone()); // tak powinno być, żeby stworzyć głęboką kopię
			cv::dct(tablica_latek_po_1_kroku[i][j], tablica_latek_po_1_kroku[i][j]);
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


			wyszukiwacz2krok(x, y, k_Wien, n_Wien, tablica_latek_1, tablica_latek_po_1_kroku, MacierzWyjsciowa2krok, MacierzWag2krok, szerokosc_Tablicy, wysokosc_Tablicy, WektorGrupy3d_2krok, WektorGrupy3dpo1kroku, WektorWskaznikowMacierzyWyjsc2krok, WektorWskaznikowMacierzyWag2krok, szumDuzy, tau_Wien_wysoki, tau_Wien_niski, N_Wien, transformata);

			if (transformata == 1)
			{
				FiltrowanieFiltremWienaDCT(WektorGrupy3d_2krok, WektorGrupy3dpo1kroku, k_Wien, wagaWien, sigma);
			}
			else if (transformata == 2)
			{
				FiltrowanieFiltremWienaWalsh(WektorGrupy3d_2krok, WektorGrupy3dpo1kroku, k_Wien, wagaWien, sigma);
			}

			int index = 0;
			int index_max = WektorGrupy3d_2krok.size();
			while (index < index_max)
			{
				WektorWskaznikowMacierzyWyjsc2krok[index] = (WektorWskaznikowMacierzyWyjsc2krok[index]) + (WektorGrupy3d_2krok[index].mul(Macierz_Kaisera) * (1 / wagaWien));
				WektorWskaznikowMacierzyWag2krok[index] = WektorWskaznikowMacierzyWag2krok[index] + ((1 / wagaWien) * Macierz_Kaisera);
				;
				index++;
			}
		}
	}

	MacierzWyjsciowa2krok = MacierzWyjsciowa2krok / MacierzWag2krok;
	MacierzWyjsciowa1krok.convertTo(MacierzWyjsciowa1krok, CV_8U);
	MacierzWyjsciowa2krok.convertTo(MacierzWyjsciowa2krok, CV_8U);
	for (int i = 0; i < szerokosc_Tablicy; ++i)
	{
		delete[] tablica_latek_1[i]; //uwolnienie pamieci
		delete[] tablica_latek_po_1_kroku[i];
	}

	delete[] tablica_latek_1;
	delete[] tablica_latek_po_1_kroku;
	tablica_latek_1 = NULL;
	tablica_latek_po_1_kroku = NULL;
}

int main(int argc, char* argv[])
{	
	float Okno_Kaisera[8][8] =
	{ 0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924,
	  0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
	  0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
	  0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
	  0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
	  0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
	  0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
	  0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924 };
	
	int sigma; // sigma - poziom szumu
	cv::Mat Macierz_Kaisera;
	cv::Mat Obrazek;
	cv::Mat MacierzWyjsciowa1krok;
	cv::Mat MacierzWyjsciowa2krok;
	std::string wpisana_nazwa;
	std::string nazwa_pilku_zaszumionego;
	std::string nazwa_sciezki = "obrazki_testowe/";

	int transformata = 2;
	int tryb_szybkosci=0;
	int licznik = 0;

	Macierz_Kaisera = cv::Mat(8, 8, CV_32F, Okno_Kaisera);
	if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) 
	{
		std::cout << "Filtr BM3D CPU, filtruje z szumu obrazy w skali szarosci.\n";
		std::cout << "Uzycie: BM3D_CPU.exe <nazwa pliku> <liczba całkowita> <liczba zmiennoprzecinkowa> <bool>\n";
		std::cout << "Argumenty:\n";
		std::cout << "  <nazwa pliku>       Nazwa pliku (string). Mozna podac nazwe i sciezke folderu lub sama nazwe \n";
		std::cout << "                      jezeli znajduje sie w jednym folderze z programem - zostanąa przetworzone \n";
		std::cout << "                      wszystkie pliki graficzne w folderze\n";
		std::cout << "  <poziom szumu>      Liczba calkowita (int)\n";
		std::cout << "  <tryb szybkosci>	0 - normalny, 1 - szybki\n";
		return 0;  
	}

	if (argc != 4) { 
		std::cerr << "Użycie: " << argv[0] << " <nazwa pliku> <poziom szumu> <normalny czy szybki?>\n pomoc: --help lub -h";
		cv::waitKey(0);
		return 1;
	}

	wpisana_nazwa = argv[1];       //nazwa wczytywanego pliku argv[0] to nazwa programu
	sigma = std::atoi(argv[2]);  // drugi arg - poziom szumu
	tryb_szybkosci = std::atoi(argv[3]); // Argument float
	
	time_t czasStart = time(NULL);

	if (std::filesystem::is_regular_file(wpisana_nazwa)) 
	{
		Obrazek = cv::imread(wpisana_nazwa, cv::IMREAD_GRAYSCALE);

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
	else if (std::filesystem::is_directory(wpisana_nazwa)) {
		for (const auto& entry : std::filesystem::directory_iterator(wpisana_nazwa)) {
			// Sprawdzenie, czy plik ma odpowiednie rozszerzenie
			if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg" || entry.path().extension() == ".png"
				|| entry.path().extension() == ".bmp" || entry.path().extension() == ".tiff" || entry.path().extension() == ".tif" || entry.path().extension() == ".webp"
				|| entry.path().extension() == ".hdr" || entry.path().extension() == ".jp2"))			
			{
				Obrazek = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

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
	else {
		std::cerr << "Podana ścieżka lub nazwa pliku  jest bledna" << std::endl;
		return 1;
	}
	time_t czasStop = time(NULL);
	std::cout << "Przefiltrowano " << licznik<< " obrazow w czasie : " << czasStop - czasStart << " s." << std::endl;
	
	cv::waitKey(0);

	return 0;

}
