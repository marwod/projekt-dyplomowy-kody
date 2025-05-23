﻿/*
© Marcin Wodejko 2024.
marwod@interia.pl
*/

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <stdint.h>
#include <string>
#include <iostream>
#include <cmath>
#include <math.h>
#include <cmath>
#include<vector>
#include <assert.h>
#include <random>
#include <numeric>


using namespace std;
using namespace cv;



//******************************************************************************************************************
///////////////////////////////FUNKCJE OGoLNE////////////////////////////////////////////////


inline float ObliczanieMSE_Gray(cv::Mat referencyjna, cv::Mat porownywana, int wielkosc_okienka, float** tablica_rozkladu_gaussa)
{

	float sum = 0.0f;
	float dzielnik = wielkosc_okienka * wielkosc_okienka;

	for (int i = 0; i < wielkosc_okienka; i++)

	{
		float* pixel1 = referencyjna.ptr<float>(i);
		float* pixel2 = porownywana.ptr<float>(i);
		for (int j = 0; j < wielkosc_okienka; j++)
		{
			float K = (float)tablica_rozkladu_gaussa[i][j];
			sum += K * ((pixel1[j] - pixel2[j]) * (pixel1[j] - pixel2[j]));
		}
	}
	float mad = sum;

	return mad;
}

inline Vec3f ObliczanieMSE_Kolor(cv::Mat referencyjna, cv::Mat porownywana,
	int wielkosc_okienka, float** tablica_rozkladu_gaussa)
{
	Vec3f sum = 0.0f;
	float dzielnik = (wielkosc_okienka * wielkosc_okienka);

	for (int i = 0; i < wielkosc_okienka; i++)
	{
		for (int j = 0; j < wielkosc_okienka; j++)
		{
			Vec3f p_referencyjny = referencyjna.at<Vec3f>(j, i);
			Vec3f p_porownywany = porownywana.at<Vec3f>(j, i);
			float K = tablica_rozkladu_gaussa[i][j];
			sum[0] += K * ((p_referencyjny[0] - p_porownywany[0])
				* (p_referencyjny[0] - p_porownywany[0]));
			sum[1] += K * ((p_referencyjny[1] - p_porownywany[1])
				* (p_referencyjny[1] - p_porownywany[1]));
			sum[2] += K * ((p_referencyjny[2] - p_porownywany[2])
				* (p_referencyjny[2] - p_porownywany[2]));
		}

	}
	return sum;
}


inline void NLM_Grey(cv::Mat Okno_Przeszukania, cv::Mat Okienko_Referencyjne, int wielkosc_okna_przeszukania, int wielkosc_okna_podobienstwa,
	float& wartosc_pixela_odszumionego, float sigma, float stala_h, int polowa_okna_przeszukania, int polowa_okna_podobienstwa, float** tablica_rozkladu_gaussa, bool  Bodues)
{
	std::vector<float>Vector_Wag;
	std::vector<float>Vector_Pixeli;

	for (int i = polowa_okna_podobienstwa; i < wielkosc_okna_przeszukania + polowa_okna_podobienstwa; i++)
	{
		for (int j = polowa_okna_podobienstwa; j < wielkosc_okna_przeszukania + polowa_okna_podobienstwa; j++)
		{

			cv::Mat Okienko_porownywane = Okno_Przeszukania(cv::Rect(i - polowa_okna_podobienstwa,
				j - polowa_okna_podobienstwa, wielkosc_okna_podobienstwa, wielkosc_okna_podobienstwa));
			float MSE = ObliczanieMSE_Gray(Okienko_Referencyjne, Okienko_porownywane, wielkosc_okna_podobienstwa, tablica_rozkladu_gaussa);

			if (Bodues == true) MSE = fmaxf((MSE - 2 * sigma * sigma), 0.0f);
			float waga = std::expf(-(MSE / (stala_h * stala_h)));

			Vector_Wag.push_back(waga);
			float wartosc_pixela = Okno_Przeszukania.at<float>(j, i);
			wartosc_pixela = wartosc_pixela * waga;
			Vector_Pixeli.push_back(wartosc_pixela);

		}
	}
	float suma_pikseli = 0;
	float suma_wag = 0;
	for (std::vector<float>::iterator it = Vector_Pixeli.begin(); it != Vector_Pixeli.end(); ++it)
		suma_pikseli = suma_pikseli + *it;
	for (std::vector<float>::iterator it = Vector_Wag.begin(); it != Vector_Wag.end(); ++it)
		suma_wag = suma_wag + *it;
	wartosc_pixela_odszumionego = suma_pikseli / suma_wag;

}

inline void NLM_Kolor(cv::Mat Okno_Przeszukania, cv::Mat Okienko_Referencyjne,
	int wielkosc_okna_przeszukania, int wielkosc_okna_podobienstwa,
	Vec3f& wartosc_pixela_odszumionego_kolor, float sigma, float stala_h,
	int polowa_okna_przeszukania, int polowa_okna_podobienstwa,
	float** tablica_rozkladu_gaussa, bool Bodues)
{

	std::vector<float>Vector_Wag;
	std::vector<Vec3f>Vector_Pixeli;

	for (int i = polowa_okna_podobienstwa; i < wielkosc_okna_przeszukania + polowa_okna_podobienstwa; i++)
	{
		for (int j = polowa_okna_podobienstwa; j < wielkosc_okna_przeszukania + polowa_okna_podobienstwa; j++)
		{

			Vec3f MSE(0, 0, 0);
			cv::Mat Okienko_porownywane = Okno_Przeszukania(cv::Rect(i - polowa_okna_podobienstwa, j - polowa_okna_podobienstwa, wielkosc_okna_podobienstwa, wielkosc_okna_podobienstwa));
			MSE = ObliczanieMSE_Kolor(Okienko_Referencyjne, Okienko_porownywane, wielkosc_okna_podobienstwa, tablica_rozkladu_gaussa);
			float zsumowane_MSE = (MSE[0] + MSE[1] + MSE[2]) / 3;
			if (Bodues == true) zsumowane_MSE = fmaxf((zsumowane_MSE - 2 * sigma * sigma), 0.0f);
			float waga = std::expf(-(zsumowane_MSE / (stala_h * stala_h)));
			Vector_Wag.push_back(waga);
			Vec3f wartosc_pixela = Okno_Przeszukania.at<Vec3f>(j, i);
			wartosc_pixela[0] = wartosc_pixela[0] * waga;
			wartosc_pixela[1] = wartosc_pixela[1] * waga;
			wartosc_pixela[2] = wartosc_pixela[2] * waga;
			Vector_Pixeli.push_back(wartosc_pixela);
		}
	}

	Vec3f suma_pikseli(0, 0, 0);
	float suma_wag = 0;
	for (int i = 0; i < Vector_Pixeli.size(); ++i)
		suma_pikseli[0] = suma_pikseli[0] + Vector_Pixeli[i][0];
	for (int i = 0; i < Vector_Pixeli.size(); ++i)
		suma_pikseli[1] = suma_pikseli[1] + Vector_Pixeli[i][1];
	for (int i = 0; i < Vector_Pixeli.size(); ++i)
		suma_pikseli[2] = suma_pikseli[2] + Vector_Pixeli[i][2];
	for (std::vector<float>::iterator it = Vector_Wag.begin(); it != Vector_Wag.end(); ++it)
		suma_wag = suma_wag + *it;
	wartosc_pixela_odszumionego_kolor[0] = fminf(fmaxf(suma_pikseli[0] / suma_wag, 0), 255);
	wartosc_pixela_odszumionego_kolor[1] = fminf(fmaxf(suma_pikseli[1] / suma_wag, 0), 255);
	wartosc_pixela_odszumionego_kolor[2] = fminf(fmaxf(suma_pikseli[2] / suma_wag, 0), 255);

}







//******************************************************************************************************************


void dodanie_szumu(cv::Mat obrazek_zaszumiony, double sigm, int ilosc_kanalow)
{

	double sigma = sigm; // Wartosc sigma dla szumu gaussowskiego
	// Generator liczb losowych dla szumu gaussowskiego
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, sigma);
	// Dodaje szum gaussowski do kazdego piksela
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


float Funkcja_Gaussa(float x, float y, float a)
{
	return exp(-(x * x + y * y) / (2 * a * a));
}

float Generacja_K_z(float x, float y, float a, float polowa_latki)
{
	float licznik = Funkcja_Gaussa(x, y, a);
	float mianownik = 0.0;

	for (int i = -polowa_latki; i <= polowa_latki; ++i)
	{
		for (int j = -polowa_latki; j <= polowa_latki; ++j) {
			if (std::max(std::abs(i), std::abs(j)) <= polowa_latki)
			{
				mianownik += Funkcja_Gaussa(i, j, a);
			}
		}
	}

	return licznik / mianownik;
}


int main(int argv, char* argc)
{

	int kolor_obrazka = 1; // zawiera informację czy przetwarzany obraz będzie w skali szarosci czy kolorze
	int opcja_obrazka; // zawiera informację czy przetwarzany obraz jest juz zaszumiiony czy dodac szum
	float sigma; // poziom szumu
	float stala_h; // parametr okreslany przed procesem odszumiania w zaleznosci od szumu i wielkosci  Okienka referncyjnego, uzywany w procesie obiczania wagi piksela
	bool Froment = true; // czy stosowac obliczenie K zfgodnie z propozycją z artykulu Fromenta;
	bool Bodues = false; //czy odejmowac 2*sigma*sigma od obliczoej odlegosci pomiędzy pikselami
	cv::Mat Obrazek; // obiekt opencv Mat w ktorym będzie zapisany przetwarzany obrazek
	cv::Mat ObrazekReferencyjny; //obiekt opencv Mat w ktorym będzie wczytany obrazek bez szumu
	cv::Mat NLM_OpenC;// obekt opencv Mat w ktorym będzie zapisany obrazek odszumiony funkcją NLM z bibliotek opencv
	cv::Mat MacierzWyjsciowa; // obiekt opencv Mat w ktorym będzie zapisane piksele po odszumieniu
	cv::Mat Okno_Przeszukania; //obiekt opencv Mat ktory będzie stanowil "wskaznik" na obszar obrazka uwzględniany przy obliczaniu piksela odszumionego
	cv::Mat Okienko_Referencyjne; //obiekt opencv Mat ktory będzie stanowil "wskaznik" na obszar obrazka uwzględniany przy obliczaniu podobienstwa między dwoam pikselami
	std::string nazwa_pilku_zaszumionego; //nazwa wczytywanego pliku wraz z rozszerzeniem, bez sciezki;
	std::string nazwa_pilku_referencyjnego; //nazwa wczytywanego pliku wraz z rozszerzeniem, bez sciezki;
	std::string nazwa_sciezki = "obrazki_testowe/";// nazwa sciezki wzglednej do pliku w folderze "obrazki_testowe"
	int orginal_obrazek_width; // szerokosc obrazka orginalnego, bez dodanych marginesow
	int orginal_obrazek_height; // wysokosc obrazka orginalnego, bez dodanych marginesow
	int wielkosc_okna_podobienstwa; // wielkosc obszaru obrazka uwzględniany przy obliczaniu podobienstwa między dwoma pikselami
	int wielkosc_okna_przeszukania; //wielkosc obsaru obrazka uwzględnianego przy obliczaniu piksela odszumionego, 21 alblo 35 pixeli
	int polowa_okna_przeszukania; //dlugosc obszaru pomiędzy pikselem odszymianym a granicą okna, (wielkosc_okna_przeszukania-1)/2 
	int polowa_okna_podobienstwa; //dlugosc obszaru pomiędzy pikselem odszymianym a granicą okna, (wielkosc_okna_podobienstwa-1)/2
	int rzeczywiste_okno_przeszukania; //wielkosc okna podobienstwa poszerzonegopo obu stronach o polowy okna przeszukania
	int polowa_rzeczywistego_okna_przeszukania;// dlugosc obszaru pomiędzy pikselem odszymianym a granicą okna, (rzeczywiste_okna_przeszukania - 1) / 2
	int wielkosc_marginesu;// margines dodawany do obrazka z lewej strony i od gory w celu odszumienia pikseli przy brzegach obrazka 
	//int wielkosc_marginesu_prawego;// margines dodawany do obrazka z prawej strony i od dolu w celu odszumienia pikseli przy brzegach obrazka 
	int obrazek_z_marginesami_width;// szerokosc obrazka po dodaniu marginesow
	int obrazek_z_marginesami_height; // wysokosc pobrazka po dodaniu marginesow
	double poziom_rozmycia_gauss; // poziom rozmycia fitrem gaussa (w przypadku zastosowania prefiltracji) zalezny od poziomu szumu
	float redukcja_sigmy; // wspolczynnik redukcji wsplczynnika szumu podczas dalszego przetwarzania, po zastosowaniu prefiltracjii gaussem
	float wartosc_pixela_odszumionego;// tymczasowe miejsce zapisu wartosci odszumionego piksela w obrazku szarym
	Vec3f wartosc_pixela_odszumionego_kolor; // tymczasowe miejsce zapisu wartosci odszumionego piksela w obrazku kolorowym



	do
	{
		std::cout << "Obrazek w skali szarosci czy kolowowy" << std::endl;
		std::cout << "1) skala szarosci" << std::endl;
		std::cout << "2) kolorowy" << std::endl;
		std::cin >> kolor_obrazka;
	} while (kolor_obrazka != 1 && kolor_obrazka != 2);
	kolor_obrazka = kolor_obrazka - 1;

	do
	{
		std::cout << "czy chcesz przetwarzac juz zaszumiony obrazek, czy dodac szum do obrazka referencyjnego>" << std::endl;
		std::cout << "1) chce przetwarzac juz zaszumiony obrazek" << std::endl;
		std::cout << "2) chce dodac szum do obrazka referencyjnego" << std::endl;
		std::cin >> opcja_obrazka;
	} while (opcja_obrazka != 1 && opcja_obrazka != 2);

	if (opcja_obrazka == 1)
	{
		std::cout << "prosze wpisac nazwe pliku obrazka zaszumionego wraz z rozszerzeniem" << std::endl;
		std::cin >> nazwa_pilku_zaszumionego;
		//nazwa_pilku_zaszumionego = nazwa_sciezki + nazwa_pilku_zaszumionego;
		std::cout << "prosze wpisac nazwe pliku obrazka referencyjnego wraz z rozszerzeniem" << std::endl;
		std::cin >> nazwa_pilku_referencyjnego;
		//nazwa_pilku_referencyjnego = nazwa_sciezki + nazwa_pilku_referencyjnego;
		Obrazek = cv::imread(nazwa_pilku_zaszumionego, kolor_obrazka);
		ObrazekReferencyjny = cv::imread(nazwa_pilku_referencyjnego, kolor_obrazka);
		if (Obrazek.empty())
		{
			std::cout << "Nie mozna wczytac obrazka do odszumienia." << std::endl;
			cv::waitKey(0);
			return -1;
		}
		if (ObrazekReferencyjny.empty())
		{
			std::cout << "Nie mozna wczytac obrazka referencyjnego." << std::endl;
			cv::waitKey(0);
			return -1;
		}

		std::cout << "prosze podaj poziom szumu" << std::endl;
		std::cin >> sigma;
	}

	else if (opcja_obrazka == 2)
	{
		std::cout << "prosze wpisac nazwe pliku obrazka referencyjnego wraz z rozszerzeniem" << std::endl;
		std::cin >> nazwa_pilku_referencyjnego;
		//nazwa_pilku_referencyjnego = nazwa_sciezki + nazwa_pilku_referencyjnego;
		ObrazekReferencyjny = cv::imread(nazwa_pilku_referencyjnego, kolor_obrazka);
		if (ObrazekReferencyjny.empty())
		{
			std::cout << "Nie mozna wczytac obrazka." << std::endl;
			return -1;
		}
		std::cout << "prosze podaj poziom szumu" << std::endl;
		std::cin >> sigma;
		Obrazek = ObrazekReferencyjny.clone(); // Skopiuj obraz do macierzy z szumem
		if (kolor_obrazka == 0)
		{
			dodanie_szumu(Obrazek, sigma, 1);
		}
		else dodanie_szumu(Obrazek, sigma, 3);
	}

	orginal_obrazek_width = Obrazek.cols;
	std::cout << orginal_obrazek_width << std::endl;
	orginal_obrazek_height = Obrazek.rows;
	std::cout << orginal_obrazek_height << std::endl;


	if (sigma < 25) {
		wielkosc_okna_przeszukania = 15;
	}
	else if (sigma < 60)
	{
		wielkosc_okna_przeszukania = 21;
	}
	else if (sigma < 80)
	{
		wielkosc_okna_przeszukania = 25;
	}
	else
	{
		wielkosc_okna_przeszukania = 31;
	}

	do
	{
		std::cout << "prosze podaj wielkosc okna podobienstwa: 3, 5, 7 lub 9" << std::endl;
		std::cin >> wielkosc_okna_podobienstwa;
	} while ((wielkosc_okna_podobienstwa != 5) && (wielkosc_okna_podobienstwa != 7) && (wielkosc_okna_podobienstwa != 9) && (wielkosc_okna_podobienstwa != 3));

	char filtracja_wstepna;
	do
	{
		std::cout << "czy przeprowadzic wstepna filtracje romyciem gaussa?" << std::endl;
		std::cout << "tak, wcisnij t" << std::endl;
		std::cout << "nie, wcisnij n" << std::endl;
		std::cin >> filtracja_wstepna;
	} while ((filtracja_wstepna != 't') && (filtracja_wstepna != 'n'));

	string odpowiedz1;
	do
	{
		std::cout << "czy podczas obliczania dystanu podobienstwa pomiedzy pikselami stosowac wspolczynnik K oparty na rozkladzie gaussa?" << std::endl;
		std::cout << "tak, wcisnij t" << std::endl;
		std::cout << "nie, wcisnij n" << std::endl;
		std::cin >> odpowiedz1;
		if (odpowiedz1=="t") Froment = true;
		else Froment = false;
	} while ((odpowiedz1 != "t") && (odpowiedz1 != "n"));

	string odpowiedz2;
	do
	{
		std::cout << "czy od sumy dystansow pomiędzy pikselami odjac 2*sigma*sigma?" << std::endl;
		std::cout << "tak, wcisnij t" << std::endl;
		std::cout << "nie, wcisnij n" << std::endl;
		std::cin >> odpowiedz2;
		if (odpowiedz2 == "t")  Bodues = true;
		else Bodues = false;
	} while ((odpowiedz2 != "t") && (odpowiedz2 != "n"));

	std::cout << "prosze podaj wysokosc H" << std::endl;
	std::cin >> stala_h;


	///////////////////////////////////////////////////////

	orginal_obrazek_width = Obrazek.cols;
	orginal_obrazek_height = Obrazek.rows;

	if (sigma < 25) {
		wielkosc_okna_przeszukania = 15;
	}
	else if (sigma < 60)
	{
		wielkosc_okna_przeszukania = 21;
	}
	else if (sigma < 80)
	{
		wielkosc_okna_przeszukania = 25;
	}
	else
	{
		wielkosc_okna_przeszukania = 31;
	}



	//////////////////////////////////////////////////////////////

	cv::fastNlMeansDenoising(Obrazek, NLM_OpenC, sigma, wielkosc_okna_podobienstwa, wielkosc_okna_przeszukania);

		////////////////dodajemy "marginesy" do obrazka (po polowie obszaru przeszukania -po16)/////////////////////////////

	polowa_okna_przeszukania = (wielkosc_okna_przeszukania - 1) / 2;
	polowa_okna_podobienstwa = (wielkosc_okna_podobienstwa - 1) / 2;
	rzeczywiste_okno_przeszukania = wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1;
	polowa_rzeczywistego_okna_przeszukania = polowa_okna_przeszukania + polowa_okna_podobienstwa;

	wielkosc_marginesu = std::floor(wielkosc_okna_przeszukania / 2)
		+ std::floor(wielkosc_okna_podobienstwa / 2);
	cv::copyMakeBorder(Obrazek, Obrazek, wielkosc_marginesu, wielkosc_marginesu,
		wielkosc_marginesu, wielkosc_marginesu, cv::BORDER_REFLECT);
	if (sigma < 50)
	{
		poziom_rozmycia_gauss = 0.5;
		redukcja_sigmy = 0.65;
	}

	else
	{
		poziom_rozmycia_gauss = 1.0;
		redukcja_sigmy = 0.33;
	}

	if (filtracja_wstepna == 't')
	{
		cv::GaussianBlur(Obrazek, Obrazek, cv::Size(3, 3), poziom_rozmycia_gauss, poziom_rozmycia_gauss);
		sigma = redukcja_sigmy * sigma;
	}
	if (kolor_obrazka == 0)
	{
		Obrazek.convertTo(Obrazek, CV_32F);
	}
	else Obrazek.convertTo(Obrazek, CV_32FC3);
	obrazek_z_marginesami_width = Obrazek.cols;
	obrazek_z_marginesami_height = Obrazek.rows;

	time_t czasStart = time(NULL);

	if (kolor_obrazka == 0)
	{
		MacierzWyjsciowa = cv::Mat::zeros(obrazek_z_marginesami_height, obrazek_z_marginesami_width, CV_32F);
	}
	else MacierzWyjsciowa = cv::Mat::zeros(obrazek_z_marginesami_height, obrazek_z_marginesami_width, CV_32FC3);
	
	float** tablica_rozkladu_K = new float* [wielkosc_okna_podobienstwa];

	for (int i = 0; i < wielkosc_okna_podobienstwa; i++)
	{
		tablica_rozkladu_K[i] = new float[wielkosc_okna_podobienstwa];
	}
	for (int i = 0; i < wielkosc_okna_podobienstwa; ++i)
	{
		for (int j = 0; j < wielkosc_okna_podobienstwa; ++j) {
			if (Froment == true)
			{
				tablica_rozkladu_K[i][j] = Generacja_K_z(i, j, sigma / 10, polowa_okna_podobienstwa);
			}
			else
			{
				tablica_rozkladu_K[i][j] = 1.0f / (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa);
			}
		}
	}

	for (int i = wielkosc_marginesu; i < obrazek_z_marginesami_width - wielkosc_marginesu; i++)
	{
		for (int j = wielkosc_marginesu; j < obrazek_z_marginesami_height - wielkosc_marginesu; j++)
		{
			if (kolor_obrazka == 0)
			{
				Okno_Przeszukania = Obrazek(cv::Rect(i - polowa_rzeczywistego_okna_przeszukania, j - polowa_rzeczywistego_okna_przeszukania, rzeczywiste_okno_przeszukania, rzeczywiste_okno_przeszukania));
				Okienko_Referencyjne = Obrazek(cv::Rect(i - polowa_okna_podobienstwa, j - polowa_okna_podobienstwa, wielkosc_okna_podobienstwa, wielkosc_okna_podobienstwa));

				NLM_Grey(Okno_Przeszukania, Okienko_Referencyjne, wielkosc_okna_przeszukania, wielkosc_okna_podobienstwa, wartosc_pixela_odszumionego, sigma, stala_h, polowa_okna_przeszukania, polowa_okna_podobienstwa, tablica_rozkladu_K, Bodues);
				MacierzWyjsciowa.at<float>(j, i) = wartosc_pixela_odszumionego;
			}
			else
			{
				Okno_Przeszukania = Obrazek(cv::Rect(i - polowa_rzeczywistego_okna_przeszukania, j - polowa_rzeczywistego_okna_przeszukania, rzeczywiste_okno_przeszukania, rzeczywiste_okno_przeszukania));
				Okienko_Referencyjne = Obrazek(cv::Rect(i - polowa_okna_podobienstwa, j - polowa_okna_podobienstwa, wielkosc_okna_podobienstwa, wielkosc_okna_podobienstwa));

				NLM_Kolor(Okno_Przeszukania, Okienko_Referencyjne, wielkosc_okna_przeszukania, wielkosc_okna_podobienstwa, wartosc_pixela_odszumionego_kolor, sigma, stala_h, polowa_okna_przeszukania, polowa_okna_podobienstwa, tablica_rozkladu_K, Bodues);
				Vec3f& pixel = MacierzWyjsciowa.at<Vec3f>(j, i);
				pixel = wartosc_pixela_odszumionego_kolor;
			}
		}
	}

	for (int i = 0; i < wielkosc_okna_podobienstwa; i++)
	{
		delete[] tablica_rozkladu_K[i];
	}

	delete[] tablica_rozkladu_K;


	if (kolor_obrazka == 0)
	{
		MacierzWyjsciowa.convertTo(MacierzWyjsciowa, CV_8UC1);
	}
	else MacierzWyjsciowa.convertTo(MacierzWyjsciowa, CV_8UC3);

	MacierzWyjsciowa = MacierzWyjsciowa(cv::Rect(wielkosc_marginesu, wielkosc_marginesu, orginal_obrazek_width, orginal_obrazek_height));

	time_t czasStop = time(NULL);
	printf("Uplynelo %.2fsek.", difftime(czasStop, czasStart));

	if(kolor_obrazka == 0)
	{
		Obrazek.convertTo(Obrazek, CV_8UC1);
	}
	else Obrazek.convertTo(Obrazek, CV_8UC3);

	Obrazek = Obrazek(cv::Rect(wielkosc_marginesu, wielkosc_marginesu, orginal_obrazek_width, orginal_obrazek_height));

	cv::imshow("Obrazek zaszumiony", Obrazek);
	cv::imshow("referncyjny", ObrazekReferencyjny);

	cv::imshow("Obrazek odszumiony", MacierzWyjsciowa);
	cv::imshow("funkcja wbudowana NLM", NLM_OpenC);

	double R = 255;
	std::cout << "PSNR implementacja: " << cv::PSNR(ObrazekReferencyjny, MacierzWyjsciowa, R) << std::endl;
	std::cout << "PSNR wbudowany NLM: " << cv::PSNR(ObrazekReferencyjny, NLM_OpenC, R) << std::endl;
	cv::waitKey(0);
	return 0;

}
