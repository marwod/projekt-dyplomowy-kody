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
#include <numeric>
#include <cstdlib>
#include <filesystem> // musi byc C++17 lub wyzej

using namespace std;
using namespace cv;



//******************************************************************************************************************
///////////////////////////////FUNKCJE OGÓLNE////////////////////////////////////////////////


inline float ObliczanieMSE_Gray(cv::Mat referencyjna, cv::Mat porownywana, int wielkosc_okienka)
{

	float sum = 0.0f;
	float dzielnik = wielkosc_okienka * wielkosc_okienka;

	for (int i = 0; i < wielkosc_okienka; i++)

	{
		float* pixel1 = referencyjna.ptr<float>(i);
		float* pixel2 = porownywana.ptr<float>(i);
		for (int j = 0; j < wielkosc_okienka; j++)
		{
			float K = 1/dzielnik;
			sum += K * ((pixel1[j] - pixel2[j]) * (pixel1[j] - pixel2[j]));
		}
	}
	float mad = sum;

	return mad;
}

inline Vec3f ObliczanieMSE_Kolor(cv::Mat referencyjna, cv::Mat porownywana,
	int wielkosc_okienka)
{
	Vec3f sum = 0.0f;
	float dzielnik = (wielkosc_okienka * wielkosc_okienka);

	for (int i = 0; i < wielkosc_okienka; i++)
	{
		for (int j = 0; j < wielkosc_okienka; j++)
		{
			Vec3f p_referencyjny = referencyjna.at<Vec3f>(j, i);
			Vec3f p_porownywany = porownywana.at<Vec3f>(j, i);
			float K = 1/dzielnik;
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
	float& wartosc_pixela_odszumionego, float sigma, float stala_h, int polowa_okna_przeszukania, int polowa_okna_podobienstwa, bool  Bodues)
{
	std::vector<float>Vector_Wag;
	std::vector<float>Vector_Pixeli;

	for (int i = polowa_okna_podobienstwa; i < wielkosc_okna_przeszukania + polowa_okna_podobienstwa; i++)
	{
		for (int j = polowa_okna_podobienstwa; j < wielkosc_okna_przeszukania + polowa_okna_podobienstwa; j++)
		{

			cv::Mat Okienko_porownywane = Okno_Przeszukania(cv::Rect(i - polowa_okna_podobienstwa,
				j - polowa_okna_podobienstwa, wielkosc_okna_podobienstwa, wielkosc_okna_podobienstwa));
			float MSE = ObliczanieMSE_Gray(Okienko_Referencyjne, Okienko_porownywane, wielkosc_okna_podobienstwa);

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
	int polowa_okna_przeszukania, int polowa_okna_podobienstwa, bool Bodues)
{

	std::vector<float>Vector_Wag;
	std::vector<Vec3f>Vector_Pixeli;

	for (int i = polowa_okna_podobienstwa; i < wielkosc_okna_przeszukania + polowa_okna_podobienstwa; i++)
	{
		for (int j = polowa_okna_podobienstwa; j < wielkosc_okna_przeszukania + polowa_okna_podobienstwa; j++)
		{

			Vec3f MSE(0, 0, 0);
			cv::Mat Okienko_porownywane = Okno_Przeszukania(cv::Rect(i - polowa_okna_podobienstwa, j - polowa_okna_podobienstwa, wielkosc_okna_podobienstwa, wielkosc_okna_podobienstwa));
			MSE = ObliczanieMSE_Kolor(Okienko_Referencyjne, Okienko_porownywane, wielkosc_okna_podobienstwa);
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


//***************************************************************************************************************

void funkcja_glowna(cv::Mat Obrazek, cv::Mat& Obrazek_odszumiony, int kolor_obrazka, float sigma, float stala_h, char filtracja_wstepna)
{
	cv::Mat MacierzWyjsciowa; // obiekt opencv Mat w którym będzie zapisane piksele po odszumieniu
	cv::Mat Okno_Przeszukania; //obiekt opencv Mat który będzie stanowił "wskaznik" na obszar obrazka uwzględniany przy obliczaniu piksela odszumionego
	cv::Mat Okienko_Referencyjne; //obiekt opencv Mat który będzie stanowił "wskaznik" na obszar obrazka uwzględniany przy obliczaniu podobienstwa między dwoam pikselami
	std::string nazwa_pilku_zaszumionego; //nazwa wczytywanego pliku wraz z rozszerzeniem, bez sciezki;
	int orginal_obrazek_width; // szerokosc obrazka orginalnego, bez dodanych marginesów
	int orginal_obrazek_height; // wysokosc obrazka orginalnego, bez dodanych marginesów
	int wielkosc_okna_podobienstwa; // wielkosc obszaru obrazka uwzględniany przy obliczaniu podobienstwa między dwoma pikselami
	int wielkosc_okna_przeszukania; //wielkosc obsaru obrazka uwzględnianego przy obliczaniu piksela odszumionego, 21 alblo 35 pixeli
	int polowa_okna_przeszukania; //długośc obszaru pomiędzy pikselem odszymianym a granicą okna, (wielkosc_okna_przeszukania-1)/2 
	int polowa_okna_podobienstwa; //długośc obszaru pomiędzy pikselem odszymianym a granicą okna, (wielkosc_okna_podobienstwa-1)/2
	int rzeczywiste_okno_przeszukania; //wielkość okna podobienstwa poszerzonegopo obu stronach o połowy okna przeszukania
	int polowa_rzeczywistego_okna_przeszukania;// długośc obszaru pomiędzy pikselem odszymianym a granicą okna, (rzeczywiste_okna_przeszukania - 1) / 2
	int wielkosc_marginesu;// margines dodawany do obrazka z lewej strony i od góry w celu odszumienia pikseli przy brzegach obrazka 
	int obrazek_z_marginesami_width;// szerokosc obrazka po dodaniu marginesów
	int obrazek_z_marginesami_height; // wysokość pobrazka po dodaniu marginesów
	double poziom_rozmycia_gauss; // poziom rozmycia fitrem gaussa (w przypadku zastosowania prefiltracji) zależny od poziomu szumu
	float redukcja_sigmy; // współczynnik redukcji wspłczynnika szumu podczas dalszego przetwarzania, po zastosowaniu prefiltracjii gaussem
	float wartosc_pixela_odszumionego;// tymczasowe miejsce zapisu wartości odszumionego piksela w obrazku szarym
	Vec3f wartosc_pixela_odszumionego_kolor; // tymczasowe miejsce zapisu wartości odszumionego piksela w obrazku kolorowym
	

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

	if (kolor_obrazka == 0)
	{
		if (sigma < 20)
		{
			wielkosc_okna_podobienstwa = 3;
		}
		else if (sigma < 40)
		{
			wielkosc_okna_podobienstwa = 5;
		}
		else if (sigma < 60)
		{
			wielkosc_okna_podobienstwa = 7;
		}
		else
		{
			wielkosc_okna_podobienstwa = 9;
		}
	}
	else if (kolor_obrazka == 1)
	{
		if (sigma < 20)
		{
			wielkosc_okna_podobienstwa = 3;
		}
		else if (sigma < 40)
		{
			wielkosc_okna_podobienstwa = 5;
		}
		else
		{
			wielkosc_okna_podobienstwa = 7;
		}
	}

	orginal_obrazek_width = Obrazek.cols;
	orginal_obrazek_height = Obrazek.rows;

	////////////////dodajemy "marginesy" do obrazka (po połowie obszaru przeszukania -po16)/////////////////////////////

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

	

	if (kolor_obrazka == 0)
	{
		MacierzWyjsciowa = cv::Mat::zeros(obrazek_z_marginesami_height, obrazek_z_marginesami_width, CV_32F);
	}
	else MacierzWyjsciowa = cv::Mat::zeros(obrazek_z_marginesami_height, obrazek_z_marginesami_width, CV_32FC3);

	//if (sigma > 60) sigma = sigma - 5;
	for (int i = wielkosc_marginesu; i < obrazek_z_marginesami_width - wielkosc_marginesu; i++)
	{
		for (int j = wielkosc_marginesu; j < obrazek_z_marginesami_height - wielkosc_marginesu; j++)
		{
			if (kolor_obrazka == 0)
			{
				Okno_Przeszukania = Obrazek(cv::Rect(i - polowa_rzeczywistego_okna_przeszukania, j - polowa_rzeczywistego_okna_przeszukania, rzeczywiste_okno_przeszukania, rzeczywiste_okno_przeszukania));
				Okienko_Referencyjne = Obrazek(cv::Rect(i - polowa_okna_podobienstwa, j - polowa_okna_podobienstwa, wielkosc_okna_podobienstwa, wielkosc_okna_podobienstwa));

				NLM_Grey(Okno_Przeszukania, Okienko_Referencyjne, wielkosc_okna_przeszukania, wielkosc_okna_podobienstwa, wartosc_pixela_odszumionego, sigma, stala_h, polowa_okna_przeszukania, polowa_okna_podobienstwa, true);
				MacierzWyjsciowa.at<float>(j, i) = wartosc_pixela_odszumionego;
			}
			else
			{
				Okno_Przeszukania = Obrazek(cv::Rect(i - polowa_rzeczywistego_okna_przeszukania, j - polowa_rzeczywistego_okna_przeszukania, rzeczywiste_okno_przeszukania, rzeczywiste_okno_przeszukania));
				Okienko_Referencyjne = Obrazek(cv::Rect(i - polowa_okna_podobienstwa, j - polowa_okna_podobienstwa, wielkosc_okna_podobienstwa, wielkosc_okna_podobienstwa));

				NLM_Kolor(Okno_Przeszukania, Okienko_Referencyjne, wielkosc_okna_przeszukania, wielkosc_okna_podobienstwa, wartosc_pixela_odszumionego_kolor, sigma, stala_h, polowa_okna_przeszukania, polowa_okna_podobienstwa, true);
				Vec3f& pixel = MacierzWyjsciowa.at<Vec3f>(j, i);
				pixel = wartosc_pixela_odszumionego_kolor;
			}
		}
	}


	if (kolor_obrazka == 0)
	{
		MacierzWyjsciowa.convertTo(MacierzWyjsciowa, CV_8UC1);
	}
	else MacierzWyjsciowa.convertTo(MacierzWyjsciowa, CV_8UC3);

	Obrazek_odszumiony = MacierzWyjsciowa(cv::Rect(wielkosc_marginesu, wielkosc_marginesu, orginal_obrazek_width, orginal_obrazek_height));

}


//******************************************************************************************************************


void dodanie_szumu(cv::Mat obrazek_zaszumiony, double sigm, int ilosc_kanalow)
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



int main(int argc, char* argv[])
{

	int kolor_obrazka = 1; // zawiera informację czy przetwarzany obraz będzie w skali szarości czy kolorze
	//int opcja_obrazka; // zawiera informację czy przetwarzany obraz jest już zaszumiiony czy dodać szum
	float sigma; // poziom szumu
	float stala_h; // parametr określany przed procesem odszumiania w zależnosci od szumu i wielkosci  Okienka referncyjnego, uzywany w procesie obiczania wagi piksela
	bool Bodues = false; //czy odejmować 2*sigma*sigma od obliczoej odlegości pomiędzy pikselami
	cv::Mat Obrazek; // obiekt opencv Mat w którym będzie zapisany przetwarzany obrazek
	cv::Mat Obrazek_odszumiony;
	string wpisana_nazwa;
	//cv::Mat ObrazekReferencyjny; //obiekt opencv Mat w którym będzie wczytany obrazek bez szumu
	kolor_obrazka = 0;
	//opcja_obrazka = 1;
	//Obrazek  = cv::imread(nazwa_pilku_zaszumionego, kolor_obrazka);
	sigma = 0;
	char filtracja_wstepna = 't';
	Bodues = true;
	stala_h=1;
	int licznik = 0;
	

	if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
	{
		std::cout << "Filtr NLM CPU, filtruje z szumu obrazy w skali szarosci i kolorze.\n";
		std::cout << "Uzycie: NLM_CPU.exe <liczba calkowita>  <int>  <liczba calkowita> <liczba calkowita>\n";
		std::cout << "Argumenty:\n";
		std::cout << "  <nazwa pliku>       Nazwa pliku. Mozna podac nazwe i sciezke folderu lub sama nazwe \n";
		std::cout << "		      jezeli znajduje sie w jednym folderze z programem - zostana \n";
		std::cout << "		      przetworzone wszystkie pliki graficzne w folderze\n";
		std::cout << "  <kolor>	      Liczba calkowita: 0 - obraz w skali szarosci, 1 - obraz w koolorze\n";
		std::cout << "  <poziom szumu>      Liczba calkowita: 0 do 100\n";
		std::cout << "  <stala filtracji>   Liczba calkowita: sila odzumiania";
		return 0;
	}

	if (argc != 5) {
		std::cerr << "Użycie: " << argv[0] << " <nazwa pliku> <kolor> <poziom szumu> <stala filtracji>\n pomoc: --help lub -h";
		cv::waitKey(0);
		return 1;
	}

	wpisana_nazwa = argv[1];       //nazwa wczytywanego pliku argv[0] to nazwa programu
	kolor_obrazka = std::atoi(argv[2]);
	sigma = std::atof(argv[3]);  // drugi arg - poziom szumu
	stala_h = std::atof(argv[4]); // Argument float
	time_t czasStart = time(NULL);

	if (std::filesystem::is_regular_file(wpisana_nazwa))
	{
		Obrazek = cv::imread(wpisana_nazwa, kolor_obrazka);

		if (Obrazek.empty())
		{
			std::cerr << "Nie mozna wczytac obrazka do odszumienia." << std::endl;
			cv::waitKey(0);
			return -1;
		}
		funkcja_glowna(Obrazek, Obrazek_odszumiony, kolor_obrazka, sigma, stala_h, filtracja_wstepna);
		//std::filesystem::path plik = std::filesystem::path(wpisana_nazwa);
		for (std::filesystem::path plik : {std::filesystem::absolute(wpisana_nazwa)})
		{
			std::string nowa_nazwa = plik.stem().string() + "_filtered" + plik.extension().string();
			std::string nowa_nazwa_i_sciezka = plik.parent_path().string() + "/" + nowa_nazwa;
			std::cout << plik.parent_path().string() << endl;
			std::cout << nowa_nazwa_i_sciezka << endl;
			// Zapis przetworzonego obrazu
			//imshow("Obrazek po 2 kroku", Obrazek_odszumiony);
			cv::imwrite(nowa_nazwa_i_sciezka, Obrazek_odszumiony);
		}
		licznik++;
	}
	else if (std::filesystem::is_directory(wpisana_nazwa)) 
	{
		for (const auto& entry : std::filesystem::directory_iterator(wpisana_nazwa)) {
			// Sprawdzenie, czy plik ma odpowiednie rozszerzenie (.jpg, .png itd.)
			if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg" || entry.path().extension() == ".png"
				|| entry.path().extension() == ".bmp" || entry.path().extension() == ".tiff" || entry.path().extension() == ".tif" || entry.path().extension() == ".webp"
				|| entry.path().extension() == ".hdr" || entry.path().extension() == ".jp2"))
			{
				Obrazek = cv::imread(entry.path().string(), kolor_obrazka);

				if (Obrazek.empty())
				{
					std::cerr << "Nie mozna wczytac obrazka do odszumienia." << std::endl;
					cv::waitKey(0);
					return -1;
				}
				funkcja_glowna(Obrazek, Obrazek_odszumiony, kolor_obrazka, sigma, stala_h, filtracja_wstepna);
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
					cv::imwrite(nowa_nazwa_i_sciezka, Obrazek_odszumiony);
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
	std::cout << "Przefiltrowano " << licznik << " obrazow w czasie : " << czasStop - czasStart << " s." << std::endl;

	cv::waitKey(0);
	return 0;

}
