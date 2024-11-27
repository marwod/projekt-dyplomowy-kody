/*
© Marcin Wodejko 2024.
marwod@interia.pl
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <stdint.h>
#include <string>
#include <time.h>
#include <iostream>
#include <cmath>
#include <conio.h>
#include <math.h>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <assert.h>
#include <random>
#include <cstdlib>
#include <filesystem> // musi byc C++17 lub wyzej



//#define wielkosc_okna_przeszukania      31 //musi byc nieparzysty, 21 do 35 w zalezności od szumu
#define WIELKOSC_OKNA_PODOBIENSTWA       9 //musi byc nieparzysty, zgodnie z publikacjami w przedziale 3 do 11
#define MAX_ODLEGLOSC_PIXELA_SUMOWANEGO  4 // WIELKOSC_OKNA_PODOBIENSTWA/2-1
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 37

#define BLOCK_DIM_X 31
#define BLOCK_DIM_Y 31
#define BLOCK_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)



#include <stdio.h>

#define BLOCK_DIM_X 31
#define BLOCK_DIM_Y 31
#define BLOCK_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)


__global__ void Device_Non_Local_Means(float* __restrict__ device_obrazek_poczatkowy, float* device_obrazek_odszumiony, int wielkosc_okna_podobienstwa, int wielkosc_okna_przeszukania, int ilosc_okien_w_boku_x, int ilosc_okien_w_boku_y, int szerokosc, int wielkosc_marginesu_lewego, float sigma, float stala_h, bool Bodues)
{
    int rzeczywisty_obszar_przeszukania = wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1;
    int row_pos = blockIdx.x*blockDim.x+ threadIdx.x;
    int col_pos = blockIdx.y*blockDim.y+threadIdx.y;

    __shared__ float suma_wag;
    __shared__ float suma_pikseli;

    extern __shared__ float okienko[];
    float* okienko_referencyjne = (float*)&okienko[0];
    float* Okno_preszukana_shared = (float*)&okienko[wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa]; //okno ma wilekosć rzeczywistego obszaru przeszukania

    float* tablica_wartosci_pikseli = (float*)&Okno_preszukana_shared[rzeczywisty_obszar_przeszukania * rzeczywisty_obszar_przeszukania];
    float* tablica_wag_pikseli = (float*)&tablica_wartosci_pikseli[wielkosc_okna_przeszukania * wielkosc_okna_przeszukania];

            suma_wag = 0;
            suma_pikseli = 0;

            int index_x_pixela_gorny_lewy_obszaru_przeszukania = blockIdx.x ;//przetestować czy blo z czy y czy jeden i drugi!!!!!!!
            int index_y_pixela_gorny_lewy_obszaru_przeszukania = blockIdx.y;
            int index_x_pixela_przetwarzanego = blockIdx.x  + wielkosc_marginesu_lewego;
            int index_y_pixela_przetwarzanego = blockIdx.y  + wielkosc_marginesu_lewego;
            int index2dObszaru = threadIdx.x + threadIdx.y * wielkosc_okna_przeszukania;

            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < wielkosc_okna_przeszukania))
            {
                Okno_preszukana_shared[threadIdx.y * rzeczywisty_obszar_przeszukania + threadIdx.x] = (device_obrazek_poczatkowy[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
                //przpisujemy obszar preszukania (37 pixeli) dla łatki do pamięci dzielonej bloku, ze względu na zmieszczenie się w dostępnej w wywołaniu funkcji iosci wątków musiałem zrealizować przypisanie w czterech krokach.
            }
            __syncthreads();
            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar łatki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x] = (device_obrazek_poczatkowy[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            __syncthreads();
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < wielkosc_okna_przeszukania)) //przpisujemy obszar łatki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y)*rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (device_obrazek_poczatkowy[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);;
            }
            __syncthreads();
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar łatki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {
                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (device_obrazek_poczatkowy[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);;
            }
            __syncthreads();

            int ofset = wielkosc_okna_przeszukania / 2;

            if ((threadIdx.x < wielkosc_okna_podobienstwa) && (threadIdx.y < wielkosc_okna_podobienstwa)) //przpisujemy obszar łatki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {
                okienko_referencyjne[threadIdx.y * wielkosc_okna_podobienstwa + threadIdx.x] = Okno_preszukana_shared[(threadIdx.y + ofset) * (rzeczywisty_obszar_przeszukania)+(threadIdx.x + ofset)];

            }
            __syncthreads();

            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < wielkosc_okna_przeszukania))
            {
                float wartosc_piksela = Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_podobienstwa / 2) * (rzeczywisty_obszar_przeszukania)+threadIdx.x + wielkosc_okna_podobienstwa / 2];
                float MSE = 0;
                float waga = 0;
                for (int i = 0; i < wielkosc_okna_podobienstwa; i++)
                {
                    for (int j = 0; j < wielkosc_okna_podobienstwa; j++)
                    {

                        MSE += ((okienko_referencyjne[i * wielkosc_okna_podobienstwa + j] - Okno_preszukana_shared[(threadIdx.y + i) * rzeczywisty_obszar_przeszukania + threadIdx.x + j]) * (okienko_referencyjne[i * wielkosc_okna_podobienstwa + j] - Okno_preszukana_shared[(threadIdx.y + i) * rzeczywisty_obszar_przeszukania + threadIdx.x + j]));
                        __syncthreads();
                    }
                }
                __syncthreads();
                waga = MSE / (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa);
                
                if (Bodues == true)
                {
                    waga = MAX((waga - 2 * sigma * sigma), 0.0);
                }
                
                __syncthreads();

                waga = exp(-(waga / (stala_h * stala_h)));
                tablica_wag_pikseli[index2dObszaru] = waga;
                tablica_wartosci_pikseli[index2dObszaru] = waga * wartosc_piksela;
            }
            __syncthreads();

            if ((threadIdx.x < (wielkosc_okna_przeszukania)) && (threadIdx.y < (wielkosc_okna_przeszukania)))
            {
                
                 if (threadIdx.x == 0 )

                 {
                     float suma1;
                     float suma2;
                     suma1 = 0;
                     suma2 = 0;
                     for (int i = 0; i < wielkosc_okna_przeszukania; i++)
                     {
                         suma1 = suma1 + tablica_wag_pikseli[threadIdx.x + i + (threadIdx.y) * wielkosc_okna_przeszukania];
                         suma2 = suma2 + tablica_wartosci_pikseli[threadIdx.x + i + (threadIdx.y) * wielkosc_okna_przeszukania];
                     }
                     tablica_wag_pikseli[threadIdx.x + threadIdx.y * wielkosc_okna_przeszukania] = suma1;
                     tablica_wartosci_pikseli[threadIdx.x + threadIdx.y * wielkosc_okna_przeszukania] = suma2;
                 }
             }
             __syncthreads();

             if (threadIdx.y == 0 && threadIdx.x == 0)

             {
                 for (int i = 0; i < wielkosc_okna_przeszukania; i++)
                 {
                     suma_wag = suma_wag + tablica_wag_pikseli[threadIdx.x + (threadIdx.y + i) * wielkosc_okna_przeszukania];
                     suma_pikseli = suma_pikseli + tablica_wartosci_pikseli[threadIdx.x + (threadIdx.y + i) * wielkosc_okna_przeszukania];
                 }
             }
             
               
         
            if ((threadIdx.x ==0) && (threadIdx.y == 0))
            {
                device_obrazek_odszumiony[index_y_pixela_przetwarzanego * szerokosc + index_x_pixela_przetwarzanego] = suma_pikseli / suma_wag;
           
            }
            __syncthreads();
}


void initializeCUDA(int argc, char** argv, int& devID )
{
     cudaError_t error;
    devID = 0;

    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
 

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    
}

void dodanie_szumu(cv::Mat obrazek_zaszumiony, float sigm, int ilosc_kanalow)
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

void funkcja_glowna(cv::Mat Obrazek, cv::Mat& Obrazek_odszumiony, float sigma, float stala_h, char filtracja_wstepna, bool Bodues)
{
    int szerokosc_obrazka_oryginalnego;
    int wysokosc_obrazka_oryginalnego;
    int wielkosc_okna_podobienstwa;
    int wielkosc_okna_przeszukania; int wielkosc_marginesu;
    int szerokosc_obrazka_z_marginesami;
    int wysokosc_obrazka_z_marginesami;
    int wielkosc_tablicy_z_marginesami;
    int ilosc_blokow_w_boku_x;
    int ilosc_blokow_w_boku_y;
    float poziom_rozmycia_gauss;
    float redukcja_sigmy;

    if (sigma < 25) 
    {
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


    if (filtracja_wstepna == 't')
    {

         if (sigma < 60)
         {
            poziom_rozmycia_gauss = 0.5;
            cv::GaussianBlur(Obrazek, Obrazek, cv::Size(3, 3), poziom_rozmycia_gauss, poziom_rozmycia_gauss);
            poziom_rozmycia_gauss = 0.5;
            redukcja_sigmy = 0.65;
         }
         else
         {
            redukcja_sigmy = 0.45;
            cv::medianBlur(Obrazek, Obrazek, 3);
         }
         sigma = redukcja_sigmy * sigma;
    }


    /////////////////////////////////////////////////////////////////dodajemy "marginesy" do obrazka (po połowie obszaru przeszukania -po16)////////////////////////////////////////////////////
   szerokosc_obrazka_oryginalnego = Obrazek.cols;
   wysokosc_obrazka_oryginalnego = Obrazek.rows;
   wielkosc_marginesu = (wielkosc_okna_przeszukania - 1) / 2 + (wielkosc_okna_podobienstwa - 1) / 2;

   cv::copyMakeBorder(Obrazek, Obrazek, wielkosc_marginesu, wielkosc_marginesu, wielkosc_marginesu, wielkosc_marginesu, cv::BORDER_REFLECT_101);

   Obrazek.convertTo(Obrazek, CV_32F);

   szerokosc_obrazka_z_marginesami = szerokosc_obrazka_oryginalnego + 2 * wielkosc_marginesu;
   wysokosc_obrazka_z_marginesami = wysokosc_obrazka_oryginalnego + 2 * wielkosc_marginesu;
   wielkosc_tablicy_z_marginesami = szerokosc_obrazka_z_marginesami * wysokosc_obrazka_z_marginesami;

   Obrazek = Obrazek.reshape(1, wielkosc_tablicy_z_marginesami);
//////////////////////////////////////////////////////////////////tworzymy zmienne do przekazania do Kernelu starowego///////////////////////////
   float* host_obrazek_odszumiony = new float[wielkosc_tablicy_z_marginesami];

   ilosc_blokow_w_boku_x = szerokosc_obrazka_oryginalnego;
   ilosc_blokow_w_boku_y = wysokosc_obrazka_oryginalnego;

   float* host_obrazek_poczatkowy = (float*)Obrazek.data;

   size_t array_byte_size = (wielkosc_tablicy_z_marginesami) * sizeof(float);
   float* device_obrazek_odszumiony;
   cudaMalloc((void**)&device_obrazek_odszumiony, array_byte_size);
   float* device_obrazek_poczatkowy ;
   cudaMalloc((void**)&device_obrazek_poczatkowy, array_byte_size);
   cudaMemcpy(device_obrazek_poczatkowy, host_obrazek_poczatkowy, array_byte_size, cudaMemcpyHostToDevice);

////////////////////////////////////////////przygotowanie i lokowanie w pamięci tablic pomocniczych////////////////////////////////
/////////lokujemy je w pamięci przed rozpoczęciem wykonywania programu przez kartę gdyż dynamiczne lokowanie pamięci przez CUDĘ wielokrotnie spowalnia program///////////////

   if (sigma > 50) sigma = 0.9 * sigma;
   if (stala_h == 0) stala_h = 0.1;
   int szerokosc = szerokosc_obrazka_z_marginesami;
   int wielkosc_extern_sh_memory = sizeof(float) * (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa) + sizeof(float) * (wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1) * (wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1) + 2 * (sizeof(float) * (wielkosc_okna_przeszukania * wielkosc_okna_przeszukania));

   dim3 bloki_NLM(ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, 1);
   dim3 watki_NLM(wielkosc_okna_przeszukania, wielkosc_okna_przeszukania, 1);
   Device_Non_Local_Means << <bloki_NLM, watki_NLM, wielkosc_extern_sh_memory >> > (device_obrazek_poczatkowy, device_obrazek_odszumiony, wielkosc_okna_podobienstwa, wielkosc_okna_przeszukania, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, wielkosc_marginesu, sigma, stala_h, Bodues);
   cudaMemcpy(host_obrazek_odszumiony, device_obrazek_odszumiony, array_byte_size, cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();

   cv::Mat testDataMat(wysokosc_obrazka_z_marginesami, szerokosc_obrazka_z_marginesami, CV_32F, host_obrazek_odszumiony);
   testDataMat.convertTo(testDataMat, CV_8U);
   Obrazek_odszumiony =testDataMat;
   Obrazek_odszumiony = Obrazek_odszumiony(cv::Rect(wielkosc_marginesu, wielkosc_marginesu, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));

   cudaFree(device_obrazek_poczatkowy);
   cudaFree(device_obrazek_odszumiony);
   delete[] host_obrazek_odszumiony;
   host_obrazek_odszumiony = nullptr;
}
int main(int argc, char* argv[])
{

    float sigma; // poziom szumu
    float stala_h; // parametr określany przed procesem odszumiania w zależnosci od szumu i wielkosci  Okienka referncyjnego, uzywany w procesie obiczania wagi piksela
    bool Bodues = false; //czy odejmować 2*sigma*sigma od obliczoej odlegości pomiędzy pikselami
    cv::Mat Obrazek; // obiekt opencv Mat w którym będzie zapisany przetwarzany obrazek
    cv::Mat Obrazek_odszumiony;
    std::string wpisana_nazwa;
    sigma = 0;
    char filtracja_wstepna = 't';
    Bodues = true;
    stala_h = 1;
    int licznik = 0;

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
    {
        std::cout << "Filtr NLM GPU, filtruje z szumu obrazy w skali szarosci.\n";
        std::cout << "Uzycie: NLM_Gray_GPU.exe <liczba calkowita>  <liczba calkowita lub zmiennoprzecinkowa>\n ";
        std::cout << "Argumenty:\n";
        std::cout << "  <nazwa pliku>         Nazwa pliku. Mozna podac nazwe i sciezke folderu lub sama nazwe \n";
        std::cout << "                        jezeli znajduje sie w jednym folderze z programem\n";
        std::cout << "                        -zostaną przetworzone wszystkie pliki graficzne w folderze\n";
        std::cout << "  <poziom szumu>        Liczba calkowita: 0 do 100\n";
        std::cout << "  <stala filtracji>	Liczba calkowita: sila odzumiania";
        return 0;
    }

    if (argc != 4) {
        std::cerr << "Użycie: " << argv[0] << " <nazwa pliku>  <poziom szumu> <stala filtracji>\n pomoc: --help lub -h";
        cv::waitKey(0);
        return 1;
    }

    wpisana_nazwa = argv[1];       //nazwa wczytywanego pliku argv[0] to nazwa programu
    sigma = std::atof(argv[2]);  // drugi arg - poziom szumu
    stala_h = std::atof(argv[3]); // Argument float

    
    int devID = 0;
    initializeCUDA(argc, argv, devID);
    

    time_t czasStart = clock();

    if (std::filesystem::is_regular_file(wpisana_nazwa))
    {
        Obrazek = cv::imread(wpisana_nazwa, cv::IMREAD_GRAYSCALE);

        if (Obrazek.empty())
        {
            std::cerr << "Nie mozna wczytac obrazka do odszumienia." << std::endl;
            cv::waitKey(0);
            return -1;
        }
        funkcja_glowna(Obrazek, Obrazek_odszumiony, sigma, stala_h, filtracja_wstepna, Bodues);
        for (std::filesystem::path plik : {std::filesystem::absolute(wpisana_nazwa)})
        {
            std::string nowa_nazwa = plik.stem().string() + "_filtered" + plik.extension().string();
            std::string nowa_nazwa_i_sciezka = plik.parent_path().string() + "/" + nowa_nazwa;
            std::cout << plik.parent_path().string() << std::endl;
            std::cout << nowa_nazwa_i_sciezka << std::endl;
            // Zapis przetworzonego obrazu
            cv::imshow("Obrazek po 2 kroku", Obrazek_odszumiony);
            cv::imwrite(nowa_nazwa_i_sciezka, Obrazek_odszumiony);
        }
        licznik++;
    }
    else if (std::filesystem::is_directory(wpisana_nazwa)) {
        //std::cout << "Przetwarzanie folderu: " << inputPath << std::endl;
        for (const auto& entry : std::filesystem::directory_iterator(wpisana_nazwa)) {
            // Sprawdzenie, czy plik ma odpowiednie rozszerzenie (.jpg, .png itd.)
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
                funkcja_glowna(Obrazek, Obrazek_odszumiony, sigma, stala_h, filtracja_wstepna, Bodues);
                for (std::filesystem::path plik : {std::filesystem::absolute(std::filesystem::path(entry))})
                {
                    //std::cerr << "przetwarzam sekwencje" << std::endl;
                    std::string nowa_nazwa = plik.stem().string() + "_filtered" + plik.extension().string();
                    std::string nowa_sciezka = plik.parent_path().string() + "/" + "filtered";
                    std::string nowa_nazwa_i_sciezka = nowa_sciezka + "/" + nowa_nazwa;
                    //std::cerr << "rozszerzenie:" << plik.extension().string() << std::endl;
                    //std::cerr << "sciezka" << nowa_nazwa_i_sciezka << std::endl;
                    // Zapis przetworzonego obrazu
                    //cv::imshow("Obrazek po 2 kroku", Obrazek_odszumiony);
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
    time_t czasStop = clock();
    //time_t czasStop = time(NULL);
    //clock()
    //double czas = (double)(czasStop - czasStart);
    double czas = (double)(czasStop - czasStart) / (double)CLOCKS_PER_SEC;
    std::cout << "Przefiltrowano " << licznik << " obrazow w czasie : " << czas << " s." << std::endl;


    cv::waitKey(0);

    cudaDeviceReset();

    return 0;


//////////////////////////////////////////////////////////////


        



   cv::waitKey(0);

  return 0;

}