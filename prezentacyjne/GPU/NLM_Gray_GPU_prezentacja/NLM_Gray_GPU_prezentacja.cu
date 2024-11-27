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
    //int rzeczywisty_obszar_przeszukania = 37;

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
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;


    // get number of SMs on this GPU
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


int main(int argc, char** argv)
{

    
    int devID = 0;
    initializeCUDA(argc, argv, devID);
    float sigma;
    int opcja_obrazka;
    cv::Mat ObrazekSzary;
    cv::Mat ObrazekReferencyjny;
    cv::Mat NLM_OpenC;
    std::string nazwa_pilku_zaszumionego;
    std::string nazwa_pilku_referencyjnego;
    std::string nazwa_sciezki = "obrazki_testowe/";
    int szerokosc_obrazka_oryginalnego;
    int wysokosc_obrazka_oryginalnego;
    int wielkosc_okna_podobienstwa;
    int wielkosc_okna_przeszukania;
    float stala_h;    
    int wielkosc_marginesu;
    int szerokosc_obrazka_z_marginesami;
    int wysokosc_obrazka_z_marginesami;
    int wielkosc_tablicy_z_marginesami;
    int ilosc_blokow_w_boku_x;
    int ilosc_blokow_w_boku_y;
    char filtracja_wstepna;
    float poziom_rozmycia_gauss;
    float redukcja_sigmy;
    bool Bodues;
    
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
        ObrazekSzary = cv::imread(nazwa_pilku_zaszumionego, cv::IMREAD_GRAYSCALE);
        ObrazekReferencyjny = cv::imread(nazwa_pilku_referencyjnego, cv::IMREAD_GRAYSCALE);
        if (ObrazekSzary.empty())
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
        //cv::imshow("Obrazek zaszumiony", ObrazekSzary);
        //cv::imshow("referncyjny", ObrazekReferencyjny);

        std::cout << "prosze podaj poziom szumu" << std::endl;
        std::cin >> sigma;
    }

    else if (opcja_obrazka == 2)
    {
        std::cout << "prosze wpisac nazwe pliku obrazka referencyjnego wraz z rozszerzeniem" << std::endl;
        std::cin >> nazwa_pilku_referencyjnego;
        //nazwa_pilku_referencyjnego = nazwa_sciezki + nazwa_pilku_referencyjnego;
        ObrazekReferencyjny = cv::imread(nazwa_pilku_referencyjnego, cv::IMREAD_GRAYSCALE);
        if (ObrazekReferencyjny.empty()) 
        {
            std::cout << "Nie mozna wczytac obrazka." << std::endl;
            return -1;
        }
      
        
        //cv::imshow("referencyjny", ObrazekReferencyjny);
        std::cout << "prosze podaj poziom szumu" << std::endl;
        std::cin >> sigma;
        ObrazekSzary = ObrazekReferencyjny.clone(); // Skopiuj obraz do macierzy z szumem
        dodanie_szumu(ObrazekSzary, sigma,1);

        //cv::imshow("Obrazek zaszumiony", ObrazekSzary);
    }
    

    do
    {
        std::cout << "prosze podaj wielkosc okna podobienstwa: 3, 5, 7 lub 9" << std::endl;
        std::cin >> wielkosc_okna_podobienstwa;
    } while ((wielkosc_okna_podobienstwa != 5) && (wielkosc_okna_podobienstwa != 7) && (wielkosc_okna_podobienstwa != 9) && (wielkosc_okna_podobienstwa != 3));

    do
    {
        std::cout << "czy przeprowadzic wstepa filtracje romyciem gaussa?" << std::endl;
        std::cout << "tak, wcisnij t" << std::endl;
        std::cout << "nie, wcisnij n" << std::endl;
        std::cin >> filtracja_wstepna;
    } while ((filtracja_wstepna != 't')&& (filtracja_wstepna != 'n'));

    std::string odpowiedz2;
    do
    {
        std::cout << "czy od sumy dystansow pomiędzy pikselami odjac 2*sigma*sigma?" << std::endl;
        std::cout << "tak, wcisnij t" << std::endl;
        std::cout << "nie, wcisnij n" << std::endl;
        std::cin >> odpowiedz2;
        if (odpowiedz2 == "t")  Bodues = true;
        else Bodues = false;
    } while ((odpowiedz2 != "t") && (odpowiedz2 != "n"));

    std::cout << "prosze podaj wysokosc parametru H" << std::endl;
    std::cin >> stala_h;
  

    if (sigma < 25) {
        wielkosc_okna_przeszukania = 15;
    }
    else if (sigma < 40)
    {
        wielkosc_okna_przeszukania = 21;
    }
    else if (sigma < 70)
    {
        wielkosc_okna_przeszukania = 25;
    }
    else
    {
        wielkosc_okna_przeszukania = 31;
    }

    
    cv::fastNlMeansDenoising(ObrazekSzary, NLM_OpenC, sigma, wielkosc_okna_podobienstwa, wielkosc_okna_przeszukania);
    cv::imshow("funkcja wbudowana NLM", NLM_OpenC);
    
 
    ///////////////////////////////////////////////////////



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

filtracja_wstepna = 't';

Bodues = true;


//////////////////////////////////////////////////////////////


        if (filtracja_wstepna == 't')
        {

            if (sigma < 60)
            {
                poziom_rozmycia_gauss = 0.5;
                cv::GaussianBlur(ObrazekSzary, ObrazekSzary, cv::Size(3, 3), poziom_rozmycia_gauss, poziom_rozmycia_gauss);
                poziom_rozmycia_gauss = 0.5;
                redukcja_sigmy = 0.65;
            }
            else
            {
                redukcja_sigmy = 0.45;
                cv::medianBlur(ObrazekSzary, ObrazekSzary, 3);
            }
            sigma = redukcja_sigmy * sigma;
        }




    /////////////////////////////////////////////////////////////////dodajemy "marginesy" do obrazka (po połowie obszaru przeszukania -po16)////////////////////////////////////////////////////
    szerokosc_obrazka_oryginalnego = ObrazekSzary.cols;
    //std::cout << szerokosc_obrazka_oryginalnego << std::endl;
    wysokosc_obrazka_oryginalnego = ObrazekSzary.rows;
    //std::cout << wysokosc_obrazka_oryginalnego << std::endl;
    wielkosc_okna_podobienstwa;
    wielkosc_marginesu = (wielkosc_okna_przeszukania - 1) / 2 + (wielkosc_okna_podobienstwa - 1) / 2;

    cv::copyMakeBorder(ObrazekSzary, ObrazekSzary, wielkosc_marginesu, wielkosc_marginesu, wielkosc_marginesu, wielkosc_marginesu, cv::BORDER_REFLECT_101);

    ObrazekSzary.convertTo(ObrazekSzary, CV_32F);


    szerokosc_obrazka_z_marginesami = szerokosc_obrazka_oryginalnego + 2*wielkosc_marginesu;
    wysokosc_obrazka_z_marginesami = wysokosc_obrazka_oryginalnego + 2*wielkosc_marginesu;
    wielkosc_tablicy_z_marginesami = szerokosc_obrazka_z_marginesami * wysokosc_obrazka_z_marginesami;
    ObrazekSzary = ObrazekSzary.reshape(1, wielkosc_tablicy_z_marginesami);

    //////////////////////////////////////////////////////////////////tworzymy zmienne do przekazania do Kernelu starowego///////////////////////////
    
    ilosc_blokow_w_boku_x = szerokosc_obrazka_oryginalnego;
    ilosc_blokow_w_boku_y = wysokosc_obrazka_oryginalnego;
    
    float* host_obrazek_poczatkowy;
    host_obrazek_poczatkowy = (float*)ObrazekSzary.data;
    float* host_obrazek_odszumiony = new float[wielkosc_tablicy_z_marginesami];
    
    size_t array_byte_size = (wielkosc_tablicy_z_marginesami) * sizeof(float);
    float* device_obrazek_odszumiony;
    cudaMalloc((void**)&device_obrazek_odszumiony, array_byte_size);
    float* device_obrazek_poczatkowy = new float[wielkosc_tablicy_z_marginesami];
    cudaMalloc((void**)&device_obrazek_poczatkowy, array_byte_size);
    
    ////////////////////////////////////////////przygotowanie i lokowanie w pamięci tablic pomocniczych////////////////////////////////
    /////////lokujemy je w pamięci przed rozpoczęciem wykonywania programu przez kartę gdyż dynamiczne lokowanie pamięci przez CUDĘ wielokrotnie spowalnia program///////////////
    

   ////////////////
    if (sigma >50) sigma = 0.85 *sigma;
    int szerokosc = szerokosc_obrazka_z_marginesami;
    int wielkosc_extern_sh_memory= sizeof(float) * (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa) 
        + sizeof(float) * (wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1) * (wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1)
        + 2*(sizeof(float) * (wielkosc_okna_przeszukania * wielkosc_okna_przeszukania));

    dim3 bloki_NLM(ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, 1);
    dim3 watki_NLM(wielkosc_okna_przeszukania, wielkosc_okna_przeszukania, 1);
    int start = clock();
    cudaMemcpy(device_obrazek_poczatkowy, host_obrazek_poczatkowy, array_byte_size, cudaMemcpyHostToDevice);
    //std::cout << "uruch kernel" << std::endl;
    Device_Non_Local_Means << <bloki_NLM, watki_NLM, wielkosc_extern_sh_memory >> > (device_obrazek_poczatkowy, device_obrazek_odszumiony, wielkosc_okna_podobienstwa, wielkosc_okna_przeszukania, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, wielkosc_marginesu, sigma, stala_h, Bodues);
    //std::cout << "skonczono kernel" << std::endl;
   cudaMemcpy(host_obrazek_odszumiony, device_obrazek_odszumiony, array_byte_size, cudaMemcpyDeviceToHost);  
   cudaDeviceSynchronize();
   int stop = clock();
   cv::Mat testDataMat(wysokosc_obrazka_z_marginesami, szerokosc_obrazka_z_marginesami, CV_32F, host_obrazek_odszumiony);
   testDataMat.convertTo(testDataMat, CV_8U);
   testDataMat = testDataMat(cv::Rect(wielkosc_marginesu, wielkosc_marginesu, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));
   double R = 255;
   //int stop_1 = clock();
   double czas = (double)(stop - start) / (double)CLOCKS_PER_SEC;
   std::cout << std::endl << "Czas wykonania kernelu: " << czas << " s" << std::endl;
   std::cout << "PSNR po odszumieniu zaimplementowanym NLM: " << cv::PSNR(ObrazekReferencyjny, testDataMat, R) << std::endl;
   std::cout << "PSNR po odszymieniu NLM wbudowanym w opencv: " << cv::PSNR(ObrazekReferencyjny, NLM_OpenC, R) << std::endl;

   ObrazekSzary = ObrazekSzary.reshape(0, wysokosc_obrazka_z_marginesami);
   ObrazekSzary.convertTo(ObrazekSzary, CV_8U);
   ObrazekSzary = ObrazekSzary(cv::Rect(wielkosc_marginesu, wielkosc_marginesu, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));

   cv::imshow("Obrazek zaszumiony", ObrazekSzary);
   cv::imshow("Obrazek referncyjny", ObrazekReferencyjny);

   cv::imshow("funkcja wbudowana NLM", NLM_OpenC);
   cv::imshow("Obrazek odszumiony zaimplementowanym NLM", testDataMat);
   cv::waitKey(0);
   cudaFree(device_obrazek_poczatkowy);
   cudaFree(device_obrazek_odszumiony);
   delete[] host_obrazek_odszumiony;
   host_obrazek_odszumiony = nullptr;
   cudaDeviceReset();



   cv::waitKey(0);

  return 0;

}