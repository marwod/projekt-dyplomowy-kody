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
#include <math.h>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <assert.h>
#include <random>
#include <cstdlib>
#include <filesystem> // musi byc C++17 lub wyzej


#define MAX_WIELKOSC_OKNA_PRZESZUKANIA  31   //musi byc nieparzysty, 21 do 35 w zaleznosci od szumu


struct Obrazek_RGB
{
    float* kanal_R;
    float* kanal_G;
    float* kanal_B;
};

__global__ void Device_Non_Local_Means(Obrazek_RGB Dev_Macierz_wejsciowa, Obrazek_RGB Dev_Macierz_odszumiona, int wielkosc_okna_podobienstwa, int wielkosc_okna_przeszukania, int szerokosc, int wielkosc_marginesu, float sigma, float Stala_h, bool Bodues)
{

    int rzeczywisty_obszar_przeszukania = wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1;

    __shared__ float suma_wag_R;
    __shared__ float suma_pikseli_R;
    __shared__ float suma_pikseli_G;
    __shared__ float suma_pikseli_B;

    extern __shared__ float okienko[];
    float* okienko_referencyjne = (float*)&okienko[0];
    float* Okno_preszukana_shared = (float*)&okienko_referencyjne
        [wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa];
    //Niestety nie miesci sie przy polach wiekszych niz 25*25 w extern shared memory i musze ustawić ja na sztywno jako stala zeby okno przeszukania moglo mieć 31 na 31
    __shared__ float tablica_wartosci_pikseli_R[MAX_WIELKOSC_OKNA_PRZESZUKANIA * MAX_WIELKOSC_OKNA_PRZESZUKANIA];
    __shared__ float tablica_wag_pikseli_R[MAX_WIELKOSC_OKNA_PRZESZUKANIA * MAX_WIELKOSC_OKNA_PRZESZUKANIA];
    __shared__ float tablica_wartosci_pikseli_G[MAX_WIELKOSC_OKNA_PRZESZUKANIA * MAX_WIELKOSC_OKNA_PRZESZUKANIA];
    __shared__ float tablica_wag_pikseli_G[MAX_WIELKOSC_OKNA_PRZESZUKANIA * MAX_WIELKOSC_OKNA_PRZESZUKANIA];
    __shared__ float tablica_wartosci_pikseli_B[MAX_WIELKOSC_OKNA_PRZESZUKANIA * MAX_WIELKOSC_OKNA_PRZESZUKANIA];
    __shared__ float tablica_wag_pikseli_B[MAX_WIELKOSC_OKNA_PRZESZUKANIA * MAX_WIELKOSC_OKNA_PRZESZUKANIA];

            suma_wag_R = 0;
            suma_pikseli_R = 0;
            suma_pikseli_G = 0;
            suma_pikseli_B = 0;
            int index_x_pixela_gorny_lewy_obszaru_przeszukania = blockIdx.x;
            int index_y_pixela_gorny_lewy_obszaru_przeszukania = blockIdx.y;
            int index_x_pixela_przetwarzanego = blockIdx.x + wielkosc_marginesu;
            int index_y_pixela_przetwarzanego = blockIdx.y + wielkosc_marginesu;
            int index2dObszaru = threadIdx.x + threadIdx.y * wielkosc_okna_przeszukania;

            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < wielkosc_okna_przeszukania))
            {
                Okno_preszukana_shared[threadIdx.y * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_R[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)))
            {
                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_R[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < wielkosc_okna_przeszukania)) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci wspoldzielonej bloku
            {
                Okno_preszukana_shared[(threadIdx.y) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_R[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }        
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) 
            {
                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_R[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            __syncthreads();

            int offset = wielkosc_okna_przeszukania / 2;
            if ((threadIdx.x < wielkosc_okna_podobienstwa) && (threadIdx.y < wielkosc_okna_podobienstwa))
            {
                okienko_referencyjne[threadIdx.y * wielkosc_okna_podobienstwa + threadIdx.x] = Okno_preszukana_shared[(threadIdx.y + offset) * (rzeczywisty_obszar_przeszukania)+(threadIdx.x + offset)];
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
                        MSE += ((okienko_referencyjne[i * wielkosc_okna_podobienstwa + j] 
                           - Okno_preszukana_shared[(threadIdx.y + i) * rzeczywisty_obszar_przeszukania  + threadIdx.x + j]) * (okienko_referencyjne[i * wielkosc_okna_podobienstwa 
                           + j] - Okno_preszukana_shared[(threadIdx.y + i) * rzeczywisty_obszar_przeszukania + threadIdx.x + j]));
                        __syncthreads();
                    }
                }
                __syncthreads();
                tablica_wag_pikseli_R[index2dObszaru] = MSE / (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa);
                tablica_wartosci_pikseli_R[index2dObszaru] = wartosc_piksela;
            }

            //2 dla skladowej G/////////////////////////////////////////////////////////////

            {
                Okno_preszukana_shared[threadIdx.y * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_G[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
                //przpisujemy obszar preszukania (37 pixeli) dla latki do pamieci dzielonej bloku, ze wzgledu na zmieszczenie sie w dostepnej w wywolaniu funkcji iosci watkow musialem zrealizować przypisanie w czterech krokach.
            }
            //__syncthreads();
            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_G[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < wielkosc_okna_przeszukania)) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_G[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci dzielonej bloku
            {
                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_G[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            __syncthreads();

            if ((threadIdx.x < wielkosc_okna_podobienstwa) && (threadIdx.y < wielkosc_okna_podobienstwa)) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci dzielonej bloku
            {
                okienko_referencyjne[threadIdx.y * wielkosc_okna_podobienstwa + threadIdx.x] = Okno_preszukana_shared[(threadIdx.y + offset) * (rzeczywisty_obszar_przeszukania)+(threadIdx.x + offset)];
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

                tablica_wag_pikseli_G[index2dObszaru] = MSE / (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa);
                tablica_wartosci_pikseli_G[index2dObszaru] = wartosc_piksela;
            }

            //3 dla skladowej B/////////////////////////////////////////////////////////////

            if ((threadIdx.y < wielkosc_okna_przeszukania) && (threadIdx.x < wielkosc_okna_przeszukania))

                //1 dla skladowej B/////////////////////////////////////////////////////////////
            {
                Okno_preszukana_shared[threadIdx.y * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_B[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
                //przpisujemy obszar preszukania (37 pixeli) dla latki do pamieci dzielonej bloku, ze wzgledu na zmieszczenie sie w dostepnej w wywolaniu funkcji iosci watkow musialem zrealizować przypisanie w czterech krokach.
            }
            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_B[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }

            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < wielkosc_okna_przeszukania)) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_B[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci dzielonej bloku
            {
                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_B[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            __syncthreads();


            if ((threadIdx.x < wielkosc_okna_podobienstwa) && (threadIdx.y < wielkosc_okna_podobienstwa)) //przpisujemy obszar latki do ktorej bedziemy porownywać do pamieci dzielonej bloku
            {
                okienko_referencyjne[threadIdx.y * wielkosc_okna_podobienstwa + threadIdx.x] = Okno_preszukana_shared[(threadIdx.y + offset) * (rzeczywisty_obszar_przeszukania)+(threadIdx.x + offset)];
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
                tablica_wag_pikseli_B[index2dObszaru] = MSE / (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa);
                tablica_wartosci_pikseli_B[index2dObszaru] = wartosc_piksela;
            }
            __syncthreads();

            if ((threadIdx.y < (wielkosc_okna_przeszukania)) && (threadIdx.x < (wielkosc_okna_przeszukania)))
            {
                tablica_wag_pikseli_R[index2dObszaru] = (tablica_wag_pikseli_R[index2dObszaru] + tablica_wag_pikseli_G[index2dObszaru] + tablica_wag_pikseli_B[index2dObszaru]) / 3;
                __syncthreads();
                
                if (Bodues == true)
                {
                    
                    tablica_wag_pikseli_R[index2dObszaru] = fmaxf((tablica_wag_pikseli_R[index2dObszaru] - (2 * sigma * sigma)), 0);
                }
             
                __syncthreads();
                tablica_wag_pikseli_R[index2dObszaru] = (expf(-(tablica_wag_pikseli_R[index2dObszaru] / (Stala_h * Stala_h))));
                
            }
           
            __syncthreads();

            if ((threadIdx.x < (wielkosc_okna_przeszukania)) && (threadIdx.y < (wielkosc_okna_przeszukania)))
            {
                tablica_wartosci_pikseli_R[index2dObszaru] = tablica_wartosci_pikseli_R[index2dObszaru] * tablica_wag_pikseli_R[index2dObszaru];
                tablica_wartosci_pikseli_G[index2dObszaru] = tablica_wartosci_pikseli_G[index2dObszaru] * tablica_wag_pikseli_R[index2dObszaru];
                tablica_wartosci_pikseli_B[index2dObszaru] = tablica_wartosci_pikseli_B[index2dObszaru] * tablica_wag_pikseli_R[index2dObszaru];
            }
            __syncthreads();

            if ((threadIdx.y < (wielkosc_okna_przeszukania)) && (threadIdx.x < (wielkosc_okna_przeszukania)))
            {
                if (threadIdx.x == 0)
                {
                    float suma_wag = 0;
                    for (int i = 0; i < wielkosc_okna_przeszukania; i++)
                    {
                        suma_wag = suma_wag + tablica_wag_pikseli_R[threadIdx.x + i + threadIdx.y * wielkosc_okna_przeszukania];
                    }
                    tablica_wag_pikseli_R[threadIdx.x + threadIdx.y * wielkosc_okna_przeszukania] = suma_wag;
                }

                __syncthreads();

                if (threadIdx.x == 0)
                {
                    float suma_R = 0;
                    float suma_G = 0;
                    float suma_B = 0;
                    for (int i = 0; i < wielkosc_okna_przeszukania; i++)
                    {
                        suma_R = suma_R + tablica_wartosci_pikseli_R[threadIdx.x + i + threadIdx.y * wielkosc_okna_przeszukania];
                        suma_G = suma_G + tablica_wartosci_pikseli_G[threadIdx.x + i + threadIdx.y * wielkosc_okna_przeszukania];
                        suma_B = suma_B + tablica_wartosci_pikseli_B[threadIdx.x + i + threadIdx.y * wielkosc_okna_przeszukania];

                    }
                    tablica_wartosci_pikseli_R[threadIdx.x + threadIdx.y * wielkosc_okna_przeszukania] = suma_R;
                    tablica_wartosci_pikseli_G[threadIdx.x + threadIdx.y * wielkosc_okna_przeszukania] = suma_G;
                    tablica_wartosci_pikseli_B[threadIdx.x + threadIdx.y * wielkosc_okna_przeszukania] = suma_B;
                }
            }
                __syncthreads();
            if (threadIdx.y == 0 && threadIdx.x == 0)
            {
                for (int i = 0; i < wielkosc_okna_przeszukania; i++)
                {
                    suma_wag_R = suma_wag_R + tablica_wag_pikseli_R[threadIdx.x + (threadIdx.y + i) * wielkosc_okna_przeszukania];
                }
            }
            __syncthreads();

            if (threadIdx.y == 0 && threadIdx.x == 0)
            {
                for (int i = 0; i < wielkosc_okna_przeszukania; i++)
                {
                    suma_pikseli_R = suma_pikseli_R + tablica_wartosci_pikseli_R[threadIdx.x + (threadIdx.y + i) * wielkosc_okna_przeszukania];
                    suma_pikseli_G = suma_pikseli_G + tablica_wartosci_pikseli_G[threadIdx.x + (threadIdx.y + i) * wielkosc_okna_przeszukania];
                    suma_pikseli_B = suma_pikseli_B + tablica_wartosci_pikseli_B[threadIdx.x + (threadIdx.y + i) * wielkosc_okna_przeszukania];
                }
            }
            __syncthreads();


            if ((threadIdx.x < 1) && (threadIdx.y < 1))
            {
                Dev_Macierz_odszumiona.kanal_R[index_y_pixela_przetwarzanego * szerokosc + index_x_pixela_przetwarzanego] = fminf(fmaxf(suma_pikseli_R / suma_wag_R, 0), 255);
                Dev_Macierz_odszumiona.kanal_G[index_y_pixela_przetwarzanego * szerokosc + index_x_pixela_przetwarzanego] = fminf(fmaxf(suma_pikseli_G / suma_wag_R, 0), 255);
                Dev_Macierz_odszumiona.kanal_B[index_y_pixela_przetwarzanego * szerokosc + index_x_pixela_przetwarzanego] = fminf(fmaxf(suma_pikseli_B / suma_wag_R, 0), 255);
            }
            __syncthreads();

}



void initializeCUDA(int argc, char** argv, int& devID)
{
    //funkcja na podstawie gotowego kodu udostepnionego na stronie:
    //https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp
    //linie 149 - 178
    
    cudaError_t error;
    devID = 0;
    //pobiera wersje SMs dla GPU
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
    cv::Vec3f kanaly_do_przepisania;

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
    else
    {
        wielkosc_okna_podobienstwa = 7;
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

    szerokosc_obrazka_oryginalnego = Obrazek.cols;
    wysokosc_obrazka_oryginalnego = Obrazek.rows;

    wielkosc_marginesu = (wielkosc_okna_przeszukania - 1) / 2 + (wielkosc_okna_podobienstwa - 1) / 2;

    cv::copyMakeBorder(Obrazek, Obrazek, wielkosc_marginesu, wielkosc_marginesu, wielkosc_marginesu, wielkosc_marginesu, cv::BORDER_REFLECT);
    szerokosc_obrazka_z_marginesami = szerokosc_obrazka_oryginalnego + 2 * wielkosc_marginesu;
    wysokosc_obrazka_z_marginesami = wysokosc_obrazka_oryginalnego + 2 * wielkosc_marginesu;
    wielkosc_tablicy_z_marginesami = szerokosc_obrazka_z_marginesami * wysokosc_obrazka_z_marginesami;

    Obrazek.convertTo(Obrazek, CV_32FC3);

    Obrazek_RGB Host_Macierz_wejsciowa;
    Obrazek_RGB* wskaznik_host_Macierz_wejsciowa = &Host_Macierz_wejsciowa;
    wskaznik_host_Macierz_wejsciowa->kanal_R = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Macierz_wejsciowa->kanal_G = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Macierz_wejsciowa->kanal_B = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));

    Obrazek = Obrazek.reshape(1, wielkosc_tablicy_z_marginesami);
    for (int i = 0; i < wielkosc_tablicy_z_marginesami; i++)
    {
        kanaly_do_przepisania = Obrazek.at<cv::Vec3f>(i, 0);
        Host_Macierz_wejsciowa.kanal_R[i] = kanaly_do_przepisania[0];
        Host_Macierz_wejsciowa.kanal_G[i] = kanaly_do_przepisania[1];
        Host_Macierz_wejsciowa.kanal_B[i] = kanaly_do_przepisania[2];
    }

    Obrazek_RGB Dev_Macierz_wejsciowa;
    Obrazek_RGB* wskaznik_dev_Macierz_wejsciowa = &Dev_Macierz_wejsciowa;

    cudaMalloc((&wskaznik_dev_Macierz_wejsciowa->kanal_R), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((&wskaznik_dev_Macierz_wejsciowa->kanal_G), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((&wskaznik_dev_Macierz_wejsciowa->kanal_B), wielkosc_tablicy_z_marginesami * sizeof(float));

    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_R, wskaznik_host_Macierz_wejsciowa->kanal_R, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_G, wskaznik_host_Macierz_wejsciowa->kanal_G, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_B, wskaznik_host_Macierz_wejsciowa->kanal_B, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);

    Obrazek_RGB Dev_Macierz_odszumiona;
    Obrazek_RGB* wskaznik_dev_Macierz_odszumiona = &Dev_Macierz_odszumiona;
    cudaMalloc((&wskaznik_dev_Macierz_odszumiona->kanal_R), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((&wskaznik_dev_Macierz_odszumiona->kanal_G), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((&wskaznik_dev_Macierz_odszumiona->kanal_B), wielkosc_tablicy_z_marginesami * sizeof(float));

    Obrazek_RGB Host_Macierz_odszumiona;

    Obrazek_RGB* wskaznik_host_Macierz_odszumiona = &Host_Macierz_odszumiona;
    wskaznik_host_Macierz_odszumiona->kanal_R = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Macierz_odszumiona->kanal_G = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Macierz_odszumiona->kanal_B = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));



    //////////////////////////////////////////////////////////////////tworzymy zmienne do przekazania do Kernelu starowego///////////////////////////
    ilosc_blokow_w_boku_x = szerokosc_obrazka_oryginalnego;
    ilosc_blokow_w_boku_y = wysokosc_obrazka_oryginalnego;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int wielkosc_extern_sh_memory = sizeof(float) * (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa) + sizeof(float) * ((wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1) * (wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1));
    //sigma = 0.8*sigma;
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_R, wskaznik_host_Macierz_wejsciowa->kanal_R, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_G, wskaznik_host_Macierz_wejsciowa->kanal_G, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_B, wskaznik_host_Macierz_wejsciowa->kanal_B, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaFuncSetCacheConfig(Device_Non_Local_Means, cudaFuncCachePreferShared); //zwiekszamy ilosć dostepnej pamieci shared memory
    dim3 bloki_NLM(ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, 1);
    dim3 watki_NLM(wielkosc_okna_przeszukania, wielkosc_okna_przeszukania, 1);
    Device_Non_Local_Means << <bloki_NLM, watki_NLM, wielkosc_extern_sh_memory >> >
        (Dev_Macierz_wejsciowa, Dev_Macierz_odszumiona, wielkosc_okna_podobienstwa,
            wielkosc_okna_przeszukania, szerokosc_obrazka_z_marginesami,
            wielkosc_marginesu, sigma, stala_h, Bodues);

    cudaMemcpy(wskaznik_host_Macierz_odszumiona->kanal_R, wskaznik_dev_Macierz_odszumiona->kanal_R, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wskaznik_host_Macierz_odszumiona->kanal_G, wskaznik_dev_Macierz_odszumiona->kanal_G, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wskaznik_host_Macierz_odszumiona->kanal_B, wskaznik_dev_Macierz_odszumiona->kanal_B, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);


    cv::Mat Obrazek_odszumiony_scalony(wielkosc_tablicy_z_marginesami, 1, CV_32FC3);

    for (int i = 0; i < wielkosc_tablicy_z_marginesami; i++)
    {
        kanaly_do_przepisania[0] = Host_Macierz_odszumiona.kanal_R[i];
        kanaly_do_przepisania[1] = Host_Macierz_odszumiona.kanal_G[i];
        kanaly_do_przepisania[2] = Host_Macierz_odszumiona.kanal_B[i];

        Obrazek_odszumiony_scalony.at<cv::Vec3f>(i, 0) = kanaly_do_przepisania;
    }
    Obrazek_odszumiony_scalony = Obrazek_odszumiony_scalony.reshape(0, wysokosc_obrazka_z_marginesami);
    Obrazek_odszumiony_scalony.convertTo(Obrazek_odszumiony_scalony, CV_8UC3);
    Obrazek_odszumiony_scalony = Obrazek_odszumiony_scalony(cv::Rect(wielkosc_marginesu, wielkosc_marginesu, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));
    Obrazek_odszumiony = Obrazek_odszumiony_scalony;

    cudaFree(wskaznik_dev_Macierz_wejsciowa->kanal_R);
    cudaFree(wskaznik_dev_Macierz_wejsciowa->kanal_G);
    cudaFree(wskaznik_dev_Macierz_wejsciowa->kanal_B);
    wskaznik_dev_Macierz_wejsciowa->kanal_R = nullptr;
    wskaznik_dev_Macierz_wejsciowa->kanal_G = nullptr;
    wskaznik_dev_Macierz_wejsciowa->kanal_B = nullptr;
    cudaFree(wskaznik_dev_Macierz_wejsciowa);
    cudaFree(wskaznik_dev_Macierz_odszumiona->kanal_R);
    cudaFree(wskaznik_dev_Macierz_odszumiona->kanal_G);
    cudaFree(wskaznik_dev_Macierz_odszumiona->kanal_B);
    wskaznik_dev_Macierz_odszumiona->kanal_R = nullptr;
    wskaznik_dev_Macierz_odszumiona->kanal_G = nullptr;
    wskaznik_dev_Macierz_odszumiona->kanal_B = nullptr;
    cudaFree(wskaznik_dev_Macierz_odszumiona);

    free(wskaznik_host_Macierz_wejsciowa->kanal_R);
    free(wskaznik_host_Macierz_wejsciowa->kanal_G);
    free(wskaznik_host_Macierz_wejsciowa->kanal_B);
    wskaznik_host_Macierz_wejsciowa->kanal_R = nullptr;
    wskaznik_host_Macierz_wejsciowa->kanal_G = nullptr;
    wskaznik_host_Macierz_wejsciowa->kanal_B = nullptr;

    free(wskaznik_host_Macierz_odszumiona->kanal_R);
    free(wskaznik_host_Macierz_odszumiona->kanal_G);
    free(wskaznik_host_Macierz_odszumiona->kanal_B);
    wskaznik_host_Macierz_odszumiona->kanal_R = nullptr;
    wskaznik_host_Macierz_odszumiona->kanal_G = nullptr;
    wskaznik_host_Macierz_odszumiona->kanal_B = nullptr;

}

int main(int argc, char* argv[])
{

    float sigma; // poziom szumu
    float stala_h; // parametr okreslany przed procesem odszumiania w zaleznosci od szumu i wielkosci  Okienka referncyjnego, uzywany w procesie obiczania wagi piksela
    bool Bodues = false; //czy odejmować 2*sigma*sigma od obliczoej odlegosci pomiedzy pikselami
    cv::Mat Obrazek; // obiekt opencv Mat w ktorym bedzie zapisany przetwarzany obrazek
    cv::Mat Obrazek_odszumiony;
    std::string wpisana_nazwa;
    sigma = 0;
    char filtracja_wstepna = 't';
    Bodues = true;
    stala_h = 1;
    int licznik = 0;  
    
    ///////////////////////////////////////////////////////

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
    {
        std::cout << "Filtr NLM GPU, filtruje z szumu obrazy kolorowe.\n";
        std::cout << "Uzycie: NLM_Color_GPU.exe <liczba calkowita>  <liczba calkowita lub zmiennoprzecinkowa>\n ";
        std::cout << "Argumenty:\n";
        std::cout << "  <nazwa pliku>         Nazwa pliku. Mozna podac nazwe i sciezke folderu lub sama nazwe \n";
        std::cout << "                        jezeli znajduje sie w jednym folderze z programem\n";
        std::cout << "                        -zostana przetworzone wszystkie pliki graficzne w folderze\n";
        std::cout << "  <poziom szumu>        Liczba calkowita: 0 do 100\n";
        std::cout << "  <stala filtracji>	Liczba calkowita: sila odzumiania";
        return 0;
    }

    if (argc != 4) {
        std::cerr << "Uzycie: " << argv[0] << " <nazwa pliku>  <poziom szumu> <stala filtracji>\n pomoc: --help lub -h";
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
        Obrazek = cv::imread(wpisana_nazwa, cv::IMREAD_COLOR);

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
        for (const auto& entry : std::filesystem::directory_iterator(wpisana_nazwa)) {
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
                funkcja_glowna(Obrazek, Obrazek_odszumiony, sigma, stala_h, filtracja_wstepna, Bodues);
                for (std::filesystem::path plik : {std::filesystem::absolute(std::filesystem::path(entry))})
                {
                    std::string nowa_nazwa = plik.stem().string() + "_filtered" + plik.extension().string();
                    std::string nowa_sciezka = plik.parent_path().string() + "/" + "filtered";
                    std::string nowa_nazwa_i_sciezka = nowa_sciezka + "/" + nowa_nazwa;
                    // Zapis przetworzonego obrazu
                    if (!std::filesystem::exists(nowa_sciezka)) //sprawdza czy istnieje folder do apisania wynikowych obrazow
                    {
                        if (std::filesystem::create_directories(nowa_sciezka)) {
                            std::cout << "Utworzono folder: " << nowa_sciezka << std::endl;
                        }
                        else {
                            std::cerr << "Nie udalo sie utworzyć folderu: " << nowa_sciezka << std::endl;
                            return 1; // Zakonczenie programu z bledem
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
        std::cerr << "Podana sciezka lub nazwa pliku  jest bledna" << std::endl;
        return 1;
    }
    time_t czasStop = clock();
    double czas = (double)(czasStop - czasStart) / (double)CLOCKS_PER_SEC;
    std::cout << "Przefiltrowano " << licznik << " obrazow w czasie : " << czas << " s." << std::endl;

    cv::waitKey(0);

    cudaDeviceReset();

    return 0;

}