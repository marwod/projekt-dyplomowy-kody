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
    //Niestety nie miesci się przy polach większych niż 25*25 w extern shared memory i muszę ustawić ja na sztywno jako stalą żeby okno przeszukania moglo mieć 31 na 31
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
            //if (sigma > 50) sigma = 50;

            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < wielkosc_okna_przeszukania))
            {
                Okno_preszukana_shared[threadIdx.y * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_R[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)))
            {
                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_R[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < wielkosc_okna_przeszukania)) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci wspoldzielonej bloku
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
                //przpisujemy obszar preszukania (37 pixeli) dla latki do pamięci dzielonej bloku, ze względu na zmieszczenie się w dostępnej w wywolaniu funkcji iosci wątkow musialem zrealizować przypisanie w czterech krokach.
            }
            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_G[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < wielkosc_okna_przeszukania)) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_G[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {
                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_G[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            __syncthreads();

            if ((threadIdx.x < wielkosc_okna_podobienstwa) && (threadIdx.y < wielkosc_okna_podobienstwa)) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci dzielonej bloku
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
                //tablica_wag_pikseli_G[index2dObszaru] = MSE;
                tablica_wartosci_pikseli_G[index2dObszaru] = wartosc_piksela;
            }

            //3 dla skladowej B/////////////////////////////////////////////////////////////

            if ((threadIdx.y < wielkosc_okna_przeszukania) && (threadIdx.x < wielkosc_okna_przeszukania))

                //1 dla skladowej B/////////////////////////////////////////////////////////////
            {
                Okno_preszukana_shared[threadIdx.y * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_B[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
                //przpisujemy obszar preszukania (37 pixeli) dla latki do pamięci dzielonej bloku, ze względu na zmieszczenie się w dostępnej w wywolaniu funkcji iosci wątkow musialem zrealizować przypisanie w czterech krokach.
            }
            if ((threadIdx.x < wielkosc_okna_przeszukania) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x] = (Dev_Macierz_wejsciowa.kanal_B[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }

            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < wielkosc_okna_przeszukania)) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {

                Okno_preszukana_shared[(threadIdx.y) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_B[(threadIdx.y + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            if ((threadIdx.x < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania)) && (threadIdx.y < (rzeczywisty_obszar_przeszukania - wielkosc_okna_przeszukania))) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci dzielonej bloku
            {
                Okno_preszukana_shared[(threadIdx.y + wielkosc_okna_przeszukania) * rzeczywisty_obszar_przeszukania + threadIdx.x + wielkosc_okna_przeszukania] = (Dev_Macierz_wejsciowa.kanal_B[((threadIdx.y + wielkosc_okna_przeszukania) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + wielkosc_okna_przeszukania + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
            __syncthreads();


            if ((threadIdx.x < wielkosc_okna_podobienstwa) && (threadIdx.y < wielkosc_okna_podobienstwa)) //przpisujemy obszar latki do ktorej będziemy porownywać do pamięci dzielonej bloku
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
    //funkcja na podstawie gotowego kodu udostepnionego na stronie :
    //https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp
    //linie 149 - 178
    cudaError_t error;
    devID = 0;
    //pobiera wersję SMs dla GPU
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

    double sigma = sigm; // Wartosć sigma dla szumu gaussowskiego

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


int main(int argc, char** argv)
{

    int devID = 0;
    initializeCUDA(argc, argv, devID);

    int start_0 = clock();
    int opcja_obrazka;
    float sigma;
    cv::Mat ObrazekKolorowy;
    cv::Mat ObrazekReferencyjny;
    cv::Mat NLM_OpenCv;
    std::string nazwa_pilku_zaszumionego;
    std::string nazwa_pilku_referencyjnego;
    std::string nazwa_sciezki = "obrazki_testowe/";
    int szerokosc_obrazka_oryginalnego;
    int wysokosc_obrazka_oryginalnego;
    int szerokosc_obrazka_z_marginesami;
    int wysokosc_obrazka_z_marginesami;
    int wielkosc_tablicy_z_marginesami;
    int wielkosc_okna_podobienstwa;
    int wielkosc_okna_przeszukania;
    int wielkosc_marginesu;
    float poziom_rozmycia_gauss;
    float redukcja_sigmy;
    char filtracja_wstepna;
    float stala_h;
    bool Bodues;
    cv::Vec3f kanaly_do_przepisania;
    
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
        ObrazekKolorowy = cv::imread(nazwa_pilku_zaszumionego, 1);
        ObrazekReferencyjny = cv::imread(nazwa_pilku_referencyjnego, 1);
        if (ObrazekKolorowy.empty())
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
        //cv::imshow("Obrazek zaszumiony", ObrazekKolorowy);
        //cv::imshow("referncyjny", ObrazekReferencyjny);

        std::cout << "prosze podaj poziom szumu" << std::endl;
        std::cin >> sigma;
    }

    else if (opcja_obrazka == 2)
    {
        std::cout << "prosze wpisac nazwe pliku obrazka referencyjnego wraz z rozszerzeniem" << std::endl;
        std::cin >> nazwa_pilku_referencyjnego;
        //nazwa_pilku_referencyjnego = nazwa_sciezki + nazwa_pilku_referencyjnego;
        ObrazekReferencyjny = cv::imread(nazwa_pilku_referencyjnego, 1);
        if (ObrazekReferencyjny.empty())
        {
            std::cout << "Nie mozna wczytac obrazka." << std::endl;
            return -1;
        }
        //cv::imshow("referencyjny", ObrazekReferencyjny);
        std::cout << "prosze podaj poziom szumu" << std::endl;
        std::cin >> sigma;
        ObrazekKolorowy = ObrazekReferencyjny.clone(); // Skopiuj obraz do macierzy z szumem
        dodanie_szumu(ObrazekKolorowy, sigma, 3);
        //cv::imshow("Obrazek zaszumiony", ObrazekKolorowy);
    }

    do
    {
        std::cout << "prosze podaj wielkosc okna podobienstwa: 3, 5, 7, 9" << std::endl;
        std::cin >> wielkosc_okna_podobienstwa;
    } while ((wielkosc_okna_podobienstwa != 5) && (wielkosc_okna_podobienstwa != 7) 
        && (wielkosc_okna_podobienstwa != 3) && (wielkosc_okna_podobienstwa != 9));

    do
    {
        std::cout << "czy przeprowadzic wstepa filtracje romyciem gaussa?" << std::endl;
        std::cout << "tak, wcisnij t" << std::endl;
        std::cout << "nie, wcisnij n" << std::endl;
        std::cin >> filtracja_wstepna;
    } while ((filtracja_wstepna != 't') && (filtracja_wstepna != 'n'));

    std::string odpowiedz;
    do
    {
        std::cout << "czy od sumy dystansow pomiędzy pikselami odjac 2*sigma*sigma?" << std::endl;
        std::cout << "tak, wcisnij t" << std::endl;
        std::cout << "nie, wcisnij n" << std::endl;
        std::cin >> odpowiedz;
        if (odpowiedz == "t")  Bodues = true;
        else Bodues = false;
    } while ((odpowiedz != "t") && (odpowiedz != "n"));
   
    stala_h;
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

filtracja_wstepna = 'n';

Bodues = true;

    cv::fastNlMeansDenoising(ObrazekKolorowy, NLM_OpenCv, sigma, wielkosc_okna_podobienstwa, wielkosc_okna_przeszukania);
    cv::imshow("funkcja wbudowana NLM", NLM_OpenCv);

    if (filtracja_wstepna == 't')
    {
        if (sigma < 60)
        {
            poziom_rozmycia_gauss = 0.3;
            cv::GaussianBlur(ObrazekKolorowy, ObrazekKolorowy, cv::Size(3, 3), poziom_rozmycia_gauss, poziom_rozmycia_gauss);
            redukcja_sigmy = 0.4;
        }
        else
        {
            redukcja_sigmy = 0.4;
            cv::medianBlur(ObrazekKolorowy, ObrazekKolorowy, 3);
        }
        cv::medianBlur(ObrazekKolorowy, ObrazekKolorowy, 3);
        sigma = redukcja_sigmy * sigma;
    }
    int stop_2 = clock();

    /////////////////////////////////////////////////////////////
    szerokosc_obrazka_oryginalnego = ObrazekKolorowy.cols;
    wysokosc_obrazka_oryginalnego = ObrazekKolorowy.rows;
   
    wielkosc_marginesu = (wielkosc_okna_przeszukania - 1) / 2 + (wielkosc_okna_podobienstwa - 1) / 2;
    
    cv::copyMakeBorder(ObrazekKolorowy, ObrazekKolorowy, wielkosc_marginesu, wielkosc_marginesu, wielkosc_marginesu, wielkosc_marginesu, cv::BORDER_REFLECT);
    szerokosc_obrazka_z_marginesami = szerokosc_obrazka_oryginalnego + 2 * wielkosc_marginesu;
    wysokosc_obrazka_z_marginesami = wysokosc_obrazka_oryginalnego + 2* wielkosc_marginesu;
    wielkosc_tablicy_z_marginesami = szerokosc_obrazka_z_marginesami * wysokosc_obrazka_z_marginesami;

    ObrazekKolorowy.convertTo(ObrazekKolorowy, CV_32FC3);
    Obrazek_RGB Host_Macierz_wejsciowa;
    Obrazek_RGB* wskaznik_host_Macierz_wejsciowa = &Host_Macierz_wejsciowa;
    wskaznik_host_Macierz_wejsciowa->kanal_R = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Macierz_wejsciowa->kanal_G = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Macierz_wejsciowa->kanal_B = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));

    ObrazekKolorowy = ObrazekKolorowy.reshape(0, wielkosc_tablicy_z_marginesami);
    for (int i = 0; i < wielkosc_tablicy_z_marginesami; i++)
    {
        kanaly_do_przepisania = ObrazekKolorowy.at<cv::Vec3f>(i, 0);
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
    int ilosc_blokow_w_boku_x = szerokosc_obrazka_oryginalnego;
    int ilosc_blokow_w_boku_y = wysokosc_obrazka_oryginalnego;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int wielkosc_extern_sh_memory = sizeof(float) * (wielkosc_okna_podobienstwa * wielkosc_okna_podobienstwa) + sizeof(float) * ((wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1) * (wielkosc_okna_przeszukania + wielkosc_okna_podobienstwa - 1));
    //sigma = 0.8*sigma;
  
    int start = clock();
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_R, wskaznik_host_Macierz_wejsciowa->kanal_R, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_G, wskaznik_host_Macierz_wejsciowa->kanal_G, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_B, wskaznik_host_Macierz_wejsciowa->kanal_B, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaFuncSetCacheConfig(Device_Non_Local_Means, cudaFuncCachePreferShared); //zwiększamy ilosć dostępnej pamięci shared memory
    dim3 bloki_NLM(ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, 1);
    dim3 watki_NLM(wielkosc_okna_przeszukania, wielkosc_okna_przeszukania, 1);
    Device_Non_Local_Means << <bloki_NLM, watki_NLM, wielkosc_extern_sh_memory >> > 
    (Dev_Macierz_wejsciowa, Dev_Macierz_odszumiona, wielkosc_okna_podobienstwa,
     wielkosc_okna_przeszukania, szerokosc_obrazka_z_marginesami, 
     wielkosc_marginesu, sigma, stala_h, Bodues);

    cudaMemcpy(wskaznik_host_Macierz_odszumiona->kanal_R, wskaznik_dev_Macierz_odszumiona->kanal_R, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wskaznik_host_Macierz_odszumiona->kanal_G, wskaznik_dev_Macierz_odszumiona->kanal_G, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wskaznik_host_Macierz_odszumiona->kanal_B, wskaznik_dev_Macierz_odszumiona->kanal_B, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    int stop = clock();

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
    double R = 255;

    double czas = (double)(stop - start) / (double)CLOCKS_PER_SEC;
    std::cout << std::endl << "Czas wykonania kernelu: " << czas << " s" << std::endl;
    std::cout << "PSNR po odszumieniu zaimplementowanym NLM: " << cv::PSNR(ObrazekReferencyjny, Obrazek_odszumiony_scalony, R) << std::endl;
    std::cout << "PSNR po odszumieniu NLM wbudowanym w opencv: " << cv::PSNR(ObrazekReferencyjny, NLM_OpenCv, R) << std::endl;

    ObrazekKolorowy = ObrazekKolorowy.reshape(0, wysokosc_obrazka_z_marginesami);
    ObrazekKolorowy.convertTo(ObrazekKolorowy, CV_8UC3);
    ObrazekKolorowy = ObrazekKolorowy(cv::Rect(wielkosc_marginesu, wielkosc_marginesu, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));

    cv::imshow("Obrazek zaszumiony", ObrazekKolorowy);
    cv::imshow("Obrazek referncyjny", ObrazekReferencyjny);
    cv::imshow("funkcja wbudowana NLM", NLM_OpenCv);

    cv::imshow("Obrazek odszumiony zaimplementowanym NLM", Obrazek_odszumiony_scalony);
    cv::waitKey(0);
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

    cudaDeviceReset();
    cv::waitKey(0);
    return 0;


}