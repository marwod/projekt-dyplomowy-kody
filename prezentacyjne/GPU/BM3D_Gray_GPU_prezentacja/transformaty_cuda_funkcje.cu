/*
© Marcin Wodejko 2024.
marwod@interia.pl
*/

#include "transformaty_cuda.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"

#include <math.h>
#include <cmath>
#include <cuda_runtime.h>
#include <conio.h>
#include <iostream>

#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartoœæ w iloœci ³atek i u¿ywanych w¹tków. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru ³atki
#define ROZMIAR_LATKI       8
#define N_WIEN     32
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_£ATKI iloœæ pixeli obszaru przeszukania
#define POWIERZCHNIA_LATKI       64


__constant__ float  Const_macierz_wspolczynnikow2d_1[POWIERZCHNIA_LATKI] = { 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
                                           0.4904, 0.4157, 0.2778, 0.0976, -0.0975, -0.2778, -0.4157, -0.4904,
                                           0.4619, 0.1913, -0.1913, -0.462, -0.462, -0.1913, 0.1913, 0.4619,
                                           0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0976, -0.4157,
                                           0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536 ,
                                           0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778,
                                           0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913,
                                           0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975 };


__constant__ float Const_macierz_wspolczynnikow2d_2[POWIERZCHNIA_LATKI] = { 0.3536, 0.4904, 0.4619, 0.4157, 0.3536, 0.2778, 0.1914, 0.0975,
                                         0.3536, 0.4157, 0.1914, -0.0975, -0.3536, -0.4904, -0.4619, -0.2778,
                                         0.3536, 0.2778, -0.1914, -0.4904, -0.3536, 0.0975, 0.4619, 0.4158,
                                         0.3536, 0.0976, -0.462, -0.2778, 0.3536, 0.4157, -0.1913, -0.4904,
                                         0.3536, -0.0976, -0.462, 0.2778, 0.3536, -0.4157, -0.1915, 0.4904,
                                         0.3536, -0.2778, -0.1914, 0.4904, -0.3535, -0.0977, 0.4620, -0.4156,
                                         0.3536, -0.4157, 0.1913, 0.0977, -0.3536, 0.4904, -0.4619, 0.2778,
                                         0.3536, -0.4904, 0.4619, -0.4157, 0.3534, -0.2778, 0.1911, -0.0975 };





__global__ void DCT(float* Macierz1, float* Macierz2, int x, int y, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat, bool krok2)
{
    __shared__ float  sz_Const_macierz_wspolczynnikow2d_1[POWIERZCHNIA_LATKI];
    __shared__ float sz_Const_macierz_wspolczynnikow2d_2[POWIERZCHNIA_LATKI];
    
    float  macierz_wspolczynnikow2d_1[POWIERZCHNIA_LATKI] = { 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
                                           0.4904, 0.4157, 0.2778, 0.0976, -0.0975, -0.2778, -0.4157, -0.4904,
                                           0.4619, 0.1913, -0.1913, -0.462, -0.462, -0.1913, 0.1913, 0.4619,
                                           0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0976, -0.4157,
                                           0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536 ,
                                           0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778,
                                           0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913,
                                           0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975 };


   float macierz_wspolczynnikow2d_2[POWIERZCHNIA_LATKI] = { 0.3536, 0.4904, 0.4619, 0.4157, 0.3536, 0.2778, 0.1914, 0.0975,
                                             0.3536, 0.4157, 0.1914, -0.0975, -0.3536, -0.4904, -0.4619, -0.2778,
                                             0.3536, 0.2778, -0.1914, -0.4904, -0.3536, 0.0975, 0.4619, 0.4158,
                                             0.3536, 0.0976, -0.462, -0.2778, 0.3536, 0.4157, -0.1913, -0.4904,
                                             0.3536, -0.0976, -0.462, 0.2778, 0.3536, -0.4157, -0.1915, 0.4904,
                                             0.3536, -0.2778, -0.1914, 0.4904, -0.3535, -0.0977, 0.4620, -0.4156,
                                             0.3536, -0.4157, 0.1913, 0.0977, -0.3536, 0.4904, -0.4619, 0.2778,
                                             0.3536, -0.4904, 0.4619, -0.4157, 0.3534, -0.2778, 0.1911, -0.0975 };
                                             
   
    int nr_watku = blockIdx.z / mnoznik_tablicy_transformat;
    if (blockIdx.z - nr_watku * mnoznik_tablicy_transformat < device_tablica_ilosci_pasujacych_latek[nr_watku])
    {
        __shared__  float Macierz_wynikowa_1[POWIERZCHNIA_LATKI];
        __shared__  float Macierz_wejsciowa[POWIERZCHNIA_LATKI];

        int wysokosc = blockIdx.z * blockDim.z + threadIdx.z;
        int szerokosc = blockIdx.x * blockDim.x + threadIdx.x;
        int glebokosc = blockIdx.y * blockDim.y + threadIdx.y;
        int indeks1 = wysokosc * (x * y) + glebokosc * x + szerokosc;
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            sz_Const_macierz_wspolczynnikow2d_1[(threadIdx.y * ROZMIAR_LATKI + threadIdx.x)] = Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
            sz_Const_macierz_wspolczynnikow2d_2[(threadIdx.y * ROZMIAR_LATKI + threadIdx.x)] = Const_macierz_wspolczynnikow2d_2[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];

            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Macierz1[indeks1];
            Macierz_wynikowa_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze ni¿ rozmir Latki
            {
                Macierz_wynikowa_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += sz_Const_macierz_wspolczynnikow2d_1[(threadIdx.y * ROZMIAR_LATKI + k)] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_1[threadIdx.y * ROZMIAR_LATKI + k] * sz_Const_macierz_wspolczynnikow2d_2[(k * ROZMIAR_LATKI + threadIdx.x)];

            }
            __syncthreads();
            Macierz1[indeks1] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        }

        if (krok2 == true)
        {
            if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
            {
                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Macierz2[indeks1];
                Macierz_wynikowa_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

                __syncthreads();

                for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze ni¿ rozmir Latki
                {
                    Macierz_wynikowa_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += sz_Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

                }
                __syncthreads();

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
                __syncthreads();

                for (int k = 0; k < ROZMIAR_LATKI; k++)
                {
                    Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_1[threadIdx.y * ROZMIAR_LATKI + k] * sz_Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

                }
                __syncthreads();
                Macierz2[indeks1] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
            }

        }
    }

}


__global__ void DCT_odwrotna(float* Macierz, int x, int y, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat)
{
    float  macierz_wspolczynnikow2d_1[POWIERZCHNIA_LATKI] = { 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
                                         0.4904, 0.4157, 0.2778, 0.0976, -0.0975, -0.2778, -0.4157, -0.4904,
                                         0.4619, 0.1913, -0.1913, -0.462, -0.462, -0.1913, 0.1913, 0.4619,
                                         0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0976, -0.4157,
                                         0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536 ,
                                         0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778,
                                         0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913,
                                         0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975 };
   
    int nr_watku = blockIdx.z / mnoznik_tablicy_transformat;
    if (blockIdx.z - nr_watku * mnoznik_tablicy_transformat < device_tablica_ilosci_pasujacych_latek[nr_watku])
    {
        
        __shared__ float  sz_Const_macierz_wspolczynnikow2d_1[POWIERZCHNIA_LATKI];     
        __shared__ float Macierz_wynikowa_1[POWIERZCHNIA_LATKI];
        __shared__  float Macierz_wejsciowa[POWIERZCHNIA_LATKI];


        int wysokosc = blockIdx.z * blockDim.z + threadIdx.z;
        int szerokosc = blockIdx.x * blockDim.x + threadIdx.x;
        int glebokosc = blockIdx.y * blockDim.y + threadIdx.y;
        int indeks1 = wysokosc * (x * y) + glebokosc * x + szerokosc;
        int indeks2 = (threadIdx.y * x + threadIdx.x);

        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[indeks2] = Macierz[indeks1];
            Macierz_wynikowa_1[indeks2] = 0;
            __syncthreads();
            
            sz_Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                //optymalne, mo¿na pos³ugiwaæ siê tylko jedn¹ macierz¹ wspó³czynników za³adowan¹ do shared memoery
                Macierz_wynikowa_1[threadIdx.y + threadIdx.x * ROZMIAR_LATKI] += sz_Const_macierz_wspolczynnikow2d_1[(k * ROZMIAR_LATKI + threadIdx.y)] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

            }

            Macierz_wejsciowa[indeks2] = 0;

            __syncthreads();
            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_1[k * ROZMIAR_LATKI + threadIdx.y] * sz_Const_macierz_wspolczynnikow2d_1[(k * ROZMIAR_LATKI + threadIdx.x)];
                //__syncthreads();
            }
            __syncthreads();

            Macierz[indeks1] = Macierz_wejsciowa[indeks2];

        }
     
    }

}


__global__ void Walsh1dPojedyncza(float* macierz1, float* macierz2, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transormat, bool krok2) // macierz - przekazukemy  "device_tablice_transformaty_32_" //dobraaaaaaaaa
{

    int ilosc_pasujacych_latek = device_tablica_ilosci_pasujacych_latek[blockIdx.z];
    int przesuniecie = blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat;
    int glebokosc = blockIdx.x * blockDim.x + threadIdx.x;
    int szerokosc = blockIdx.y * blockDim.y + threadIdx.y;
    int indeks2d = szerokosc * ROZMIAR_LATKI * ROZMIAR_LATKI + glebokosc;

    if (device_tablica_ilosci_pasujacych_latek[blockIdx.z] > 1)
    {

        __shared__  float macierz_s[POWIERZCHNIA_LATKI * N_WIEN];//macie¿ w shared memory pozwalaj¹ca za³adowaæ maksymalne 32 ³atki 8 na 8
        for (int i = 0; i < 2; i++)
        {
            macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = macierz1[przesuniecie + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)];
        }
        __syncthreads();

        float pierwN = sqrtf(ilosc_pasujacych_latek);

        for (int len = 1; len < ilosc_pasujacych_latek; len *= 2)
        {
            int max_threads = ilosc_pasujacych_latek / (2 * len);
            int start = threadIdx.y * 2 * len;
            int start2 = (threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI) * len + threadIdx.x;
            if (threadIdx.y < max_threads)
            {
                if (start < ilosc_pasujacych_latek)
                {
                    for (int j = 0; j < len; j++)
                    {
                        float a = macierz_s[start2 + j * ROZMIAR_LATKI * ROZMIAR_LATKI];
                        float b = macierz_s[start2 + (j + len) * ROZMIAR_LATKI * ROZMIAR_LATKI];
                        macierz_s[start2 + j * ROZMIAR_LATKI * ROZMIAR_LATKI] = (a + b);
                        macierz_s[start2 + (j + len) * ROZMIAR_LATKI * ROZMIAR_LATKI] = (a - b);
                    }
                }
            }

        }
        __syncthreads();

        for (int i = 0; i < 2; i++)
        {
            macierz1[przesuniecie + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] / pierwN;
        }

        if (krok2 == true)
        {

            for (int i = 0; i < 2; i++)
            {
                macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * blockDim.y * i)] = macierz2[przesuniecie + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * blockDim.y * i)];
            }
            __syncthreads();

            for (int len = 1; len < ilosc_pasujacych_latek; len *= 2)
            {
                int max_threads = ilosc_pasujacych_latek / (2 * len);
            
                int start = threadIdx.y * 2 * len;
                int start2 = (threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI) * len + threadIdx.x;
                if (threadIdx.y < max_threads)
                {
                    if (start < ilosc_pasujacych_latek)
                    {
                        for (int j = 0; j < len; j++)
                        {
                            float a = macierz_s[start2 + j * ROZMIAR_LATKI * ROZMIAR_LATKI];
                            float b = macierz_s[start2 + (j + len) * ROZMIAR_LATKI * ROZMIAR_LATKI];
                            macierz_s[start2 + j * ROZMIAR_LATKI * ROZMIAR_LATKI] = (a + b);
                            macierz_s[start2 + (j + len) * ROZMIAR_LATKI * ROZMIAR_LATKI] = (a - b);
                        }
                    }
                }

            }
            __syncthreads();

            for (int i = 0; i < 2; i++)
            {
                macierz2[przesuniecie + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] / pierwN;
            }

        }

    }

}