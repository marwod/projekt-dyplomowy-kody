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
#include <iostream>
#include "naglowek_struktury.h"

#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartosc w ilosci latek i uzywanych watkow. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru latki
#define ROZMIAR_LATKI       8
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_LATKI ilosc pixeli obszaru przeszukania
#define POWIERZCHNIA_LATKI       64
#define N_WIEN     32


__constant__ float  Const_macierz_wspolczynnikow2d_1[POWIERZCHNIA_LATKI] = 
          { 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
            0.4904, 0.4157, 0.2778, 0.0976, -0.0975, -0.2778, -0.4157, -0.4904,
            0.4619, 0.1913, -0.1913, -0.462, -0.462, -0.1913, 0.1913, 0.4619,
            0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0976, -0.4157,
            0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536 ,
            0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778,
            0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913,
            0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975 };


__constant__ float Const_macierz_wspolczynnikow2d_2[POWIERZCHNIA_LATKI] = 
          { 0.3536, 0.4904, 0.4619, 0.4157, 0.3536, 0.2778, 0.1914, 0.0975,
            0.3536, 0.4157, 0.1914, -0.0975, -0.3536, -0.4904, -0.4619, -0.2778,
            0.3536, 0.2778, -0.1914, -0.4904, -0.3536, 0.0975, 0.4619, 0.4158,
            0.3536, 0.0976, -0.462, -0.2778, 0.3536, 0.4157, -0.1913, -0.4904,
            0.3536, -0.0976, -0.462, 0.2778, 0.3536, -0.4157, -0.1915, 0.4904,
            0.3536, -0.2778, -0.1914, 0.4904, -0.3535, -0.0977, 0.4620, -0.4156,
            0.3536, -0.4157, 0.1913, 0.0977, -0.3536, 0.4904, -0.4619, 0.2778,
            0.3536, -0.4904, 0.4619, -0.4157, 0.3534, -0.2778, 0.1911, -0.0975 };



///////////////////////////////////////////////wersje transformat bez streamow//////////////////
__global__ void DCT(Obrazek_YCrCb Tablice_Latek_Transformowanych, int rozmiar_latki_x, 
    int rozmiar_latki_y, int* device_tablica_ilosci_pasujacych_latek, 
    int mnoznik_tablicy_transformat)
{

    int nr_obszaru = blockIdx.z / mnoznik_tablicy_transformat;
    int nr_latki_w_bloku = blockIdx.z % mnoznik_tablicy_transformat;
    if (nr_latki_w_bloku < device_tablica_ilosci_pasujacych_latek[nr_obszaru])
    //if (blockIdx.z - nr_obszaru * mnoznik_tablicy_transformat < device_tablica_ilosci_pasujacych_latek[nr_obszaru])
    {
        __shared__  float Macierz_posrednia[POWIERZCHNIA_LATKI];
        __shared__  float Macierz_wejsciowa[POWIERZCHNIA_LATKI];


        int indeks_poczatkowy_latki = blockIdx.z * (rozmiar_latki_x * rozmiar_latki_y);
        int indeks_komorki = indeks_poczatkowy_latki + threadIdx.y * rozmiar_latki_y + threadIdx.x;

        ///////////////////////////////kanaL R:
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych.kanal_Y[indeks_komorki];
            Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];
            }
            __syncthreads();

            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
            __syncthreads();
            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();

            Tablice_Latek_Transformowanych.kanal_Y[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        }

        

        ////////////////kanal G:
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych.kanal_Cr[indeks_komorki];
            Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze niz rozmir Latki
            {
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Tablice_Latek_Transformowanych.kanal_Cr[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        }

 

        /////////////////kanal B
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych.kanal_Cb[indeks_komorki];
            Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
           __syncthreads();
            for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze niz rozmir Latki
            {
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];
            }
            __syncthreads();
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];
            }
            __syncthreads();
            Tablice_Latek_Transformowanych.kanal_Cb[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        }
        __syncthreads();
    }
}

///Przeciazenie DCT dla kolejnego przeksztalcenia dwoch tablic transformaty
__global__ void DCT(Obrazek_YCrCb Tablice_Latek_Transformowanych1, 
    Obrazek_YCrCb Tablice_Latek_Transformowanych2, int rozmiar_latki_x, int rozmiar_latki_y, 
    int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat)
{

    int nr_obszaru = blockIdx.z / mnoznik_tablicy_transformat;
    int nr_latki_w_bloku = blockIdx.z % mnoznik_tablicy_transformat;
    if (nr_latki_w_bloku < device_tablica_ilosci_pasujacych_latek[nr_obszaru])
    {
        __shared__  float Macierz_posrednia[POWIERZCHNIA_LATKI];
        __shared__  float Macierz_wejsciowa[POWIERZCHNIA_LATKI];

       
        int indeks_poczatkowy_latki = blockIdx.z * (rozmiar_latki_x * rozmiar_latki_y);
        int indeks_komorki = indeks_poczatkowy_latki + threadIdx.y * rozmiar_latki_y + threadIdx.x;
  
        ///////////////////////////////kanaL R:
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych1.kanal_Y[indeks_komorki];
            Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze niz rozmir Latki
            {
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Tablice_Latek_Transformowanych1.kanal_Y[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        }

            if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
            {
                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych2.kanal_Y[indeks_komorki];
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

                __syncthreads();

                for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze niz rozmir Latki
                {
                    Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

                }
                __syncthreads();

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
                __syncthreads();

                for (int k = 0; k < ROZMIAR_LATKI; k++)
                {
                    Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

                }
                __syncthreads();
                Tablice_Latek_Transformowanych2.kanal_Y[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
            }

        
        ////////////////kanal G:
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych1.kanal_Cr[indeks_komorki];
            Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze niz rozmir Latki
            {
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Tablice_Latek_Transformowanych1.kanal_Cr[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        }

            if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
            {
                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych2.kanal_Cr[indeks_komorki];
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

                __syncthreads();

                for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze niz rozmir Latki
                {
                    Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

                }
                __syncthreads();

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
                __syncthreads();

                for (int k = 0; k < ROZMIAR_LATKI; k++)
                {
                    Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

                }
                __syncthreads();
                Tablice_Latek_Transformowanych2.kanal_Cr[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
            }

        
        /////////////////kanal B
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych1.kanal_Cb[indeks_komorki];
            Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze niz rozmir Latki
            {
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();
            Tablice_Latek_Transformowanych1.kanal_Cb[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        }

            if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
            {
                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Tablice_Latek_Transformowanych2.kanal_Cb[indeks_komorki];
                Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;

                __syncthreads();

                for (int k = 0; k < ROZMIAR_LATKI; k++) // K mniejsze niz rozmir Latki
                {
                    Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Const_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

                }
                __syncthreads();

                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
                __syncthreads();

                for (int k = 0; k < ROZMIAR_LATKI; k++)
                {
                    Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * Const_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

                }
                __syncthreads();
                Tablice_Latek_Transformowanych2.kanal_Cb[indeks_komorki] = Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
            }

            __syncthreads();
    }
    
}


__global__ void DCT_odwrotna(Obrazek_YCrCb Tablice_Latek_transformowanych, int rozmiar_latki_x, int rozmiar_latki_y, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat)
{
    int nr_obszaru = blockIdx.z / mnoznik_tablicy_transformat;
    int nr_latki_w_bloku = blockIdx.z % mnoznik_tablicy_transformat;
    if (nr_latki_w_bloku < device_tablica_ilosci_pasujacych_latek[nr_obszaru])
    {

        __shared__ float Macierz_posrednia[POWIERZCHNIA_LATKI];
        __shared__  float Macierz_wejsciowa[POWIERZCHNIA_LATKI];

        int indeks_komorki =blockIdx.z * (rozmiar_latki_x * rozmiar_latki_y) + threadIdx.y * rozmiar_latki_x + threadIdx.x;
        int indeks2 = (threadIdx.y * rozmiar_latki_x + threadIdx.x);
///////////////////////////////kanal R:
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[indeks2] = Tablice_Latek_transformowanych.kanal_Y[indeks_komorki];
            Macierz_posrednia[indeks2] = 0;
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                //optymalne, zamaist transponowanej macierzy wspolczynnikow (tak jak powinno byc) uzywamy podstawowej ze zmienionym indeksowaniem rdzeni. Pozwala to zachowac coalescencje dostepu do pamieci
                Macierz_posrednia[threadIdx.y + threadIdx.x * ROZMIAR_LATKI] += Const_macierz_wspolczynnikow2d_1[k * ROZMIAR_LATKI + threadIdx.y] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

            }

            Macierz_wejsciowa[indeks2] = 0;

            __syncthreads();
            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[k * ROZMIAR_LATKI + threadIdx.y] * Const_macierz_wspolczynnikow2d_1[k * ROZMIAR_LATKI + threadIdx.x];

            }
            __syncthreads();

            Tablice_Latek_transformowanych.kanal_Y[indeks_komorki] = Macierz_wejsciowa[indeks2];

        }
        __syncthreads();
        ////////////////////////////////////////////kanal G:
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {

            Macierz_wejsciowa[indeks2] = Tablice_Latek_transformowanych.kanal_Cr[indeks_komorki];

            Macierz_posrednia[indeks2] = 0;
 


            __syncthreads();



            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                //optymalne, zamaist transponowanej macierzy wspolczynnikow (tak jak powinno byc) uzywamy podstawowej ze zmienionym indeksowaniem rdzeni. Pozwala to zachowac coalescencje dostepu do pamieci
                Macierz_posrednia[threadIdx.y + threadIdx.x * ROZMIAR_LATKI] += Const_macierz_wspolczynnikow2d_1[k * ROZMIAR_LATKI + threadIdx.y] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];

            }

            Macierz_wejsciowa[indeks2] = 0;

            __syncthreads();
            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[k * ROZMIAR_LATKI + threadIdx.y] * Const_macierz_wspolczynnikow2d_1[k * ROZMIAR_LATKI + threadIdx.x];
            }
            __syncthreads();

            Tablice_Latek_transformowanych.kanal_Cr[indeks_komorki] = Macierz_wejsciowa[indeks2];

        }
        ////////////////////////////////////kanal B:
        if (threadIdx.x < ROZMIAR_LATKI && threadIdx.y < ROZMIAR_LATKI)
        {
            Macierz_wejsciowa[indeks2] = Tablice_Latek_transformowanych.kanal_Cb[indeks_komorki];
            Macierz_posrednia[indeks2] = 0;
            __syncthreads();

            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                //optymalne, zamaist transponowanej macierzy wspolczynnikow (tak jak powinno byc) uzywamy podstawowej ze zmienionym indeksowaniem rdzeni. Pozwala to zachowac coalescencje dostepu do pamieci
                Macierz_posrednia[threadIdx.y + threadIdx.x * ROZMIAR_LATKI] += Const_macierz_wspolczynnikow2d_1[k * ROZMIAR_LATKI + threadIdx.y] * Macierz_wejsciowa[k * ROZMIAR_LATKI + threadIdx.x];
            }

            Macierz_wejsciowa[indeks2] = 0;

            __syncthreads();
            for (int k = 0; k < ROZMIAR_LATKI; k++)
            {
                Macierz_wejsciowa[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_posrednia[k * ROZMIAR_LATKI + threadIdx.y] * Const_macierz_wspolczynnikow2d_1[k * ROZMIAR_LATKI + threadIdx.x];
            }
            __syncthreads();

            Tablice_Latek_transformowanych.kanal_Cb[indeks_komorki] = Macierz_wejsciowa[indeks2];
        }
        __syncthreads();
    }

}
__global__ void Walsh_1D(Obrazek_YCrCb Tablice_Latek_transformowanych,
    int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transormat)
{
    int ilosc_pasujacych_latek = device_tablica_ilosci_pasujacych_latek[blockIdx.z];
    float pierwN = sqrtf(ilosc_pasujacych_latek);
    int indeks_poczatkowy_grupy = blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat;
    int szerokosc = blockIdx.x * blockDim.x + threadIdx.x;
    int glebokosc = blockIdx.y * blockDim.y + threadIdx.y;
    int indeks2d = glebokosc * ROZMIAR_LATKI * ROZMIAR_LATKI + szerokosc;

    extern __shared__ float  macierz_s[];

    if (device_tablica_ilosci_pasujacych_latek[blockIdx.z] > 1)
    {

        /////////////////////////////////////////Kanal R:
        for (int i = 0; i < 2; i++)
        {
            if (threadIdx.y < ilosc_pasujacych_latek / 2)
            {
                macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = Tablice_Latek_transformowanych.kanal_Y[indeks_poczatkowy_grupy + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)];
            }
        }

        __syncthreads();
        
        if (threadIdx.y < ilosc_pasujacych_latek / 2)
        {
            int h = 1;
            while (h < ilosc_pasujacych_latek)

            {
                if (threadIdx.y % h == 0)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float tymczasowa1 = macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + i * ROZMIAR_LATKI * ROZMIAR_LATKI)];
                        float tymczasowa2 = macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + (i + h) * ROZMIAR_LATKI * ROZMIAR_LATKI)];
                        macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + i * ROZMIAR_LATKI * ROZMIAR_LATKI)] = tymczasowa1 + tymczasowa2;
                        macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + (i + h) * ROZMIAR_LATKI * ROZMIAR_LATKI)] = tymczasowa1 - tymczasowa2;
                    }
                }
                h *= 2;
            }
        }
        __syncthreads();
        


        for (int i = 0; i < 2; i++)
        {
            if (threadIdx.y < ilosc_pasujacych_latek / 2)
            {
            Tablice_Latek_transformowanych.kanal_Y[indeks_poczatkowy_grupy + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] / pierwN;
            }
        }
        __syncthreads();
        ////////////////////////////////kanal Cr:
        for (int i = 0; i < 2; i++)
        {
            if (threadIdx.y < ilosc_pasujacych_latek / 2)
            {
                macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = Tablice_Latek_transformowanych.kanal_Cr[indeks_poczatkowy_grupy + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)];
            }
        }
        __syncthreads();

        
        if (threadIdx.y < ilosc_pasujacych_latek / 2)
        {

            int h = 1;
            while (h < ilosc_pasujacych_latek)

            {
                if ((threadIdx.y) % (h) == 0)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float tymczasowa1 = macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + i * ROZMIAR_LATKI * ROZMIAR_LATKI)];
                        float tymczasowa2 = macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + (i + h) * ROZMIAR_LATKI * ROZMIAR_LATKI)];
                        macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + i * ROZMIAR_LATKI * ROZMIAR_LATKI)] = tymczasowa1 + tymczasowa2;
                        macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + (i + h) * ROZMIAR_LATKI * ROZMIAR_LATKI)] = tymczasowa1 - tymczasowa2;
                    }
                }
                h *= 2;
            }
        }

        __syncthreads();
        
        for (int i = 0; i < 2; i++)
        {

            if (threadIdx.y < ilosc_pasujacych_latek / 2)
            {
                Tablice_Latek_transformowanych.kanal_Cr[indeks_poczatkowy_grupy + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] / pierwN;
            }
        }
        __syncthreads();

        /////////////////////////////////kanal B:
        for (int i = 0; i < 2; i++)
        {
            if (threadIdx.y < ilosc_pasujacych_latek / 2)
            {
                macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = Tablice_Latek_transformowanych.kanal_Cb[indeks_poczatkowy_grupy + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)];
            }
        }

        __syncthreads();
        
        if (threadIdx.y < ilosc_pasujacych_latek / 2)
        {

            int h = 1;
            while (h < ilosc_pasujacych_latek)

            {
                if ((threadIdx.y) % (h) == 0)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float tymczasowa1 = macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + i * ROZMIAR_LATKI * ROZMIAR_LATKI)];
                        float tymczasowa2 = macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + (i + h) * ROZMIAR_LATKI * ROZMIAR_LATKI)];
                        macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + i * ROZMIAR_LATKI * ROZMIAR_LATKI)] = tymczasowa1 + tymczasowa2;
                        macierz_s[(threadIdx.y * 2 * ROZMIAR_LATKI * ROZMIAR_LATKI + threadIdx.x + (i + h) * ROZMIAR_LATKI * ROZMIAR_LATKI)] = tymczasowa1 - tymczasowa2;
                    }
                }
                h *= 2;
            }
        }
        __syncthreads();
        
        for (int i = 0; i < 2; i++)
        {
            Tablice_Latek_transformowanych.kanal_Cb[indeks_poczatkowy_grupy + indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] = macierz_s[indeks2d + (ROZMIAR_LATKI * ROZMIAR_LATKI * ilosc_pasujacych_latek / 2 * i)] / pierwN;
            
        }
    }

}
