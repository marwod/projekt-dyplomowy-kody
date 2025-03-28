/*
© Marcin Wodejko 2024.
marwod@interia.pl
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <device_functions.h>
#include "device_launch_parameters.h"
#include <cuda_texture_types.h>
#include <stdint.h>
#include <string>
#include <time.h>
#include <iostream>
#include <cmath>
//#include <conio.h>
#include <math.h>
#include <cmath>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "Naglowek_struktury.h"
#include "transformaty_cuda.cuh"
#include <random>
//.werrsja po poprawkach 8 maja 2024, bardzo szybka. poprawiono transformate walsha


#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartoœc w iloœci latek i uzywanych watkow. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru latki
#define ROZMIAR_LATKI       8
#define POWIERZCHNIA_LATKI       64
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_LATKI iloœc pixeli obszaru przeszukania
#define SIGMA       20.0f
#define LAMBDA2DHARD       0.9f
#define N_HARD  16 //maks ilosc lek w grupie 3D
#define N_WIEN  32//maks ilosc lek w grupie 3D
#define P_HARD  3 //p_Hard krok tworzenia latek, w oryginale 1,2 lub 3, u Lebruna 3
#define P_WIEN  3 //krok tworzenia latek, w oryginale 1, 2 lub 3, u Lebruna 3
#define TAU_HARD_NISKI 4000.0 //maksymalna odlegloœc MSE latki przysz szumie niskim
#define TAU_HARD_WYSOKI 3000.0 //maksymalna odlegloœc MSE latki przysz szumie niskim
#define LAMBDA3D_HARD 2.7  //LambdaHard2d	progowanie(trasholding) Grupy3d w pierwszym kroku filtra, u Lebruna 2,7
#define LAMBDA2D_HARD 2.0//Lambda_hard3d progowanie(trasholding) przy block matchingu, u Lebruna 2.0




__constant__ float Macierz_wspolczynnikow_Kaizerra[POWIERZCHNIA_LATKI] = 
{ 0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924,
  0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
  0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
  0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
  0.4325, 0.6717, 0.8644, 0.9718, 0.9718, 0.8644, 0.6717, 0.4325,
  0.3846, 0.5974, 0.7688, 0.8644, 0.8644, 0.7688, 0.5974, 0.3846,
  0.2989, 0.4642, 0.5974, 0.6717, 0.6717, 0.5974, 0.4642, 0.2989,
  0.1924, 0.2989, 0.3846, 0.4325, 0.4325, 0.3846, 0.2989, 0.1924 };

__constant__ float  aConst_macierz_wspolczynnikow2d_1[POWIERZCHNIA_LATKI] =
{ 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536,
0.4904, 0.4157, 0.2778, 0.0976, -0.0975, -0.2778, -0.4157, -0.4904,
0.4619, 0.1913, -0.1913, -0.462, -0.462, -0.1913, 0.1913, 0.4619,
0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0976, -0.4157,
0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536 ,
0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778,
0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913,
0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975};

__constant__ float aConst_macierz_wspolczynnikow2d_2[POWIERZCHNIA_LATKI] =
{ 0.3536, 0.4904, 0.4619, 0.4157, 0.3536, 0.2778, 0.1914, 0.0975,  
  0.3536, 0.4157, 0.1914, -0.0975, -0.3536, -0.4904, -0.4619, -0.2778,
  0.3536, 0.2778, -0.1914, -0.4904, -0.3536, 0.0975, 0.4619, 0.4157,
  0.3536, 0.0976, -0.462, -0.2778, 0.3536, 0.4157, -0.1913, -0.4904,
  0.3536, -0.0976, -0.462, 0.2778, 0.3536, -0.4157, -0.1915, 0.4904,
  0.3536, -0.2778, -0.1914, 0.4904, -0.3535, -0.0977, 0.4620, -0.4157,
  0.3536, -0.4157, 0.1913, 0.0977, -0.3536, 0.4904, -0.4619, 0.2778, 
  0.3536, -0.4904, 0.4619, -0.4157, 0.3534, -0.2778, 0.1911, -0.0975 };


/*
__global__ void Najmniejsze_liczby(Tablice_koordynatLatek koordynatySOA, int* device_tablica_ilosci_pasujacych_latek, int ilosc_najmniejszych, int tau, bool krok2) // wykorzystanie algorytmu redykcji uzywanego zwykle do sumowania tablicy
{

    int indeks = threadIdx.x;

    int przesuniecie = blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA;
    __shared__ int s_koordynatySOA[ROZMIAR_OBSZARU_PRZESZUKANIA*ROZMIAR_OBSZARU_PRZESZUKANIA];
    __shared__ float s_MSE_SOA[ROZMIAR_OBSZARU_PRZESZUKANIA*ROZMIAR_OBSZARU_PRZESZUKANIA];
    __shared__ int s_koordynaty_najmniejszych_SOA[N_WIEN];
    for (int i = 0; i < 2; i++)
    {
        if (indeks < 512)
        {
            s_MSE_SOA[indeks + (i * 512)] = koordynatySOA.MSE[przesuniecie + indeks + (i * 512)];
            s_koordynatySOA[indeks + (i * 512)] = indeks + (i * 512);

        }
    }
    __syncthreads();

    for (int i = 0; i < ilosc_najmniejszych; i++)//ilosc najmniejszych wynosi 16 dla pierwszego kroku lub 32 dla drugiego stad w pamieci zarezerwowano miejsce dla 32
    {

        for (int s = 512; s > 0; s >>= 1)//s zmniejszamy dwukrotnie za kazda iteracja
        {

            if (threadIdx.x < s)
            {
                if (s_MSE_SOA[s_koordynatySOA[indeks]] > s_MSE_SOA[s_koordynatySOA[indeks + s]])
                {
                    s_koordynatySOA[indeks] = s_koordynatySOA[indeks + s];
                }
            }
        }
        //__syncthreads();

        if (threadIdx.x == 0)
        {
            s_koordynaty_najmniejszych_SOA[i] = s_koordynatySOA[0];
            s_MSE_SOA[s_koordynatySOA[0]] = 10000000000000000000; //aby w nastepnej iteracji wyszukiwania zostal zigm=norowana jako kandydat do najmniejszego
        }
    }
    //__syncthreads();

    if (indeks < ilosc_najmniejszych)
    {
        float tymczasowa_MSE = koordynatySOA.MSE[przesuniecie + s_koordynaty_najmniejszych_SOA[indeks]];
        int tymczasowa_x = koordynatySOA.koordynata_x[przesuniecie + s_koordynaty_najmniejszych_SOA[indeks]];
        int tymczasowa_y = koordynatySOA.koordynata_y[przesuniecie + s_koordynaty_najmniejszych_SOA[indeks]];

        koordynatySOA.MSE[indeks + przesuniecie] = tymczasowa_MSE;
        koordynatySOA.koordynata_x[indeks + przesuniecie] = tymczasowa_x;
        koordynatySOA.koordynata_y[indeks + przesuniecie] = tymczasowa_y;
    }
    __syncthreads();

    if (threadIdx.x == 1)
    {
        for (int i = ilosc_najmniejszych; i > 0; i = i / 2)
        {
            if (koordynatySOA.MSE[przesuniecie + i - 1] < tau)
            {
                device_tablica_ilosci_pasujacych_latek[blockIdx.z] = i;
                break;
            }
        }
        if (device_tablica_ilosci_pasujacych_latek[blockIdx.z] == 0)
        {
            device_tablica_ilosci_pasujacych_latek[blockIdx.z] = 1;
        }
    }

}
*/

__global__ void Najmniejsze_liczby(Tablice_koordynatLatek koordynatySOA, int* device_tablica_ilosci_pasujacych_latek, int ilosc_najmniejszych, float tau, bool krok2)
// wykorzystanie algorytmu redykcji uzywanego zwykle do sumowania tablicy
{
    int przesuniecie = blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA;
    __shared__ int s_tablica_indeksow_poczatkowych[ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA];
    __shared__ float s_tablica_wartosci_MSE[ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA];
    __shared__ int s_koordynaty_najmniejszych_SOA[32];
    for (int i = 0; i < 2; i++)
    {
        if (threadIdx.x < 512)
        {
            s_tablica_wartosci_MSE[threadIdx.x + (i * 512)] = koordynatySOA.MSE[przesuniecie + threadIdx.x + (i * 512)];
            s_tablica_indeksow_poczatkowych[threadIdx.x + (i * 512)] = threadIdx.x + (i * 512);
        }
    }
    __syncthreads();


    for (int i = 0; i < ilosc_najmniejszych; i++)
    {
        for (int s = (ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA / 2); s > 0; s >>= 1)//s zmniejszamy dwukrotnie za kazda iteracja
        {

            if ((threadIdx.x < s))
            {
                if (s_tablica_wartosci_MSE[s_tablica_indeksow_poczatkowych[threadIdx.x]] > s_tablica_wartosci_MSE[s_tablica_indeksow_poczatkowych[threadIdx.x + s]])
                {
                    s_tablica_indeksow_poczatkowych[threadIdx.x] = s_tablica_indeksow_poczatkowych[threadIdx.x + s];
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            s_koordynaty_najmniejszych_SOA[i] = s_tablica_indeksow_poczatkowych[0];
            s_tablica_wartosci_MSE[s_tablica_indeksow_poczatkowych[0]] = 10000000000000000000;
        }
    }
    __syncthreads();

    if (threadIdx.x < ilosc_najmniejszych)
    {
        float tymczasowa_MSE = koordynatySOA.MSE[przesuniecie + s_koordynaty_najmniejszych_SOA[threadIdx.x]];
        int tymczasowa_x = koordynatySOA.koordynata_x[przesuniecie + s_koordynaty_najmniejszych_SOA[threadIdx.x]];
        int tymczasowa_y = koordynatySOA.koordynata_y[przesuniecie + s_koordynaty_najmniejszych_SOA[threadIdx.x]];

        koordynatySOA.MSE[threadIdx.x + przesuniecie] = tymczasowa_MSE;
        koordynatySOA.koordynata_x[threadIdx.x + przesuniecie] = tymczasowa_x;
        koordynatySOA.koordynata_y[threadIdx.x + przesuniecie] = tymczasowa_y;
    }
    __syncthreads();

    if (threadIdx.x == 1)
    {
        for (int i = ilosc_najmniejszych; i > 0; i = i / 2)
        {
            if (koordynatySOA.MSE[przesuniecie + i - 1] < tau)
            {
                device_tablica_ilosci_pasujacych_latek[blockIdx.z] = i;
                break;
            }
        }
        if (device_tablica_ilosci_pasujacych_latek[blockIdx.z] == 0)
        {
            device_tablica_ilosci_pasujacych_latek[blockIdx.z] = 1;
        }
    }
}

__global__ void Kalkulator_MSE(float* __restrict__ device_obrazek_poczatkowy, Tablice_koordynatLatek dev_koordynatySOA, int ilosc_blokow_w_boku_x, int ilosc_blokow_w_boku_y, int szerokosc, int i, int j)
{

    int row_pos = threadIdx.y;
    int col_pos = threadIdx.x;
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index2dObszaru = threadIdx.x + threadIdx.y * ROZMIAR_OBSZARU_PRZESZUKANIA;
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    __shared__ float latka_referencyjna[POWIERZCHNIA_LATKI];
    __shared__ float obszar_preszukana_shared[(ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI) * (ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI)];
    int ofset = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2;
    if ((row_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2) && (col_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2))
    {

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                obszar_preszukana_shared[(row_pos + i * ofset) * (RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA)+col_pos + (j * ofset)] = (device_obrazek_poczatkowy[((row_pos + i * ofset) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + col_pos + (j * ofset) + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
                //przpisujemy obszar preszukania (40 pixeli) dla latki do pamieci dzielonej bloku, ze wzgledu na zmieszczenie sie w dostepnej w wywolaniu funkcji iosci watkow musialem zrealizowac przypisanie w czterech krokach.
            }

        }
    }
    __syncthreads();
    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI)) //przpisujemy obszar latki do ktorej bedziemy porownywac do pamieci dzielonej bloku
    {
        latka_referencyjna[row_pos * ROZMIAR_LATKI + col_pos] = obszar_preszukana_shared[(row_pos + ofset) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + (col_pos + ofset)];
        // przypisujemy wartoœci dla latki referencyjnej (latka o rozmiarze 8*8) dla kazdego obszaru przeszukania. Ofset jest potrzebny w zwiazku z rozna wielkoœcia latki i obaszaru przeszukania oraz tym ze latka referencyjna umieszona jest w œrodku tzn jej lewy gorny rog jest umieszczony w œrodku x=19, y=19.
        //Latka nie jest umieszczona idealnie po œrodu, ale jest to kompromis ktory zapewnia pokrycie calgo obszaru i wspolprace z 1024 watkami w bloku.
    }
    __syncthreads();

    if ((row_pos < (ROZMIAR_OBSZARU_PRZESZUKANIA)) && (col_pos < (ROZMIAR_OBSZARU_PRZESZUKANIA))) //Obliczamy MSE dla 32*32 latek z obszaru przeszukania
    {
        dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + index2dObszaru] = col_pos;
        dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + index2dObszaru] = row_pos;
        float MSE = 0;

        for (int i = 0; i < ROZMIAR_LATKI; i++)
        {
            for (int j = 0; j < ROZMIAR_LATKI; j++)
            {

                MSE += (((latka_referencyjna[i * ROZMIAR_LATKI + j] - obszar_preszukana_shared[(row_pos + i) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + col_pos + j])) * (latka_referencyjna[i * ROZMIAR_LATKI + j] - obszar_preszukana_shared[(row_pos + i) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + col_pos + j]));
                //__syncthreads(); // zeby nie bylo pchania sie watkow jednoczeœnie do tech samych pikseli slre niestety nie dziala przez petle jheœli wszystkie watki nie sa zatrudnione
            }
           
        }
        __syncthreads();
        dev_koordynatySOA.MSE[index_elmentu_zero_tablicy_koordynat + index2dObszaru] = (MSE / (POWIERZCHNIA_LATKI));
    }
}

__global__ void Kalkulator_MSE_szum_duzy(float* __restrict__ Obrazek, Tablice_koordynatLatek dev_koordynatySOA, int ilosc_blokow_w_boku_x, int szerokosc, int i, int j, int sigma)
{

    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index2dLatki = threadIdx.x + threadIdx.y * ROZMIAR_LATKI;
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int ofset = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2;
    __shared__ float  sz_Const_macierz_wspolczynnikow2d_1[POWIERZCHNIA_LATKI+8];
    __shared__ float sz_Const_macierz_wspolczynnikow2d_2[POWIERZCHNIA_LATKI+8];
    //__shared__ float  sz_Const_macierz_wspolczynnikow2d_1[8][9];
    //__shared__ float sz_Const_macierz_wspolczynnikow2d_2[8][9];
    __shared__ float latka_referencyjna[ROZMIAR_LATKI * ROZMIAR_LATKI];
    __shared__ float latka_porownywana[ROZMIAR_LATKI * ROZMIAR_LATKI];
    __shared__ float Macierz_wynikowa_posrednia[POWIERZCHNIA_LATKI];
    //sigma = sigma * sigma;

    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {

        sz_Const_macierz_wspolczynnikow2d_1[(threadIdx.y * (ROZMIAR_LATKI+1) + threadIdx.x)] = aConst_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        sz_Const_macierz_wspolczynnikow2d_2[(threadIdx.y * (ROZMIAR_LATKI+1) + threadIdx.x)] = aConst_macierz_wspolczynnikow2d_2[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        
        //sz_Const_macierz_wspolczynnikow2d_1[threadIdx.y] [threadIdx.x] = aConst_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        //sz_Const_macierz_wspolczynnikow2d_2[threadIdx.y][threadIdx.x] = aConst_macierz_wspolczynnikow2d_2[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];

        latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Obrazek[((threadIdx.y + ofset - 4) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + (ofset - 4) + index_x_pixela_gorny_lewy_obszaru_przeszukania];
        //__syncthreads();

        latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Obrazek[((threadIdx.y + blockIdx.y) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + blockIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania];
        //__syncthreads();

    }
    __syncthreads();
    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {
        Macierz_wynikowa_posrednia[index2dLatki] = 0;
        __syncthreads();
        for (int k = 0; k < ROZMIAR_LATKI; k++)

        {
            Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += sz_Const_macierz_wspolczynnikow2d_1[(threadIdx.y * (ROZMIAR_LATKI+1) + k)] * latka_referencyjna[k * ROZMIAR_LATKI + threadIdx.x];
            //Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += sz_Const_macierz_wspolczynnikow2d_1[threadIdx.y ][k] * latka_referencyjna[k * ROZMIAR_LATKI + threadIdx.x];
            //__syncthreads();
        }
        __syncthreads();
        latka_referencyjna[index2dLatki] = 0;
        __syncthreads();
        for (int k = 0; k < ROZMIAR_LATKI; k++)

        {
            latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * sz_Const_macierz_wspolczynnikow2d_2[(k * (ROZMIAR_LATKI+1) + threadIdx.x)];
            //latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * sz_Const_macierz_wspolczynnikow2d_2[k] [threadIdx.x];
            //__syncthreads();
        }
    }
    __syncthreads();
    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {
        Macierz_wynikowa_posrednia[index2dLatki] = 0;
        for (int k = 0; k < ROZMIAR_LATKI; k++)

        {
            Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += sz_Const_macierz_wspolczynnikow2d_1[threadIdx.y * (ROZMIAR_LATKI+1) + k] * latka_porownywana[k * ROZMIAR_LATKI + threadIdx.x];
            //Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += sz_Const_macierz_wspolczynnikow2d_1[threadIdx.y] [k ] * latka_porownywana[k * ROZMIAR_LATKI + threadIdx.x];
            //__syncthreads();
        }
        __syncthreads();
        latka_porownywana[index2dLatki] = 0;
        __syncthreads();
        for (int k = 0; k < ROZMIAR_LATKI; k++)
        {
            latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * sz_Const_macierz_wspolczynnikow2d_2[k * (ROZMIAR_LATKI+1) + threadIdx.x];
            //latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * sz_Const_macierz_wspolczynnikow2d_2[k][threadIdx.x];
            //__syncthreads();
        }
    }
    __syncthreads();
    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {
        
        if (fabs(latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x])
            < LAMBDA2D_HARD * sigma)
        {
            latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
        }
        if (fabs(latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x])
            < LAMBDA2D_HARD * sigma)
        {
            latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = 0;
        }
        
        //latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = fmaxf((latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] - LAMBDA2D_HARD * sigma),0);
        //latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = fmaxf((latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] -LAMBDA2D_HARD * sigma),0);
    }
    __syncthreads();

    if ((threadIdx.y < 8) && (threadIdx.x < 8))
    {
        float zmienna_sumowana = ((latka_referencyjna[index2dLatki] - latka_porownywana[index2dLatki]) * (latka_referencyjna[index2dLatki] - latka_porownywana[index2dLatki]));
        latka_referencyjna[index2dLatki] = zmienna_sumowana;

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            float suma = 0;
            for (int i = 0; i < 64; i++)//atomic add dla floatow jest wolniejsze
            {
                suma = suma + latka_referencyjna[i];
            }
            dev_koordynatySOA.MSE[index_elmentu_zero_tablicy_koordynat + (blockIdx.x + blockIdx.y * ROZMIAR_OBSZARU_PRZESZUKANIA)] = suma / POWIERZCHNIA_LATKI;
            dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + blockIdx.x + blockIdx.y * ROZMIAR_OBSZARU_PRZESZUKANIA] = blockIdx.x;
            dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + blockIdx.x + blockIdx.y * ROZMIAR_OBSZARU_PRZESZUKANIA] = blockIdx.y;
        }
    }
}

__global__ void Przepisywacz_do_tabloc_transformaty(float* __restrict__ obrazek_przepisywany, Tablice_koordynatLatek tablica_koordynat_latek_SOA, int* tablica_ilosci_pasujacych_latek, float* tablice_transformaty, int ilosc_blokow_w_boku_x, int ilosc_blokow_w_boku_y, int szerokosc, int i, int j, int mnoznik_tablicy_transormat)
{
    int row_pos = threadIdx.y;
    int col_pos = threadIdx.x;
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int index_elmentu_zero_tablicy_transformat = (blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat);

    __shared__ float obszar_preszukana_shared[(ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI) * (ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI)];
    int ofset = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5;
    if ((row_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5) && (col_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5))
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                obszar_preszukana_shared[(row_pos + i * ofset) * (RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA)+col_pos + (j * ofset)] = obrazek_przepisywany[((row_pos + i * ofset) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + col_pos + (j * ofset) + index_x_pixela_gorny_lewy_obszaru_przeszukania];
                //przpisujemy obszar preszukania (40 pixeli) dla latki do pamieci dzielonej bloku, ze wzgledu na zmieszczenie sie w dostepnej w wywolaniu funkcji iosci watkow musialem zrealizowac przypisanie w czterech krokach.
            }
        }
    }
    __syncthreads();

    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        for (int i = 0; i < tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//latka po latce przepsisujemy latki z obszaru przeszukania do tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32_
        {
            {
                int indeks_pomocniczy1 = col_pos + (row_pos)*ROZMIAR_LATKI + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
                int indeks_pomocniczy2 = (tablica_koordynat_latek_SOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i]) + col_pos + (tablica_koordynat_latek_SOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + row_pos) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA;
                tablice_transformaty[index_elmentu_zero_tablicy_transformat + indeks_pomocniczy1] = obszar_preszukana_shared[indeks_pomocniczy2];
            }
        }
    }
    //__syncthreads();
}
__global__ void Przepisywacz_z_tablic_transformaty_1krok(int* tablica_ilosci_zerowan, float* obrazek_po_kolejnym_kroku, float* obrazek_po_kolejnym_kroku_dzielnik, Tablice_koordynatLatek dev_koordynatySOA, int* device_tablica_ilosci_pasujacych_latek, float* device_tablice_transformaty_32_1krok, int ilosc_blokow_w_boku_x, int ilosc_blokow_w_boku_y, int szerokosc, int i, int j, int mnoznik_tablicy_transormat)
{
    if (tablica_ilosci_zerowan[blockIdx.z] < 1)
    {
        tablica_ilosci_zerowan[blockIdx.z] = 1;
    }
    int row_pos = threadIdx.y;
    int col_pos = threadIdx.x;
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;//przetestowac czy blo z czy y czy jeden i drugi!!!!!!!
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index_2d_latki = col_pos + (row_pos * ROZMIAR_LATKI);
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int index_elmentu_zero_tablicy_transformat = (blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat);
    float ilosc_niewyzerowanych = tablica_ilosci_zerowan[blockIdx.z];
    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        //sz_Macierz_wspolczynnikow_Kaizerra[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Macierz_wspolczynnikow_Kaizerra[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        __syncthreads();
        
        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//latka po latce przepsisujemy latki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
        {
            int indeks_pomocniczy1_odkladanie_latek = index_2d_latki + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
            int indeks_pomocniczy2_odkladanie_latek = ((dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i] + index_x_pixela_gorny_lewy_obszaru_przeszukania) + col_pos) + ((dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + index_y_pixela_gorny_lewy_obszaru_przeszukania + row_pos) * szerokosc);
            obrazek_po_kolejnym_kroku[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ((device_tablice_transformaty_32_1krok[index_elmentu_zero_tablicy_transformat + indeks_pomocniczy1_odkladanie_latek]) / ilosc_niewyzerowanych));
            obrazek_po_kolejnym_kroku_dzielnik[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] / (ilosc_niewyzerowanych));
            __syncthreads();
        }
        __syncthreads();
    }
}

__global__ void Przepisywacz_z_tablic_transformaty_2krok(float* device_tablica_wartosci_fitru_wiena, float* obrazek_po_kolejnym_kroku, float* obrazek_po_kolejnym_kroku_dzielnik, int sigma, Tablice_koordynatLatek dev_koordynatySOA, int* device_tablica_ilosci_pasujacych_latek, float* device_tablice_transformaty_32_2krok, int ilosc_blokow_w_boku_x, int ilosc_blokow_w_boku_y, int szerokosc, int i, int j, int mnoznik_tablicy_transormat)
{
    int row_pos = threadIdx.y;
    int col_pos = threadIdx.x;
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index_2d_latki = col_pos + (row_pos * ROZMIAR_LATKI);
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int index_elmentu_zero_tablicy_transformat = (blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat);
    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//latka po latce przepsisujemy latki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
        {
            int indeks_pomocniczy1_odkladanie_latek = index_2d_latki + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
            int indeks_pomocniczy2_odkladanie_latek = ((dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i] + index_x_pixela_gorny_lewy_obszaru_przeszukania) + col_pos) + ((dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + index_y_pixela_gorny_lewy_obszaru_przeszukania + row_pos) * szerokosc);
            obrazek_po_kolejnym_kroku[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ((device_tablice_transformaty_32_2krok[index_elmentu_zero_tablicy_transformat + indeks_pomocniczy1_odkladanie_latek])) / device_tablica_ilosci_pasujacych_latek[blockIdx.z]);
            obrazek_po_kolejnym_kroku_dzielnik[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] / device_tablica_ilosci_pasujacych_latek[blockIdx.z]);
            __syncthreads();
        }
        __syncthreads();
    }
}


__global__ void Zerowanie(float* device_tablice_transformaty_32_1krok, int* device_tablica_ilosci_zerowan, int* device_tablica_ilosci_pasujacych_latek, float sigma, int mnoznik_tablicy_transormat)
{
    device_tablica_ilosci_zerowan[blockIdx.z] = 0;
    int indeks = blockIdx.x * blockDim.x + threadIdx.x;
    int przesuniecie = blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat;
    int ilosc_pasujacych_latek = device_tablica_ilosci_pasujacych_latek[blockIdx.z];
    if (indeks < 1)
    {
        device_tablica_ilosci_zerowan[blockIdx.z] = 0;
    }
    __syncthreads();

    if (indeks < ilosc_pasujacych_latek * ROZMIAR_LATKI * ROZMIAR_LATKI)
    {
        if (abs(device_tablice_transformaty_32_1krok[indeks + przesuniecie]) < (LAMBDA3D_HARD * sigma))

        {
            device_tablice_transformaty_32_1krok[indeks + przesuniecie] = 0.0f;
        }
        else
        {
            atomicAdd(&device_tablica_ilosci_zerowan[blockIdx.z], 1);
        }
    }
    
}

__global__ void Filtr_Wiena(float* device_tablica_wartosci_fitru_wiena, float* device_tablice_transformaty_32_1krok, float* device_tablice_transformaty_32_2krok, int* device_tablica_ilosci_pasujacych_latek, float sigma, int mnoznik_tablicy_transormat)
{
    int przesuniecie = blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat;
    float coef = 1.0f / (float)device_tablica_ilosci_pasujacych_latek[blockIdx.z];
    int indeks1 = blockIdx.x * blockDim.x + threadIdx.x;
    int indeks = blockIdx.x * blockDim.x + threadIdx.x + przesuniecie;
    if (indeks1 < 1)
    {
        device_tablica_wartosci_fitru_wiena[blockIdx.z] = 0.0;
    }
    //__syncthreads();

    if (device_tablica_ilosci_pasujacych_latek[blockIdx.z] > 0)
    {
        if (indeks1 < device_tablica_ilosci_pasujacych_latek[blockIdx.z] * ROZMIAR_LATKI * ROZMIAR_LATKI)
        {
            float x; //wartosc posrednia obliczen wspolczynnika filtracji wiena


            x = (device_tablice_transformaty_32_2krok[indeks] * device_tablice_transformaty_32_2krok[indeks]);
            float wspolczynnik_filtracji_wiena = x / (x + (float)(sigma * sigma));
            device_tablice_transformaty_32_2krok[indeks] = ((device_tablice_transformaty_32_1krok[indeks] * wspolczynnik_filtracji_wiena));
            //__syncthreads();
            //device_tablice_transformaty_32_1krok[indeks] = wspolczynnik_filtracji_wiena;
        }
    }

    //__syncthreads();

    if (indeks1 < 1)
    {
        device_tablica_wartosci_fitru_wiena[blockIdx.z] = 1.0 / device_tablica_ilosci_pasujacych_latek[blockIdx.z];
    }

    /*
   if (indeks1 < device_tablica_ilosci_pasujacych_latek[blockIdx.z] * ROZMIAR_LATKI * ROZMIAR_LATKI)
      {
      atomicAdd(&device_tablica_wartosci_fitru_wiena[blockIdx.z], device_tablice_transformaty_32_1krok[indeks]);
     }

   __syncthreads();

    if (indeks1 < 1)
    {
         if (device_tablica_wartosci_fitru_wiena[blockIdx.z] > 1.0f)
        {
             device_tablica_wartosci_fitru_wiena[blockIdx.z] = (1.0 / device_tablica_wartosci_fitru_wiena[blockIdx.z]) * coef;
             //device_tablica_wartosci_fitru_wiena[blockIdx.z] =  (1.0/ device_tablica_wartosci_fitru_wiena[blockIdx.z] * device_tablica_ilosci_pasujacych_latek[blockIdx.z]);
        }
        else
        {
            device_tablica_wartosci_fitru_wiena[blockIdx.z] = 1;
        }
        if (device_tablica_ilosci_pasujacych_latek[blockIdx.z] == 1)
        {
            device_tablica_wartosci_fitru_wiena[blockIdx.z] = 1.0 / (device_tablica_wartosci_fitru_wiena[blockIdx.z] * N_WIEN);
        }

    }
    */

}

__global__ void Nadpisywanie_marginesow1(float* device_obrazek_po1kroku, float* device_obrazek_po1kroku_dzielnik, int szerokosc, int wysokosc, int margines_lewy, int margines_prawy) //doanaie nowych marginesow
{

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int margines = margines_lewy + margines_prawy;

    if (row_pos < wysokosc && col_pos < margines_lewy)

    {
        device_obrazek_po1kroku[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku[(margines_lewy + (margines_lewy - col_pos)) + (row_pos * szerokosc)];
    }

    if (row_pos < wysokosc && col_pos < szerokosc)
    {
        if (col_pos > (szerokosc - margines_prawy))

        {
            device_obrazek_po1kroku[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku[(szerokosc - margines_prawy - (col_pos - (szerokosc - margines_prawy))) + (row_pos * szerokosc)]; //-szerokosc + margines            
        }
    }

}

__global__ void Nadpisywanie_marginesow2(float* device_obrazek_po1kroku, float* device_obrazek_po1kroku_dzielnik, int szerokosc, int wysokosc, int margines_lewy, int margines_prawy) //dzielenie wyiku sumowania zerowanych latek zprzez ilosc zerowañ oraz doanaie nowych marginesow
{

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int margines = margines_lewy + margines_prawy;

    if (row_pos < margines_lewy && col_pos < szerokosc)

    {
        float x = device_obrazek_po1kroku[col_pos + (margines_lewy + (margines_lewy - row_pos)) * szerokosc];
        device_obrazek_po1kroku[col_pos + row_pos * szerokosc] = x;
    }

    if (row_pos < wysokosc && col_pos < szerokosc)
    {
        if (row_pos > wysokosc - margines_prawy)

        {
            device_obrazek_po1kroku[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku[(col_pos + (wysokosc - margines_prawy - (row_pos - (wysokosc - margines_prawy))) * szerokosc)];

        }
    }
}



__global__ void DzielenieMacierzy(float* device_obrazek_po_n_kroku, float* __restrict__ device_obrazek_po_n_kroku_dzielnik, int szerokosc, int wysokosc, int margines_lewy, int margines_prawy) //dzielenie wyiku sumowania zerowanych latek zprzez ilosc zerowañ
{

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;


    int index2d_pixela = col_pos + row_pos * szerokosc;
    int margines = margines_lewy + margines_prawy;

    if (row_pos < wysokosc && col_pos < szerokosc)
    {
        device_obrazek_po_n_kroku[index2d_pixela] = device_obrazek_po_n_kroku[index2d_pixela] / device_obrazek_po_n_kroku_dzielnik[index2d_pixela];
    }
}



void initializeCUDA(int argc, char** argv, int& devID)
{
    //funkcja na podstawie gotowego kodu udostepnionego na stronie :
    //https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/matrixMulCUBLAS/matrixMulCUBLAS.cpp
    //linie 149 - 178
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
    double sigma = sigm; // Wartoœc sigma dla szumu gaussowskiego
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
                    int nowa_warosc = cv::saturate_cast<uchar>(pixele[c] + szum);
                    pixele[c] = nowa_warosc;
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    
    int devID = 0;
    initializeCUDA(argc, argv, devID);
    cudaDeviceReset();

    float sigma = SIGMA;
    cv::Mat ObrazekSzary;
    cv::Mat ObrazekReferencyjny;
    std::string nazwa_pilku_zaszumionego;
    std::string nazwa_pilku_referencyjnego;
    std::string nazwa_sciezki = "obrazki_testowe/";
    int p_hard = P_HARD; //przesyniecie pomidzy latkami w kroku 1, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
    int p_wien = P_WIEN;//przesuniecie pomiedzy latkami w kroku 2, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
    int zakladka_obszaru_przeszukania = 2;
    int szybkosc = 1;
    int opcja_obrazka;   
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
        std::cout << "prosze podaj poziom szumu" << std::endl;
        std::cin >> sigma;
    }

    else if (opcja_obrazka == 2)
    {
        std::cout << "prosze wpisac nazwe pliku obrazka referencyjnego wraz z rozszerzeniem" << std::endl;
        std::cin >> nazwa_pilku_referencyjnego;
        //nazwa_pilku_referencyjnego = nazwa_sciezki + nazwa_pilku_referencyjnego;
        ObrazekReferencyjny = cv::imread(nazwa_pilku_referencyjnego, cv::IMREAD_GRAYSCALE);
        std::cout << "prosze podaj poziom szumu" << std::endl;
        std::cin >> sigma;
        ObrazekSzary = ObrazekReferencyjny.clone(); // Skopiuj obraz do macierzy z szumem
        dodanie_szumu(ObrazekSzary, sigma, 1);
    }

    do
    {
        std::cout << "Tryb 'normalny' czy 'szybki'?" << std::endl;
        std::cout << "1) WOLNY -przesuniecie pomiedzy latkami wynosi 1, najwyzsza jakoœc (ale œladowa roznica w stosunku do NORMALNEGO), bardzo wolny" << std::endl;
        std::cout << "2) NORMALNY - przesuniecie pomiedzy latkami wynosi 3" << std::endl;
        std::cout << "3) SZYBKI -przesuniecie pomiedzy latkami wynosi 4, powoduje to bardzo niewielkie pogorszenie jakosci" << std::endl;
        std::cout << "4) NAJSZYBSZY -przesuniecie pomiedzy latkami wynosi 5, powoduje to niewielkie pogorszenie jakosci" << std::endl;
        std::cin >> szybkosc;
    } while (opcja_obrazka != 1 && opcja_obrazka != 2);

    if (szybkosc == 1)
    {
        p_hard = 1;
        p_wien = 1;
        zakladka_obszaru_przeszukania = 1;
    }

    else if (szybkosc == 3)
    {
        p_hard = 4;
        p_wien = 4;
        zakladka_obszaru_przeszukania = 5;
    }
    else if (szybkosc == 4)
    {
        p_hard = 5;
        p_wien = 5;
        zakladka_obszaru_przeszukania = 6;
    }
    else
    {
        int p_hard = P_HARD; //przesyniecie pomidzy latkami w kroku 1, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
        int p_wien = P_WIEN;//przesuniecie pomiedzy latkami w kroku 2, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
        int zakladka_obszaru_przeszukania = 2;
    }

        /////////////////////////////////////////////////////////////////////////////

  
    int szerokosc_obrazka_oryginalnego = ObrazekSzary.cols;
    int wysokosc_obrazka_oryginalnego = ObrazekSzary.rows;
    int wielkosc_marginesu_lewego = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2;
    int wielkosc_marginesu_prawego = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA;

    cv::copyMakeBorder(ObrazekSzary, ObrazekSzary, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego, cv::BORDER_REFLECT_101);
    ObrazekSzary.convertTo(ObrazekSzary, CV_32F);
    int szerokosc_obrazka_z_marginesami = szerokosc_obrazka_oryginalnego + wielkosc_marginesu_lewego + wielkosc_marginesu_prawego;
    int wysokosc_obrazka_z_marginesami = wysokosc_obrazka_oryginalnego + wielkosc_marginesu_lewego + wielkosc_marginesu_prawego;
    int wielkosc_tablicy_z_marginesami = szerokosc_obrazka_z_marginesami * wysokosc_obrazka_z_marginesami;

    ObrazekSzary = ObrazekSzary.reshape(1, wielkosc_tablicy_z_marginesami);
    //////////////////////////////////////////////////////////////////tworzymy zmienne do przekazania do Kernelu starowego///////////////////////////
    
    float* host_obrazek_poczatkowy = new float[wielkosc_tablicy_z_marginesami];
    float* host_obrazek_po1kroku = new float[wielkosc_tablicy_z_marginesami];
    float* host_obrazek_po2kroku = new float[wielkosc_tablicy_z_marginesami];
    float* host_obrazek_po1kroku_dzielnik = new float[wielkosc_tablicy_z_marginesami];
    
    int ilosc_blokow_w_boku_x = (int)std::ceil(((double)szerokosc_obrazka_oryginalnego / (ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI)));
    int ilosc_blokow_w_boku_y = (int)ceil(((double)wysokosc_obrazka_oryginalnego) / (ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI));
    int ilosc_blokow = ilosc_blokow_w_boku_x * ilosc_blokow_w_boku_y;
    
    int wielkosc_tablicy_transformaty_32 = ilosc_blokow * ROZMIAR_LATKI * ROZMIAR_LATKI * N_WIEN;
    int wielkosc_tablicy_transformaty_16 = ilosc_blokow * ROZMIAR_LATKI * ROZMIAR_LATKI * N_HARD;
    int wielkosc_tablicy_ilosci_pasujacych_latek = ilosc_blokow;
    int wielkosc_tablicy_zerowan = ilosc_blokow;
    int wielkosc_tablicy_koordynat = ilosc_blokow * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA;
    
    
    host_obrazek_poczatkowy = (float*)ObrazekSzary.data;
    
    float* device_obrazek_poczatkowy;
    float* device_obrazek_po1kroku;
    float* device_obrazek_po2kroku;
    float* device_obrazek_po1kroku_dzielnik;
    float* device_obrazek_po2kroku_dzielnik;
    int start_1 = clock();
    //cudaFree(0);
    cudaMalloc((void**)&device_obrazek_poczatkowy, wielkosc_tablicy_z_marginesami * sizeof(float));
    int stop_1 = clock();
    cudaMalloc((void**)&device_obrazek_po1kroku, wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((void**)&device_obrazek_po2kroku, wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((void**)&device_obrazek_po1kroku_dzielnik, wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((void**)&device_obrazek_po2kroku_dzielnik, wielkosc_tablicy_z_marginesami * sizeof(float));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////przygotowanie i lokowanie w pamieci tablic pomocniczych////////////////////////////////
    /////////lokujemy je w pamieci przed rozpoczeciem wykonywania programu przez karte gdyz dynamiczne lokowanie pamieci przez CUDE wielokrotnie spowalnia program///////////////


    int rozmiar_w_pamieci_tablic_koordynat_inty = sizeof(int) * wielkosc_tablicy_koordynat;
    int rozmiar_w_pamieci_tablic_koordynat_floaty = sizeof(float) * wielkosc_tablicy_koordynat;
    Tablice_koordynatLatek dev_koordynatySOA;
    Tablice_koordynatLatek* wskaznik_dev_koordynatySOA = &dev_koordynatySOA;


    cudaMalloc((&wskaznik_dev_koordynatySOA->MSE), rozmiar_w_pamieci_tablic_koordynat_floaty);
    cudaMalloc(&(wskaznik_dev_koordynatySOA->koordynata_x), rozmiar_w_pamieci_tablic_koordynat_inty);
    cudaMalloc(&(wskaznik_dev_koordynatySOA->koordynata_y), rozmiar_w_pamieci_tablic_koordynat_inty);

    float* device_tablice_transformaty_16;
    float* device_tablice_transformaty_32_1krok;
    float* device_tablice_transformaty_32_2krok;
    int* device_tablica_ilosci_pasujacych_latek;
    //int* device_tablica_do_najmniejszych;
    int* device_tablica_ilosci_zerowan;
    float* device_tablica_wartosci_fitru_wiena;
    cudaMalloc((void**)&device_tablice_transformaty_16, wielkosc_tablicy_transformaty_16 * sizeof(float));
    cudaMalloc((void**)&device_tablice_transformaty_32_1krok, wielkosc_tablicy_transformaty_32 * sizeof(float));
    cudaMalloc((void**)&device_tablice_transformaty_32_2krok, wielkosc_tablicy_transformaty_32 * sizeof(float));
    cudaMalloc(&device_tablica_ilosci_pasujacych_latek, wielkosc_tablicy_ilosci_pasujacych_latek * sizeof(int));
    cudaMalloc((void**)&device_tablica_ilosci_zerowan, wielkosc_tablicy_zerowan * sizeof(int));
    cudaMalloc((void**)&device_tablica_wartosci_fitru_wiena, wielkosc_tablicy_zerowan * sizeof(float));
    //cudaMalloc(&device_tablica_do_najmniejszych, ilosc_blokow_w_bloku_x * ilosc_blokow_w_bloku_y * N_WIEN * sizeof(int));
    int start = clock();
    //cudaFuncSetCacheConfig(Kalkulator_MSE_szum_duzy, cudaFuncCachePreferShared);
    //cudaFuncSetCacheConfig(DCT, cudaFuncCachePreferShared);
    //cudaFuncSetCacheConfig(DCT_odwrotna, cudaFuncCachePreferShared);
   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int start_3 = clock();
    cudaMemcpy(device_obrazek_poczatkowy, host_obrazek_poczatkowy, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    int x = 11;
    int ilosc_latek_w_obszarze_przeszukania = (ROZMIAR_OBSZARU_PRZESZUKANIA) * (ROZMIAR_OBSZARU_PRZESZUKANIA);
    dim3 bloki_Kalkulator_MSE(1, 1, ilosc_blokow);
    dim3 watki_Kalkulator_MSE(ROZMIAR_OBSZARU_PRZESZUKANIA, ROZMIAR_OBSZARU_PRZESZUKANIA, 1);
    dim3 bloki_Kalkulator_MSE_szum_duzy(ROZMIAR_OBSZARU_PRZESZUKANIA, ROZMIAR_OBSZARU_PRZESZUKANIA, ilosc_blokow);
    dim3 watki_Kalkulator_MSE_szum_duzy(ROZMIAR_LATKI, ROZMIAR_LATKI, 1);
    dim3 bloki_najmniejsze_liczby(1, 1, ilosc_blokow);
    dim3 watki_najmniejsze_liczby(ilosc_latek_w_obszarze_przeszukania / 2, 1, 1);
    dim3 watki_Przepisywacz(ROZMIAR_LATKI, ROZMIAR_LATKI, 1);
    dim3 bloki_Przepisywacz(1, 1, ilosc_blokow);
    dim3 bloki_DCT_krok1(1, 1, ilosc_blokow* N_HARD);
    dim3 watki_DCT_krok1(ROZMIAR_LATKI, ROZMIAR_LATKI, 1);
    dim3 bloki_DCT_krok2(1, 1, ilosc_blokow* N_WIEN);
    dim3 watki_DCT_krok2(ROZMIAR_LATKI, ROZMIAR_LATKI, 1);
    dim3 bloki_Walsh(1, 1, ilosc_blokow);
    dim3 watki_Walsh_krok1(ROZMIAR_LATKI* ROZMIAR_LATKI, N_HARD / 2, 1);
    dim3 watki_Walsh_krok2(ROZMIAR_LATKI* ROZMIAR_LATKI, N_WIEN / 2, 1);

    dim3 bloki_Zerowanie(1, 1, ilosc_blokow);
    dim3 watki_Zerowanie(N_HARD* ROZMIAR_LATKI* ROZMIAR_LATKI, 1, 1);
    dim3 bloki_Wien(N_WIEN, 1, ilosc_blokow);
    dim3 watki_Wien(ROZMIAR_LATKI* ROZMIAR_LATKI, 1, 1);
    int dzielenie_macierzy_watki_x = 32;
    int dzielenie_macierzy_watki_y = 32;
    int dzielenie_macierzy_bloki_x = (szerokosc_obrazka_z_marginesami + dzielenie_macierzy_watki_x) / dzielenie_macierzy_watki_x;
    int dzielenie_macierzy_bloki_y = (wysokosc_obrazka_z_marginesami + dzielenie_macierzy_watki_y) / dzielenie_macierzy_watki_y;
    dim3 bloki_dzielnie_Macierzy(dzielenie_macierzy_bloki_x, dzielenie_macierzy_bloki_y, 1);
    dim3 watki_dzielnie_Macierzy(dzielenie_macierzy_watki_x, dzielenie_macierzy_watki_y, 1);
    for (int i = 0; i < (ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania); i += p_hard)
    {
        for (int j = 0; j < (ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania); j += p_hard)
        {
            //int wielkosc_tabeli_koordynat = (ROZMIAR_OBSZARU_PRZESZUKANIA) * (ROZMIAR_OBSZARU_PRZESZUKANIA);
            int szerokosc = szerokosc_obrazka_z_marginesami;
            int tau_hard= TAU_HARD_NISKI;
            if (sigma > 40)
            {
                tau_hard = TAU_HARD_WYSOKI*3;
               
                Kalkulator_MSE_szum_duzy << <bloki_Kalkulator_MSE_szum_duzy, watki_Kalkulator_MSE_szum_duzy >> > (device_obrazek_poczatkowy, dev_koordynatySOA, ilosc_blokow_w_boku_x, szerokosc, i, j, sigma);
                
                //Kalkulator_MSE << <bloki_Kalkulator_MSE, watki_Kalkulator_MSE >> > (device_obrazek_poczatkowy, dev_koordynatySOA, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j);

            }
            else
            {
                Kalkulator_MSE << <bloki_Kalkulator_MSE, watki_Kalkulator_MSE >> > (device_obrazek_poczatkowy, dev_koordynatySOA, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j);
            }

            ///////////////////////////////////////////////////////////wyszukanie N_HARD najblizszych latek////////////////////////////////////////////

            Najmniejsze_liczby << <bloki_najmniejsze_liczby, watki_najmniejsze_liczby >> > (dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, N_HARD, tau_hard, false);
            Przepisywacz_do_tabloc_transformaty << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_obrazek_poczatkowy, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_16, ilosc_blokow_w_boku_x,0, szerokosc, i, j, N_HARD);

            ////////////////////////////////////////////// pasujece latki znajdujace sie w "device_tablice_transformaty_32_1krok" (tylko tyle z tej tablicy ile spelnia warunek max dopasowania) poddajemy transformacie cosinusowej 2d (cale latki), a nastepnie transformacie 1D walsha-hadamarda "w poprzek" grupy latek////////////////////////////////////////
            DCT << <bloki_DCT_krok1, watki_DCT_krok1 >> > (device_tablice_transformaty_16, device_tablice_transformaty_32_2krok, ROZMIAR_LATKI, ROZMIAR_LATKI, device_tablica_ilosci_pasujacych_latek, N_HARD, false);
            Walsh1dPojedyncza << <bloki_Walsh, watki_Walsh_krok1 >> > (device_tablice_transformaty_16, device_tablice_transformaty_32_2krok, device_tablica_ilosci_pasujacych_latek, N_HARD, false);
            Zerowanie << <bloki_Zerowanie, watki_Zerowanie >> > (device_tablice_transformaty_16, device_tablica_ilosci_zerowan, device_tablica_ilosci_pasujacych_latek, sigma, N_HARD);

            ////////////////////////////////////////////////////////////// Odwracamy transformaty w celu uzyskania wlaœciwego obrazu//////////////////////////////////////////////////////

            Walsh1dPojedyncza << <bloki_Walsh, watki_Walsh_krok1 >> > (device_tablice_transformaty_16, device_tablice_transformaty_32_2krok, device_tablica_ilosci_pasujacych_latek, N_HARD, false);
            DCT_odwrotna << <bloki_DCT_krok1, watki_DCT_krok1 >> > (device_tablice_transformaty_16, ROZMIAR_LATKI, ROZMIAR_LATKI, device_tablica_ilosci_pasujacych_latek, N_HARD);
            ///// //////////////////////////teraz trzeba poodkladac llatki w odpowiednie miejsca tablicy wynikowej po 1 kroku, oraz pododawac wartoœci iliœci niewyzerowanych w jej dzielniku
            Przepisywacz_z_tablic_transformaty_1krok << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_tablica_ilosci_zerowan, device_obrazek_po1kroku, device_obrazek_po1kroku_dzielnik, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_16, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j, N_HARD);
        }
    }

    DzielenieMacierzy << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (device_obrazek_po1kroku, device_obrazek_po1kroku_dzielnik, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego);

    Nadpisywanie_marginesow1 << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (device_obrazek_po1kroku, device_obrazek_po1kroku_dzielnik, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego);
    Nadpisywanie_marginesow2 << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (device_obrazek_po1kroku, device_obrazek_po1kroku_dzielnik, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego);
    for (int i = 0; i < (ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania); i += p_wien)
    {
        for (int j = 0; j < (ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania); j += p_wien)
        {

            //int wielkosc_tabeli_koordynat = (ROZMIAR_OBSZARU_PRZESZUKANIA) * (ROZMIAR_OBSZARU_PRZESZUKANIA);
            int szerokosc = szerokosc_obrazka_z_marginesami;
            int tau_wien;
            if (sigma < 40)
            {
                tau_wien = 1000;
            }
            else
                tau_wien = 3000;

            Kalkulator_MSE << <bloki_Kalkulator_MSE, watki_Kalkulator_MSE >> > (device_obrazek_po1kroku, dev_koordynatySOA, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j);

              ///////////////////////////////////////////////////////////wyszukanie N_WIEN najblizszych latek////////////////////////////////////////////

            Najmniejsze_liczby << <bloki_najmniejsze_liczby, watki_najmniejsze_liczby >> > (dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, N_WIEN, tau_wien, true);
            //przepisujemy latki z tablicy reprezentujacej obrazek wejœciowego do "device_tablice_transformaty_32_1krok": 
            Przepisywacz_do_tabloc_transformaty << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_obrazek_poczatkowy, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_32_1krok, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j, N_WIEN);
            //przepisujemy latki z tablicy repezentujacej obrazek wstepnie odszumiony w 1 kroku do device_tablice_transformaty_32_2krok
            Przepisywacz_do_tabloc_transformaty << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_obrazek_po1kroku, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_32_2krok, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j, N_WIEN);
            ////////////////////////////////////////////// pasujece latki znajdujace sie w "device_tablice_transformaty_32_1krok" (tylko tyle z tej tablicy ile spelnia warunek max dopasowania) poddajemy transformacie cosinusowej 2d (cale latki), a nastepnie transformacie 1D walsha-hadamarda "w poprzek" grupy latek////////////////////////////////////////
            DCT << <bloki_DCT_krok2, watki_DCT_krok2 >> > (device_tablice_transformaty_32_1krok, device_tablice_transformaty_32_2krok, ROZMIAR_LATKI, ROZMIAR_LATKI, device_tablica_ilosci_pasujacych_latek, N_WIEN, true);
            Walsh1dPojedyncza << <bloki_Walsh, watki_Walsh_krok2 >> > (device_tablice_transformaty_32_1krok, device_tablice_transformaty_32_2krok, device_tablica_ilosci_pasujacych_latek, N_WIEN, true); // przesuniecie to indeks elentu zerowego w macierzy transformat dla danego wywolania kernela                                                                                                                      
             /////////////////////////////////////////////////////////////// W przeksztalconych latkach zerujemy wspolczynniki ktorych abs jest mmniejszy niz Lambda_Hard_3D*SIGMA/////////////////////////

            Filtr_Wiena << <bloki_Wien, watki_Wien >> > (device_tablica_wartosci_fitru_wiena, device_tablice_transformaty_32_1krok, device_tablice_transformaty_32_2krok, device_tablica_ilosci_pasujacych_latek, sigma, N_WIEN);
            Walsh1dPojedyncza << <bloki_Walsh, watki_Walsh_krok2 >> > (device_tablice_transformaty_32_2krok, device_tablice_transformaty_32_1krok, device_tablica_ilosci_pasujacych_latek, N_WIEN, false);
            cudaFuncSetCacheConfig(DCT_odwrotna, cudaFuncCachePreferShared);
            DCT_odwrotna << <bloki_DCT_krok2, watki_DCT_krok2 >> > (device_tablice_transformaty_32_2krok, ROZMIAR_LATKI, ROZMIAR_LATKI, device_tablica_ilosci_pasujacych_latek, N_WIEN);
  
            Przepisywacz_z_tablic_transformaty_2krok << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_tablica_wartosci_fitru_wiena, device_obrazek_po2kroku, device_obrazek_po2kroku_dzielnik, sigma, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_32_2krok, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j, N_WIEN); 
        }
    }

    DzielenieMacierzy << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (device_obrazek_po2kroku, device_obrazek_po2kroku_dzielnik, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego);
    cudaMemcpy(host_obrazek_po2kroku, device_obrazek_po2kroku, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_obrazek_po1kroku, device_obrazek_po1kroku, wielkosc_tablicy_z_marginesami*sizeof(float), cudaMemcpyDeviceToHost);
    int stop_3 = clock();
    //int start_4 = clock();

    cv::Mat testDataMat(wysokosc_obrazka_z_marginesami, szerokosc_obrazka_z_marginesami, CV_32F, host_obrazek_po1kroku);
    cv::Mat testDataMat2(wysokosc_obrazka_z_marginesami, szerokosc_obrazka_z_marginesami, CV_32F, host_obrazek_po2kroku);
    testDataMat.convertTo(testDataMat, CV_8U);
    testDataMat2.convertTo(testDataMat2, CV_8U);
    testDataMat = testDataMat(cv::Rect(wielkosc_marginesu_lewego, wielkosc_marginesu_lewego, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));
    testDataMat2 = testDataMat2(cv::Rect(wielkosc_marginesu_lewego, wielkosc_marginesu_lewego, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));
    //int stop_4 = clock();

    double czas_3 = (double)(stop_3 - start_3) / (double)CLOCKS_PER_SEC;
    std::cout << std::endl << "Czas wykonania kernelu: " << czas_3 << " s" << std::endl;
    double R = 255;
    std::cout << "PSNR pierwszgo kroku w stosunky do brazka odszumionego wynosi: " << cv::PSNR(ObrazekReferencyjny, testDataMat, R) << std::endl;
    std::cout << "PSNR drugiego kroku w stosunky do brazka odszumionego wynosi: " << cv::PSNR(ObrazekReferencyjny, testDataMat2, R) << std::endl;

    ObrazekSzary = ObrazekSzary.reshape(0, wysokosc_obrazka_z_marginesami);
    ObrazekSzary.convertTo(ObrazekSzary, CV_8U);
    ObrazekSzary = ObrazekSzary(cv::Rect(wielkosc_marginesu_lewego, wielkosc_marginesu_lewego, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));

    cv::imshow("Obrazek zaszumiony", ObrazekSzary);
    cv::imshow("Obrazek referncyjny", ObrazekReferencyjny);
    cv::imshow("Obrazek po pierwszym kroku", testDataMat);
    cv::imshow("Obrazek po drugim kroku", testDataMat2);
    cv::waitKey(0);


    cudaFree(device_obrazek_poczatkowy);
    cudaFree(device_obrazek_po1kroku);
    cudaFree(device_tablice_transformaty_32_1krok);
    cudaFree(device_tablice_transformaty_16);
    cudaFree(device_tablice_transformaty_32_2krok);
    cudaFree(device_tablica_ilosci_pasujacych_latek);
    cudaFree(device_tablica_ilosci_zerowan);
    cudaFree(device_obrazek_po2kroku);
    cudaFree(device_tablica_wartosci_fitru_wiena);
    cudaFree(device_obrazek_po2kroku_dzielnik);
    cudaFree(device_obrazek_po1kroku_dzielnik);

    cudaFree(wskaznik_dev_koordynatySOA->MSE);
    cudaFree(wskaznik_dev_koordynatySOA->koordynata_x);
    cudaFree(wskaznik_dev_koordynatySOA->koordynata_y);
    wskaznik_dev_koordynatySOA->MSE = nullptr;
    wskaznik_dev_koordynatySOA->koordynata_x = nullptr;
    wskaznik_dev_koordynatySOA->koordynata_y = nullptr;
    //cudaFree(&dev_koordynatySOA);
    cudaFree(wskaznik_dev_koordynatySOA);


    delete[] host_obrazek_po2kroku;

    cudaDeviceReset();
    //getch();

    return 0;
}