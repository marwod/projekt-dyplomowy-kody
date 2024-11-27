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
#include<vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "naglowek_struktury.h"
#include "transformaty_cuda.cuh"
#include <random>
#include <cstdlib>
#include <filesystem> // musi byc C++17 lub wyzej



#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartoœæ w iloœci ³atek i u¿ywanych w¹tków. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru ³atki
#define ROZMIAR_LATKI       8
#define POWIERZCHNIA_LATKI       64
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_£ATKI iloœæ pixeli obszaru przeszukania
#define SIGMA       60.0f
#define LAMBDA2DHARD       0.9f
#define P_HARD  3 //p_Hard krok tworzenia latek, w oryginale 1,2 lub 3, u Lebruna 3
#define P_WIEN  3 //krok tworzenia latek, w oryginale 1, 2 lub 3, u Lebruna 3
#define N_HARD  16 //maks ilosc lek w grupie 3D
#define N_WIEN  32//maks ilosc lek w grupie 3D
#define TAU_HARD_NISKI 2500.0 //maksymalna odlegloœc MSE latki przysz szumie niskim
#define TAU_HARD_WYSOKI 5000.0 //maksymalna odlegloœc MSE latki przysz szumie niskim
#define LAMBDA3D_HARD 2.7  //LambdaHard2d	progowanie(trasholding) Grupy3d w pierwszym kroku filtra, u Lebruna 2,7
#define LAMBDA2D_HARD 0.5//Lambda_hard3d progowanie(trasholding) przy block matchingu, u Lebruna 2.0




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
0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975 };

__constant__ float aConst_macierz_wspolczynnikow2d_2[POWIERZCHNIA_LATKI] =
{ 0.3536, 0.4904, 0.4619, 0.4157, 0.3536, 0.2778, 0.1914, 0.0975,
  0.3536, 0.4157, 0.1914, -0.0975, -0.3536, -0.4904, -0.4619, -0.2778,
  0.3536, 0.2778, -0.1914, -0.4904, -0.3536, 0.0975, 0.4619, 0.4157,
  0.3536, 0.0976, -0.462, -0.2778, 0.3536, 0.4157, -0.1913, -0.4904,
  0.3536, -0.0976, -0.462, 0.2778, 0.3536, -0.4157, -0.1915, 0.4904,
  0.3536, -0.2778, -0.1914, 0.4904, -0.3535, -0.0977, 0.4620, -0.4157,
  0.3536, -0.4157, 0.1913, 0.0977, -0.3536, 0.4904, -0.4619, 0.2778,
  0.3536, -0.4904, 0.4619, -0.4157, 0.3534, -0.2778, 0.1911, -0.0975 };

__constant__ float testowa[64] =
{
1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 0.8, 1.0,
1.0, 0.5, 0.3, 0.0, 0.5, 1.0, 0.3, 0.0,
1.0, 0.3, 0.2, 0.0, 0.3, 1.0, 0.3, 0.2,
1.0, 0.2, 0.0, 0.0, 0.2, 1.0, 0.3, 0.2,
1.0, 0.3, 0.2, 0.0, 0.3, 1.0, 0.2, 0.0,
1.0, 0.8, 1.0, 1.0, 0.8, 1.0, 0.8, 1.0,
1.0, 0.3, 0.2, 0.0, 0.3, 1.0, 0.3, 0.2,
1.0, 0.2, 0.0, 0.0, 0.2, 1.0, 0.3, 0.2,
};




__global__ void Najmniejsze_liczby_(Tablice_koordynatLatek koordynatySOA, int* device_tablica_ilosci_pasujacych_latek, int ilosc_najmniejszych, float tau, bool krok2)
// wykorzystanie algorytmu redykcji u¿ywanego zwykle do sumowania tablicy
{
    int przesuniecie = blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA;
    __shared__ int s_tablica_indeksów_poczatkowych[1024];
    __shared__ float s_tablica_wartosci_MSE[1024];
    __shared__ int s_koordynaty_najmniejszych_SOA[N_WIEN];
    for (int i = 0; i < 2; i++)
    {
        if (threadIdx.x < 512)
        {
            s_tablica_wartosci_MSE[threadIdx.x + (i * 512)] = koordynatySOA.MSE[przesuniecie + threadIdx.x + (i * 512)];
            s_tablica_indeksów_poczatkowych[threadIdx.x + (i * 512)] = threadIdx.x + (i * 512);
        }
    }
    __syncthreads();


    for (int i = 0; i < ilosc_najmniejszych; i++)
    {
        for (int s = (ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA / 2); s > 0; s >>= 1)//s zmniejszamy dwukrotnie za ka¿d¹ iteracj¹
        {

            if ((threadIdx.x < s))
            {
                if (s_tablica_wartosci_MSE[s_tablica_indeksów_poczatkowych[threadIdx.x]] > s_tablica_wartosci_MSE[s_tablica_indeksów_poczatkowych[threadIdx.x + s]])
                {
                    s_tablica_indeksów_poczatkowych[threadIdx.x] = s_tablica_indeksów_poczatkowych[threadIdx.x + s];
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            s_koordynaty_najmniejszych_SOA[i] = s_tablica_indeksów_poczatkowych[0];
            s_tablica_wartosci_MSE[s_tablica_indeksów_poczatkowych[0]] = 10000000000000000000;
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
}

__global__ void Kalkulator_MSE_(float* __restrict__ Obrazek, Tablice_koordynatLatek dev_koordynatySOA, int ilosc_blokow_w_boku_x, int szerokosc, int i, int j)
{

    int row_pos = threadIdx.y;
    int col_pos = threadIdx.x;
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index2dObszaru = threadIdx.x + threadIdx.y * ROZMIAR_OBSZARU_PRZESZUKANIA;
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int offset = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2;

    __shared__ float latka_referencyjna[POWIERZCHNIA_LATKI];
    __shared__ float obszar_preszukana_shared[(ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI) * (ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI)];


    if ((col_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2) && (row_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2))
    {

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                obszar_preszukana_shared[(row_pos + i * offset) * (RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA)+col_pos + (j * offset)] = (Obrazek[((row_pos + i * offset) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + col_pos + (j * offset) + index_x_pixela_gorny_lewy_obszaru_przeszukania]);
            }
        }
    }
    __syncthreads();

    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        latka_referencyjna[row_pos * ROZMIAR_LATKI + col_pos] = obszar_preszukana_shared[(row_pos + offset - 4) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + (col_pos + offset - 4)];
    }
    __syncthreads();

    if ((row_pos < (ROZMIAR_OBSZARU_PRZESZUKANIA)) && (col_pos < (ROZMIAR_OBSZARU_PRZESZUKANIA)))
    {
        dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + index2dObszaru] = col_pos;
        dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + index2dObszaru] = row_pos;
        float MSE = 0;
        for (int i = 0; i < ROZMIAR_LATKI; i++)
        {
            for (int j = 0; j < ROZMIAR_LATKI; j++)
            {
                MSE += (((latka_referencyjna[i * ROZMIAR_LATKI + j] - obszar_preszukana_shared[(row_pos + i) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + col_pos + j])) * (latka_referencyjna[i * ROZMIAR_LATKI + j] - obszar_preszukana_shared[(row_pos + i) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + col_pos + j]));
            }
        }
        dev_koordynatySOA.MSE[index_elmentu_zero_tablicy_koordynat + index2dObszaru] = (MSE / POWIERZCHNIA_LATKI);
    }
}

__global__ void Kalkulator_MSE_szum_duzy(float* __restrict__ Obrazek, Tablice_koordynatLatek dev_koordynatySOA, int ilosc_blokow_w_boku_x, int szerokosc, int i, int j, int sigma)
{

    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index2d_wewnatrz_Latki = threadIdx.x + threadIdx.y * ROZMIAR_LATKI;
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int ofset = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2;

    __shared__ float latka_referencyjna[ROZMIAR_LATKI * ROZMIAR_LATKI];
    __shared__ float latka_porownywana[ROZMIAR_LATKI * ROZMIAR_LATKI];
    __shared__ float Macierz_wynikowa_posrednia[POWIERZCHNIA_LATKI];

    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {

        latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Obrazek[((threadIdx.y + ofset - 4) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + (ofset - 4) + index_x_pixela_gorny_lewy_obszaru_przeszukania];
        __syncthreads();

        latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Obrazek[((threadIdx.y + blockIdx.y) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + threadIdx.x + blockIdx.x + index_x_pixela_gorny_lewy_obszaru_przeszukania];
        __syncthreads();
    }

    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {
        Macierz_wynikowa_posrednia[index2d_wewnatrz_Latki] = 0;
        __syncthreads();
        for (int k = 0; k < ROZMIAR_LATKI; k++)

        {
            Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += aConst_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * latka_referencyjna[k * ROZMIAR_LATKI + threadIdx.x];
        }


        __syncthreads();
        latka_referencyjna[index2d_wewnatrz_Latki] = 0;
        __syncthreads();

        for (int k = 0; k < ROZMIAR_LATKI; k++)

        {
            latka_referencyjna[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * aConst_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];

        }
    }
    __syncthreads();
    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {
        Macierz_wynikowa_posrednia[index2d_wewnatrz_Latki] = 0;

        for (int k = 0; k < ROZMIAR_LATKI; k++)

        {
            Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += aConst_macierz_wspolczynnikow2d_1[threadIdx.y * ROZMIAR_LATKI + k] * latka_porownywana[k * ROZMIAR_LATKI + threadIdx.x];
        }
        __syncthreads();
        latka_porownywana[index2d_wewnatrz_Latki] = 0;
        __syncthreads();

        for (int k = 0; k < ROZMIAR_LATKI; k++)
        {
            latka_porownywana[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] += Macierz_wynikowa_posrednia[threadIdx.y * ROZMIAR_LATKI + k] * aConst_macierz_wspolczynnikow2d_2[k * ROZMIAR_LATKI + threadIdx.x];
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
    }
    __syncthreads();

    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {
        float zmienna_sumowana = ((latka_referencyjna[index2d_wewnatrz_Latki] - latka_porownywana[index2d_wewnatrz_Latki]) * (latka_referencyjna[index2d_wewnatrz_Latki] - latka_porownywana[index2d_wewnatrz_Latki]));

        latka_referencyjna[index2d_wewnatrz_Latki] = zmienna_sumowana;

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            float suma = 0;
            for (int i = 0; i < ROZMIAR_LATKI * ROZMIAR_LATKI; i++)//atomic add dla floatów jest wolniejsze
            {
                suma = suma + latka_referencyjna[i];
            }
            dev_koordynatySOA.MSE[index_elmentu_zero_tablicy_koordynat + (blockIdx.x + blockIdx.y * ROZMIAR_OBSZARU_PRZESZUKANIA)] = suma / POWIERZCHNIA_LATKI;
            dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + (blockIdx.x + blockIdx.y * ROZMIAR_OBSZARU_PRZESZUKANIA)] = blockIdx.x;
            dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + (blockIdx.x + blockIdx.y * ROZMIAR_OBSZARU_PRZESZUKANIA)] = blockIdx.y;
        }
    }
}


__global__ void Przepisywacz_do_tablic_transformaty_(Obrazek_YCrCb obrazek_przepisywany, Tablice_koordynatLatek tablica_koordynat_latek_SOA, int* tablica_ilosci_pasujacych_latek, Obrazek_YCrCb tablice_transformaty, int ilosc_blokow_w_boku_x, int szerokosc, int i, int j, int mnoznik_tablicy_transormat)
{


    int row_pos = threadIdx.y;
    int col_pos = threadIdx.x;
    int indeks_pomocniczy1;
    int indeks_pomocniczy2;
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int index_elmentu_zero_tablicy_transformat = (blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat);


    __shared__ float obszar_preszukana_shared[(ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI) * (ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI)];

    int offset = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5;
    if ((row_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5) && (col_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5))
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                obszar_preszukana_shared[(row_pos + i * offset) * (RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA)+col_pos + (j * offset)] = obrazek_przepisywany.kanal_Y[((row_pos + i * offset) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + col_pos + (j * offset) + index_x_pixela_gorny_lewy_obszaru_przeszukania];
            }
        }
    }
    __syncthreads();

    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        for (int i = 0; i < tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)
        {
            {
                int indeks_pomocniczy1 = col_pos + row_pos * ROZMIAR_LATKI + i * ROZMIAR_LATKI * ROZMIAR_LATKI + index_elmentu_zero_tablicy_transformat;
                int indeks_pomocniczy2 = (tablica_koordynat_latek_SOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i]) + col_pos + (tablica_koordynat_latek_SOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + row_pos) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA;
                tablice_transformaty.kanal_Y[indeks_pomocniczy1] = obszar_preszukana_shared[indeks_pomocniczy2];
            }
        }
    }
    __syncthreads();

    /////////////////////////////Kana³ G:
    if ((row_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5) && (col_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5))
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                obszar_preszukana_shared[(row_pos + i * offset) * (RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA)+col_pos + (j * offset)] = obrazek_przepisywany.kanal_Cr[((row_pos + i * offset) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + col_pos + (j * offset) + index_x_pixela_gorny_lewy_obszaru_przeszukania];
                //przpisujemy obszar preszukania (40 pixeli) dla ³atki do pamiêci dzielonej bloku, ze wzglêdu na zmieszczenie siê w dostêpnej w wywo³aniu funkcji iosci w¹tków musia³em zrealizowaæ przypisanie w czterech krokach.
            }
        }
    }
    __syncthreads();

    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        for (int i = 0; i < tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//³atka po ³atce przepsisujemy ³atki z obszaru przeszukania do tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32_
        {
            {
                indeks_pomocniczy1 = col_pos + row_pos * ROZMIAR_LATKI + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
                indeks_pomocniczy2 = (tablica_koordynat_latek_SOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i]) + col_pos + (tablica_koordynat_latek_SOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + row_pos) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA;
                tablice_transformaty.kanal_Cr[index_elmentu_zero_tablicy_transformat + indeks_pomocniczy1] = obszar_preszukana_shared[indeks_pomocniczy2];
            }
        }
    }
    //////////////////////////////Kana³ B
    if ((row_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5) && (col_pos < RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 5))
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                obszar_preszukana_shared[(row_pos + i * offset) * (RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA)+col_pos + (j * offset)] = obrazek_przepisywany.kanal_Cb[((row_pos + i * offset) + index_y_pixela_gorny_lewy_obszaru_przeszukania) * szerokosc + col_pos + (j * offset) + index_x_pixela_gorny_lewy_obszaru_przeszukania];
                //przpisujemy obszar preszukania (40 pixeli) dla ³atki do pamiêci dzielonej bloku, ze wzglêdu na zmieszczenie siê w dostêpnej w wywo³aniu funkcji iosci w¹tków musia³em zrealizowaæ przypisanie w czterech krokach.
            }
        }
    }
    __syncthreads();

    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        for (int i = 0; i < tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//³atka po ³atce przepsisujemy ³atki z obszaru przeszukania do tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32_
        {
            {
                int indeks_pomocniczy1 = col_pos + (row_pos)*ROZMIAR_LATKI + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
                int indeks_pomocniczy2 = (tablica_koordynat_latek_SOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i]) + col_pos + (tablica_koordynat_latek_SOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + row_pos) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA;
                tablice_transformaty.kanal_Cb[index_elmentu_zero_tablicy_transformat + indeks_pomocniczy1] = obszar_preszukana_shared[indeks_pomocniczy2];
            }
        }
    }
}




__global__ void Przepisywacz_z_tablic_transformaty_1krok_(Tablice_ilosci tablica_ilosci_zerowan, Obrazek_YCrCb obrazek_po_kolejnym_kroku, Obrazek_YCrCb obrazek_po_kolejnym_kroku_dzielnik, Tablice_koordynatLatek dev_koordynatySOA, int* device_tablica_ilosci_pasujacych_latek, Obrazek_YCrCb device_tablice_transformaty_16,
    int ilosc_blokow_w_boku_x, int szerokosc, int i, int j,
    int mnoznik_tablicy_transormat)
{
    int row_pos = threadIdx.y;
    int col_pos = threadIdx.x;
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;//przetestowaæ czy blo z czy y czy jeden i drugi!!!!!!!
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index_2d_latki = col_pos + (row_pos * ROZMIAR_LATKI);
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int index_elmentu_zero_tablicy_transformat = (blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat);
    int indeks_pomocniczy1_odkladanie_latek;
    int indeks_pomocniczy2_odkladanie_latek;

    float ilosc_niewyzerowanych_R = 1 / (float)tablica_ilosci_zerowan.kanal_Y[blockIdx.z];
    float ilosc_niewyzerowanych_G = 1 / (float)tablica_ilosci_zerowan.kanal_Cr[blockIdx.z];
    float ilosc_niewyzerowanych_B = 1 / (float)tablica_ilosci_zerowan.kanal_Cb[blockIdx.z];

    //__syncthreads();

    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)
        {

            indeks_pomocniczy1_odkladanie_latek = index_2d_latki + index_elmentu_zero_tablicy_transformat + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
            indeks_pomocniczy2_odkladanie_latek = ((dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i] + index_x_pixela_gorny_lewy_obszaru_przeszukania) + col_pos) + ((dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + index_y_pixela_gorny_lewy_obszaru_przeszukania + row_pos) * szerokosc);

            obrazek_po_kolejnym_kroku.kanal_Y[indeks_pomocniczy2_odkladanie_latek]
                += Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ((device_tablice_transformaty_16.kanal_Y[indeks_pomocniczy1_odkladanie_latek]) * ilosc_niewyzerowanych_R);
            obrazek_po_kolejnym_kroku_dzielnik.kanal_Y[indeks_pomocniczy2_odkladanie_latek]
                += Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ilosc_niewyzerowanych_R;

            __syncthreads();
        }
        //__syncthreads();
        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//³atka po ³atce przepsisujemy ³atki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
        {

            indeks_pomocniczy1_odkladanie_latek = index_2d_latki + index_elmentu_zero_tablicy_transformat + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
            indeks_pomocniczy2_odkladanie_latek = ((dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i] + index_x_pixela_gorny_lewy_obszaru_przeszukania) + col_pos) + ((dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + index_y_pixela_gorny_lewy_obszaru_przeszukania + row_pos) * szerokosc);

            obrazek_po_kolejnym_kroku.kanal_Cr[indeks_pomocniczy2_odkladanie_latek] += Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ((device_tablice_transformaty_16.kanal_Cr[indeks_pomocniczy1_odkladanie_latek]) * ilosc_niewyzerowanych_G);
            obrazek_po_kolejnym_kroku_dzielnik.kanal_Cr[indeks_pomocniczy2_odkladanie_latek] += Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ilosc_niewyzerowanych_G;
            __syncthreads();
        }


        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//³atka po ³atce przepsisujemy ³atki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
        {

            indeks_pomocniczy1_odkladanie_latek = index_2d_latki + index_elmentu_zero_tablicy_transformat + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
            indeks_pomocniczy2_odkladanie_latek = ((dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i] + index_x_pixela_gorny_lewy_obszaru_przeszukania) + col_pos) + ((dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + index_y_pixela_gorny_lewy_obszaru_przeszukania + row_pos) * szerokosc);

            obrazek_po_kolejnym_kroku.kanal_Cb[indeks_pomocniczy2_odkladanie_latek] += Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ((device_tablice_transformaty_16.kanal_Cb[indeks_pomocniczy1_odkladanie_latek]) * ilosc_niewyzerowanych_B);
            obrazek_po_kolejnym_kroku_dzielnik.kanal_Cb[indeks_pomocniczy2_odkladanie_latek] += Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ilosc_niewyzerowanych_B;

            __syncthreads();
        }
    }
}


__global__ void Przepisywacz_z_tablic_transformaty_2krok_
(Obrazek_YCrCb device_tablica_wag_fitru_wienera, Obrazek_YCrCb obrazek_po_kolejnym_kroku, Obrazek_YCrCb obrazek_po_kolejnym_kroku_dzielnik, Tablice_koordynatLatek dev_koordynatySOA,
    int* device_tablica_ilosci_pasujacych_latek,
    Obrazek_YCrCb device_tablice_transformaty_32_2krok, int ilosc_blokow_w_boku_x,
    int szerokosc, int i, int j, int mnoznik_tablicy_transormat)
{
    int row_pos = threadIdx.y;
    int col_pos = threadIdx.x;
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;//przetestowaæ czy blo z czy y czy jeden i drugi!!!!!!!
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index_2d_latki = col_pos + (row_pos * ROZMIAR_LATKI);
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int index_elmentu_zero_tablicy_transformat = (blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat);


    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {

        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)
            //³atka po ³atce przepsisujemy ³atki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
        {

            int indeks_pomocniczy1_odkladanie_latek = index_2d_latki + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
            int indeks_pomocniczy2_odkladanie_latek = ((dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i] + index_x_pixela_gorny_lewy_obszaru_przeszukania) + col_pos) + ((dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + index_y_pixela_gorny_lewy_obszaru_przeszukania + row_pos) * szerokosc);

            obrazek_po_kolejnym_kroku.kanal_Y[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ((device_tablice_transformaty_32_2krok.kanal_Y[index_elmentu_zero_tablicy_transformat + indeks_pomocniczy1_odkladanie_latek])) * device_tablica_wag_fitru_wienera.kanal_Y[blockIdx.z]);
            obrazek_po_kolejnym_kroku_dzielnik.kanal_Y[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * device_tablica_wag_fitru_wienera.kanal_Y[blockIdx.z]);

            __syncthreads();
        }
        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//³atka po ³atce przepsisujemy ³atki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
        {
            int indeks_pomocniczy1_odkladanie_latek = index_2d_latki + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
            int indeks_pomocniczy2_odkladanie_latek = ((dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i] + index_x_pixela_gorny_lewy_obszaru_przeszukania) + col_pos) + ((dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + index_y_pixela_gorny_lewy_obszaru_przeszukania + row_pos) * szerokosc);

            obrazek_po_kolejnym_kroku.kanal_Cr[indeks_pomocniczy2_odkladanie_latek] += Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * device_tablice_transformaty_32_2krok.kanal_Cr[index_elmentu_zero_tablicy_transformat + indeks_pomocniczy1_odkladanie_latek] * device_tablica_wag_fitru_wienera.kanal_Cr[blockIdx.z];
            obrazek_po_kolejnym_kroku_dzielnik.kanal_Cr[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * device_tablica_wag_fitru_wienera.kanal_Cr[blockIdx.z]);

            __syncthreads();
        }
        //__syncthreads();
    }
    for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//³atka po ³atce przepsisujemy ³atki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
    {
        int indeks_pomocniczy1_odkladanie_latek = index_2d_latki + (i * ROZMIAR_LATKI * ROZMIAR_LATKI);
        int indeks_pomocniczy2_odkladanie_latek = ((dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + i] + index_x_pixela_gorny_lewy_obszaru_przeszukania) + col_pos) + ((dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + i] + index_y_pixela_gorny_lewy_obszaru_przeszukania + row_pos) * szerokosc);

        obrazek_po_kolejnym_kroku.kanal_Cb[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * ((device_tablice_transformaty_32_2krok.kanal_Cb[index_elmentu_zero_tablicy_transformat + indeks_pomocniczy1_odkladanie_latek])) * device_tablica_wag_fitru_wienera.kanal_Cb[blockIdx.z]);
        obrazek_po_kolejnym_kroku_dzielnik.kanal_Cb[indeks_pomocniczy2_odkladanie_latek] += (Macierz_wspolczynnikow_Kaizerra[index_2d_latki] * device_tablica_wag_fitru_wienera.kanal_Cb[blockIdx.z]);
        __syncthreads();
    }
}

__global__ void Zerowanie_(Obrazek_YCrCb device_tablice_transformaty_16,
    Tablice_ilosci device_tablica_ilosci_zerowan,
    int* device_tablica_ilosci_pasujacych_latek,
    Szum_YCrCb* Dev_Szum_YCrCb, int mnoznik_tablicy_transormat)
{
    int indeks = blockIdx.x * blockDim.x + threadIdx.x;
    int indeks_poczatkowy_grupy = blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat;
    int ilosc_pasujacych_latek = device_tablica_ilosci_pasujacych_latek[blockIdx.z];
    if (indeks < 1)
    {
        device_tablica_ilosci_zerowan.kanal_Y[blockIdx.z] = 0;
        device_tablica_ilosci_zerowan.kanal_Cr[blockIdx.z] = 0;
        device_tablica_ilosci_zerowan.kanal_Cb[blockIdx.z] = 0;
    }
    __syncthreads();

    if (indeks < ilosc_pasujacych_latek * ROZMIAR_LATKI * ROZMIAR_LATKI)
    {
        if (abs(device_tablice_transformaty_16.kanal_Y[indeks + indeks_poczatkowy_grupy]) < (LAMBDA3D_HARD * Dev_Szum_YCrCb
            ->szum_kanal_Y))
        {
            device_tablice_transformaty_16.kanal_Y[indeks + indeks_poczatkowy_grupy] = 0.0f;
        }
        else
        {
            atomicAdd(&device_tablica_ilosci_zerowan.kanal_Y[blockIdx.z], 1);
        }
    }
    __syncthreads();

    if (indeks < ilosc_pasujacych_latek * ROZMIAR_LATKI * ROZMIAR_LATKI)
    {
        if (abs(device_tablice_transformaty_16.kanal_Cr[indeks + indeks_poczatkowy_grupy]) < (LAMBDA3D_HARD * Dev_Szum_YCrCb
            ->szum_kanal_Cr))
        {
            device_tablice_transformaty_16.kanal_Cr[indeks + indeks_poczatkowy_grupy] = 0.0f;
        }
        else
        {
            atomicAdd(&device_tablica_ilosci_zerowan.kanal_Cr[blockIdx.z], 1);
        }
    }
    __syncthreads();
    if (indeks < ilosc_pasujacych_latek * ROZMIAR_LATKI * ROZMIAR_LATKI)
    {
        if (abs(device_tablice_transformaty_16.kanal_Cb[indeks + indeks_poczatkowy_grupy]) < (LAMBDA3D_HARD * Dev_Szum_YCrCb->szum_kanal_Cb))

        {
            device_tablice_transformaty_16.kanal_Cb[indeks + indeks_poczatkowy_grupy] = 0.0f;
        }
        else
        {
            atomicAdd(&device_tablica_ilosci_zerowan.kanal_Cb[blockIdx.z], 1);
        }
    }

    __syncthreads();
    if (device_tablica_ilosci_zerowan.kanal_Y[blockIdx.z] < 1)
    {
        device_tablica_ilosci_zerowan.kanal_Y[blockIdx.z] = 1;
    }
    if (device_tablica_ilosci_zerowan.kanal_Cr[blockIdx.z] < 1)
    {
        device_tablica_ilosci_zerowan.kanal_Cr[blockIdx.z] = 1;
    }
    if (device_tablica_ilosci_zerowan.kanal_Cb[blockIdx.z] < 1)
    {
        device_tablica_ilosci_zerowan.kanal_Cb[blockIdx.z] = 1;
    }
}

__global__ void Filtr_Wienera(Obrazek_YCrCb device_tablica_wag_fitru_wienera, Obrazek_YCrCb device_tablice_transformaty_32_1krok, Obrazek_YCrCb device_tablice_transformaty_32_2krok, int* device_tablica_ilosci_pasujacych_latek, Szum_YCrCb* Dev_Szum_YCrCb, int mnoznik_tablicy_transormat)
{
    int przesuniecie = blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat;
    int indeks_w_grupie = blockIdx.x * blockDim.x + threadIdx.x;
    int indeks_w_tablicy = blockIdx.x * blockDim.x + threadIdx.x + przesuniecie;
    float x_Y;
    float x_Cr;
    float x_Cb;
    float wspolczynnik_filtracji_wienera_Y;
    float wspolczynnik_filtracji_wienera_Cr;
    float wspolczynnik_filtracji_wienera_Cb;
    float sigma_kwadrat_Y = Dev_Szum_YCrCb->szum_kanal_Y * Dev_Szum_YCrCb->szum_kanal_Y;
    float sigma_kwadrat_Cr = Dev_Szum_YCrCb->szum_kanal_Cr * Dev_Szum_YCrCb->szum_kanal_Cr;
    float sigma_kwadrat_Cb = Dev_Szum_YCrCb->szum_kanal_Cb * Dev_Szum_YCrCb->szum_kanal_Cb;

    if (indeks_w_grupie < 1)
    {
        device_tablica_wag_fitru_wienera.kanal_Y[blockIdx.z] = 0.0;
        device_tablica_wag_fitru_wienera.kanal_Cr[blockIdx.z] = 0.0;
        device_tablica_wag_fitru_wienera.kanal_Cb[blockIdx.z] = 0.0;
    }

    __syncthreads();

    if (device_tablica_ilosci_pasujacych_latek[blockIdx.z] > 0)
    {
        if (blockIdx.x < device_tablica_ilosci_pasujacych_latek[blockIdx.z])
        {
            x_Y = (device_tablice_transformaty_32_2krok.kanal_Y[indeks_w_tablicy] * device_tablice_transformaty_32_2krok.kanal_Y[indeks_w_tablicy]);
            wspolczynnik_filtracji_wienera_Y = x_Y / (x_Y + sigma_kwadrat_Y);
            device_tablice_transformaty_32_2krok.kanal_Y[indeks_w_tablicy] = ((device_tablice_transformaty_32_1krok.kanal_Y[indeks_w_tablicy] * wspolczynnik_filtracji_wienera_Y));
            //device_tablice_transformaty_32_1krok.kanal_Y[indeks_w_tablicy] = 1/wspolczynnik_filtracji_wienera_Y* wspolczynnik_filtracji_wienera_Y;
            //device_tablice_transformaty_32_1krok.kanal_Y[indeks_w_tablicy] = 1.0 / device_tablica_ilosci_pasujacych_latek[blockIdx.z];

            x_Cr = (device_tablice_transformaty_32_2krok.kanal_Cr[indeks_w_tablicy] * device_tablice_transformaty_32_2krok.kanal_Cr[indeks_w_tablicy]);
            wspolczynnik_filtracji_wienera_Cr = x_Cr / (x_Cr + sigma_kwadrat_Cr);
            device_tablice_transformaty_32_2krok.kanal_Cr[indeks_w_tablicy] = ((device_tablice_transformaty_32_1krok.kanal_Cr[indeks_w_tablicy] * wspolczynnik_filtracji_wienera_Cr));
            //__syncthreads();
            //device_tablice_transformaty_32_1krok.kanal_Cr[indeks_w_tablicy] = 1/wspolczynnik_filtracji_wienera_Cr* wspolczynnik_filtracji_wienera_Cr;
            //device_tablice_transformaty_32_1krok.kanal_Cr[indeks_w_tablicy] = 1.0 / device_tablica_ilosci_pasujacych_latek[blockIdx.z];

            x_Cb = (device_tablice_transformaty_32_2krok.kanal_Cb[indeks_w_tablicy] * device_tablice_transformaty_32_2krok.kanal_Cb[indeks_w_tablicy]);
            wspolczynnik_filtracji_wienera_Cb = x_Cb / (x_Cb + sigma_kwadrat_Cb);
            device_tablice_transformaty_32_2krok.kanal_Cb[indeks_w_tablicy] = ((device_tablice_transformaty_32_1krok.kanal_Cb[indeks_w_tablicy] * wspolczynnik_filtracji_wienera_Cb));
            //__syncthreads();
            //device_tablice_transformaty_32_1krok.kanal_Cb[indeks_w_tablicy] = 1/wspolczynnik_filtracji_wienera_Cb* wspolczynnik_filtracji_wienera_Cb;
            //device_tablice_transformaty_32_1krok.kanal_Cb[indeks_w_tablicy] = 1.0 / device_tablica_ilosci_pasujacych_latek[blockIdx.z];

        }
    }

    __syncthreads();
    // opis filtu wienera wed³ug Dubowa zaka³a pos³y¿enie siê odwrotnoœci¹ sumy wartoœci wspó³czynników fitracji jako wag¹.Z testów wynika, ¿e efektywniejsz¹,a przedewszystkim szybsz¹ metod¹ jest wykorzystania jako wagi iloœci pasuj¹cych ³atek

    if (indeks_w_grupie == 0)
    {
        device_tablica_wag_fitru_wienera.kanal_Y[blockIdx.z] = 1.0 / device_tablica_ilosci_pasujacych_latek[blockIdx.z];

        device_tablica_wag_fitru_wienera.kanal_Cr[blockIdx.z] = 1.0 / device_tablica_ilosci_pasujacych_latek[blockIdx.z];

        device_tablica_wag_fitru_wienera.kanal_Cb[blockIdx.z] = 1.0 / device_tablica_ilosci_pasujacych_latek[blockIdx.z];
    }

}

__global__ void Nadpisywanie_marginesow1(Obrazek_YCrCb device_obrazek_po1kroku, int szerokosc, int wysokosc, int margines_lewy, int margines_prawy) //doanaie nowych marginesów
{

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_pos < wysokosc && col_pos < margines_lewy)

    {
        device_obrazek_po1kroku.kanal_Y[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Y[(margines_lewy + (margines_lewy - col_pos)) + (row_pos * szerokosc)];
        device_obrazek_po1kroku.kanal_Cr[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Cr[(margines_lewy + (margines_lewy - col_pos)) + (row_pos * szerokosc)];
        device_obrazek_po1kroku.kanal_Cb[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Cb[(margines_lewy + (margines_lewy - col_pos)) + (row_pos * szerokosc)];
    }

    if (row_pos < wysokosc && col_pos < szerokosc)//
    {
        if (col_pos > (szerokosc - margines_prawy))

        {
            device_obrazek_po1kroku.kanal_Y[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Y[(szerokosc - margines_prawy - (col_pos - (szerokosc - margines_prawy))) + (row_pos * szerokosc)]; //-szerokosc + margines
            device_obrazek_po1kroku.kanal_Cr[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Cr[(szerokosc - margines_prawy - (col_pos - (szerokosc - margines_prawy))) + (row_pos * szerokosc)];
            device_obrazek_po1kroku.kanal_Cb[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Cb[(szerokosc - margines_prawy - (col_pos - (szerokosc - margines_prawy))) + (row_pos * szerokosc)];
        }
    }

}

__global__ void Nadpisywanie_marginesow2(Obrazek_YCrCb device_obrazek_po1kroku, int szerokosc, int wysokosc, int margines_lewy, int margines_prawy) //dzielenie wyiku sumowania zerowanych ³atek zprzez ilosc zerowañ oraz doanaie nowych marginesów
{

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_pos < margines_lewy && col_pos < szerokosc)

    {
        //float x = device_obrazek_po1kroku[col_pos + (margines_lewy + (margines_lewy - row_pos)) * szerokosc];
        device_obrazek_po1kroku.kanal_Y[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Y[col_pos + (margines_lewy + (margines_lewy - row_pos)) * szerokosc];
        device_obrazek_po1kroku.kanal_Cr[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Cr[col_pos + (margines_lewy + (margines_lewy - row_pos)) * szerokosc];
        device_obrazek_po1kroku.kanal_Cb[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Cb[col_pos + (margines_lewy + (margines_lewy - row_pos)) * szerokosc];
    }

    if (row_pos < wysokosc && col_pos < szerokosc)
    {
        if (row_pos > wysokosc - margines_prawy)

        {
            device_obrazek_po1kroku.kanal_Y[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Y[(col_pos + (wysokosc - margines_prawy - (row_pos - (wysokosc - margines_prawy))) * szerokosc)];
            device_obrazek_po1kroku.kanal_Cr[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Cr[(col_pos + (wysokosc - margines_prawy - (row_pos - (wysokosc - margines_prawy))) * szerokosc)];
            device_obrazek_po1kroku.kanal_Cb[col_pos + row_pos * szerokosc] = device_obrazek_po1kroku.kanal_Cb[(col_pos + (wysokosc - margines_prawy - (row_pos - (wysokosc - margines_prawy))) * szerokosc)];

        }
    }
}

__global__ void DzielenieMacierzy(Obrazek_YCrCb device_obrazek_po_n_kroku, Obrazek_YCrCb device_obrazek_po_n_kroku_dzielnik, int szerokosc, int wysokosc)
{

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int index2d_pixela = col_pos + row_pos * szerokosc;

    if (row_pos < wysokosc && col_pos < szerokosc)
    {
        device_obrazek_po_n_kroku.kanal_Y[index2d_pixela] = device_obrazek_po_n_kroku.kanal_Y[index2d_pixela] / device_obrazek_po_n_kroku_dzielnik.kanal_Y[index2d_pixela];
        device_obrazek_po_n_kroku.kanal_Cr[index2d_pixela] = device_obrazek_po_n_kroku.kanal_Cr[index2d_pixela] / device_obrazek_po_n_kroku_dzielnik.kanal_Cr[index2d_pixela];
        device_obrazek_po_n_kroku.kanal_Cb[index2d_pixela] = device_obrazek_po_n_kroku.kanal_Cb[index2d_pixela] / device_obrazek_po_n_kroku_dzielnik.kanal_Cb[index2d_pixela];
    }
}

__global__ void Zerowanie_Macierzy(Obrazek_YCrCb Obrazek, int szerokosc, int wysokosc) //dzielenie wyiku sumowania zerowanych ³atek zprzez ilosc zerowañ
{

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;


    int index2d_pixela = row_pos + col_pos * szerokosc;


    if (row_pos < szerokosc && col_pos < wysokosc)
    {
        Obrazek.kanal_Y[index2d_pixela] = 0;
        Obrazek.kanal_Cr[index2d_pixela] = 0;
        Obrazek.kanal_Cb[index2d_pixela] = 0;
    }
}



void initializeCUDA(int argc, char** argv, int& devID)
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

void dodanie_szumu(cv::Mat obrazek_zaszumiony, float sigm)
{

    double sigma = sigm; // Wartoœæ sigma dla szumu gaussowskiego

    // Generator liczb losowych dla szumu gaussowskiego
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, sigma);

    // Dodaje szum gaussowski do ka¿dego piksela
    for (int y = 0; y < obrazek_zaszumiony.rows; y++) {
        for (int x = 0; x < obrazek_zaszumiony.cols; x++) {
            cv::Vec3b& pixele = obrazek_zaszumiony.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; c++) {
                double szum = distribution(generator);
                int new_value = cv::saturate_cast<uchar>(pixele[c] + szum);
                pixele[c] = new_value;
            }
        }
    }
}
void funkcja_glowna(cv::Mat Obrazek, cv::Mat& Obrazek_odszumiony, float sigma, int szybkosc)
{
    int szerokosc_obrazka_oryginalnego;
    int wysokosc_obrazka_oryginalnego;
    int wielkosc_marginesu_lewego;
    int wielkosc_marginesu_prawego;
    int szerokosc_obrazka_z_marginesami;
    int wysokosc_obrazka_z_marginesami;
    int p_hard = P_HARD; //przesyniêcie pomidzy ³atkami w kroku 1, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
    int p_wien = P_WIEN;//przesuniêcie pomiêdzy ³atkami w kroku 2, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
    int zakladka_obszaru_przeszukania = 2;
    float tau_hard = 5000;
    float tau_wien = 800;

    Szum_YCrCb Host_Szum_YCrCb;
    Szum_YCrCb Dev_Szum_YCrCb;
    Obrazek_YCrCb Host_Macierz_wejsciowa;
    Obrazek_YCrCb Host_Obrazek_po_1kroku;
    Obrazek_YCrCb Host_Obrazek_po_2kroku;
    
    Obrazek_YCrCb Dev_Macierz_wejsciowa;
    Obrazek_YCrCb Dev_Obrazek_po_1kroku;
    Obrazek_YCrCb Dev_Obrazek_po_2kroku;
    Obrazek_YCrCb Device_tablice_transformaty_16;
    Obrazek_YCrCb Device_tablice_transformaty_32_1krok;
    Obrazek_YCrCb Device_tablice_transformaty_32_2krok;
    Obrazek_YCrCb Dev_Obrazek_Dzielnik;
    Obrazek_YCrCb Device_tablica_wartosci_fitru_wienera;
    Tablice_koordynatLatek dev_koordynatySOA;
    Tablice_ilosci Device_tablica_ilosci_zerowan;
    
    cv::Vec3f kanaly_do_przepisania;

    int ilosc_latek_w_obszarze_przeszukania = (ROZMIAR_OBSZARU_PRZESZUKANIA) * (ROZMIAR_OBSZARU_PRZESZUKANIA);
    int ilosc_blokow_w_boku_x;
    int ilosc_blokow_w_boku_y;
    int ilosc_blokow;

    int wielkosc_tablicy_z_marginesami;
    int wielkosc_tablicy_transformaty_32;
    int wielkosc_tablicy_transformaty_16;
    int wielkosc_tablicy_ilosci_pasujacych_latek;
    int wielkosc_tablicy_zerowan;
    int wielkosc_tablicy_koordynat;
    int rozmiar_w_pamieci_tablic_koordynat_floaty;
    int opcja_obrazka;

    if (szybkosc == 0)
    {
        p_hard = P_HARD; //przesyniêcie pomidzy ³atkami w kroku 1, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
        p_wien = P_WIEN;//przesuniêcie pomiêdzy ³atkami w kroku 2, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
        zakladka_obszaru_przeszukania = 2;
    }
    else if (szybkosc == 1)
    {
        p_hard = 4;
        p_wien = 4;
        zakladka_obszaru_przeszukania = 5;
    }

    /////////////////////////////////////////////////////////////////////////////

    Host_Szum_YCrCb = { float((sqrtf(0.299 * 0.299 + 0.587 * 0.587 + 0.114 * 0.114) * sigma) * 0.97),float((sqrtf(0.169 * 0.169 + 0.331 * 0.331 + 0.500 * 0.500) * sigma) * 1.15), float((sqrtf(0.500 * 0.500 + 0.419 * 0.419 + 0.081 * 0.081) * sigma) * 0.96) };

    szerokosc_obrazka_oryginalnego = Obrazek.cols;
    wysokosc_obrazka_oryginalnego = Obrazek.rows;
    wielkosc_marginesu_lewego = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2;
    wielkosc_marginesu_prawego = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA;
    szerokosc_obrazka_z_marginesami = szerokosc_obrazka_oryginalnego + wielkosc_marginesu_lewego + wielkosc_marginesu_prawego;
    wysokosc_obrazka_z_marginesami = wysokosc_obrazka_oryginalnego + wielkosc_marginesu_lewego + wielkosc_marginesu_prawego;
    wielkosc_tablicy_z_marginesami = szerokosc_obrazka_z_marginesami * wysokosc_obrazka_z_marginesami;
    ///////////////dodajemy "marginesy" do obrazka (po po³owie obszaru przeszukania po 16)////////////////////////       
    cv::copyMakeBorder(Obrazek, Obrazek, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego, cv::BORDER_REFLECT_101);
    cv::cvtColor(Obrazek, Obrazek, cv::COLOR_RGB2YCrCb);
    Obrazek.convertTo(Obrazek, CV_32FC3);

    //////////////////////////////////////////////////////////////////tworzymy zmienne do przekazania do Kernelu starowego///////////////////////////

    ilosc_blokow_w_boku_x = (int)std::ceil(((double)szerokosc_obrazka_oryginalnego / (ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI)));
    ilosc_blokow_w_boku_y = (int)ceil(((double)wysokosc_obrazka_oryginalnego) / (ROZMIAR_OBSZARU_PRZESZUKANIA + ROZMIAR_LATKI));
    ilosc_blokow = ilosc_blokow_w_boku_x * ilosc_blokow_w_boku_y;


    wielkosc_tablicy_transformaty_32 = ilosc_blokow_w_boku_x * ilosc_blokow_w_boku_y * ROZMIAR_LATKI * ROZMIAR_LATKI * N_WIEN;
    wielkosc_tablicy_transformaty_16 = ilosc_blokow_w_boku_x * ilosc_blokow_w_boku_y * ROZMIAR_LATKI * ROZMIAR_LATKI * N_HARD;
    wielkosc_tablicy_ilosci_pasujacych_latek = ilosc_blokow_w_boku_x * ilosc_blokow_w_boku_y;
    wielkosc_tablicy_zerowan = ilosc_blokow_w_boku_x * ilosc_blokow_w_boku_y;
    wielkosc_tablicy_koordynat = ilosc_blokow_w_boku_x * ilosc_blokow_w_boku_y * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA;
    ////////////////////////////////////////////przygotowanie i lokowanie w pamiêci obrazka i jego kolejnych wersji////////////////////////////////

    Obrazek_YCrCb* wskaznik_host_Macierz_wejsciowa = &Host_Macierz_wejsciowa;
    wskaznik_host_Macierz_wejsciowa->kanal_Y = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Macierz_wejsciowa->kanal_Cr = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Macierz_wejsciowa->kanal_Cb = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));

    Obrazek_YCrCb* wskaznik_host_Obrazek_po_1kroku = &Host_Obrazek_po_1kroku;
    wskaznik_host_Obrazek_po_1kroku->kanal_Y = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Obrazek_po_1kroku->kanal_Cr = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Obrazek_po_1kroku->kanal_Cb = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));

    Obrazek_YCrCb* wskaznik_host_Obrazek_po_2kroku = &Host_Obrazek_po_2kroku;
    wskaznik_host_Obrazek_po_2kroku->kanal_Y = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Obrazek_po_2kroku->kanal_Cr = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));
    wskaznik_host_Obrazek_po_2kroku->kanal_Cb = (float*)malloc(wielkosc_tablicy_z_marginesami * sizeof(float));

    Szum_YCrCb* wskaznik_host_Szum_YCrCb;
    wskaznik_host_Szum_YCrCb = &Host_Szum_YCrCb;

    Obrazek = Obrazek.reshape(0, wielkosc_tablicy_z_marginesami);
    for (int i = 0; i < wielkosc_tablicy_z_marginesami; i++)
    {
        kanaly_do_przepisania = Obrazek.at<cv::Vec3f>(i, 0);
        Host_Macierz_wejsciowa.kanal_Y[i] = kanaly_do_przepisania[0];
        Host_Macierz_wejsciowa.kanal_Cr[i] = kanaly_do_przepisania[1];
        Host_Macierz_wejsciowa.kanal_Cb[i] = kanaly_do_przepisania[2];
    }

    Obrazek_YCrCb* wskaznik_dev_Macierz_wejsciowa = &Dev_Macierz_wejsciowa;
    cudaMalloc((&wskaznik_dev_Macierz_wejsciowa->kanal_Y), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((&wskaznik_dev_Macierz_wejsciowa->kanal_Cr), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((&wskaznik_dev_Macierz_wejsciowa->kanal_Cb), wielkosc_tablicy_z_marginesami * sizeof(float));

    Obrazek_YCrCb* wskaznik_dev_Obrazek_po_1kroku = &Dev_Obrazek_po_1kroku;
    cudaMalloc((&wskaznik_dev_Obrazek_po_1kroku->kanal_Y), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((&wskaznik_dev_Obrazek_po_1kroku->kanal_Cr), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc((&wskaznik_dev_Obrazek_po_1kroku->kanal_Cb), wielkosc_tablicy_z_marginesami * sizeof(float));

    Obrazek_YCrCb* wskaznik_dev_Obrazek_po_2kroku = &Dev_Obrazek_po_2kroku;
    cudaMalloc((&wskaznik_dev_Obrazek_po_2kroku->kanal_Y), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc(&(wskaznik_dev_Obrazek_po_2kroku->kanal_Cr), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc(&(wskaznik_dev_Obrazek_po_2kroku->kanal_Cb), wielkosc_tablicy_z_marginesami * sizeof(float));

    Obrazek_YCrCb* wskaznik_device_tablice_transformaty_16 = &Device_tablice_transformaty_16;
    cudaMalloc((&wskaznik_device_tablice_transformaty_16->kanal_Y), wielkosc_tablicy_transformaty_16 * sizeof(float));
    cudaMalloc(&(wskaznik_device_tablice_transformaty_16->kanal_Cr), wielkosc_tablicy_transformaty_16 * sizeof(float));
    cudaMalloc(&(wskaznik_device_tablice_transformaty_16->kanal_Cb), wielkosc_tablicy_transformaty_16 * sizeof(float));

    Obrazek_YCrCb* wskaznik_device_tablice_transformaty_32_1krok = &Device_tablice_transformaty_32_1krok;
    cudaMalloc((&wskaznik_device_tablice_transformaty_32_1krok->kanal_Y), wielkosc_tablicy_transformaty_32 * sizeof(float));
    cudaMalloc(&(wskaznik_device_tablice_transformaty_32_1krok->kanal_Cr), wielkosc_tablicy_transformaty_32 * sizeof(float));
    cudaMalloc(&(wskaznik_device_tablice_transformaty_32_1krok->kanal_Cb), wielkosc_tablicy_transformaty_32 * sizeof(float));

    Obrazek_YCrCb* wskaznik_device_tablice_transformaty_32_2krok = &Device_tablice_transformaty_32_2krok;
    cudaMalloc((&wskaznik_device_tablice_transformaty_32_2krok->kanal_Y), wielkosc_tablicy_transformaty_32 * sizeof(float));
    cudaMalloc(&(wskaznik_device_tablice_transformaty_32_2krok->kanal_Cr), wielkosc_tablicy_transformaty_32 * sizeof(float));
    cudaMalloc(&(wskaznik_device_tablice_transformaty_32_2krok->kanal_Cb), wielkosc_tablicy_transformaty_32 * sizeof(float));

    Obrazek_YCrCb* wskaznik_dev_Obrazek_Dzielnik = &Dev_Obrazek_Dzielnik;
    cudaMalloc((&wskaznik_dev_Obrazek_Dzielnik->kanal_Y), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc(&(wskaznik_dev_Obrazek_Dzielnik->kanal_Cr), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc(&(wskaznik_dev_Obrazek_Dzielnik->kanal_Cb), wielkosc_tablicy_z_marginesami * sizeof(float));

    Obrazek_YCrCb* wskaznik_device_tablica_wag_fitru_wienera = &Device_tablica_wartosci_fitru_wienera;
    cudaMalloc((&wskaznik_device_tablica_wag_fitru_wienera->kanal_Y), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc(&(wskaznik_device_tablica_wag_fitru_wienera->kanal_Cr), wielkosc_tablicy_z_marginesami * sizeof(float));
    cudaMalloc(&(wskaznik_device_tablica_wag_fitru_wienera->kanal_Cb), wielkosc_tablicy_z_marginesami * sizeof(float));


    Tablice_koordynatLatek* wskaznik_dev_koordynatySOA = &dev_koordynatySOA;
    cudaMalloc((&wskaznik_dev_koordynatySOA->MSE), sizeof(float) * wielkosc_tablicy_koordynat);
    cudaMalloc(&(wskaznik_dev_koordynatySOA->koordynata_x), sizeof(int) * wielkosc_tablicy_koordynat);
    cudaMalloc(&(wskaznik_dev_koordynatySOA->koordynata_y), sizeof(int) * wielkosc_tablicy_koordynat);

    Tablice_ilosci* wskaznik_device_tablica_ilosci_zerowan = &Device_tablica_ilosci_zerowan;
    cudaMalloc((&wskaznik_device_tablica_ilosci_zerowan->kanal_Y), wielkosc_tablicy_zerowan * sizeof(int));
    cudaMalloc(&(wskaznik_device_tablica_ilosci_zerowan->kanal_Cr), wielkosc_tablicy_zerowan * sizeof(int));
    cudaMalloc(&(wskaznik_device_tablica_ilosci_zerowan->kanal_Cb), wielkosc_tablicy_zerowan * sizeof(int));

    Szum_YCrCb* wskaznik_dev_Szum_YCrCb = &Dev_Szum_YCrCb;
    cudaMalloc((&wskaznik_dev_Szum_YCrCb), sizeof(Szum_YCrCb));

    int* Device_tablica_ilosci_pasujacych_latek;
    cudaMalloc(&Device_tablica_ilosci_pasujacych_latek, ilosc_blokow_w_boku_x * ilosc_blokow_w_boku_y * sizeof(int));
    //////////////////////// kopiowane do pamiêci device

   /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    dim3 bloki_Kalkulator_MSE_(1, 1, ilosc_blokow);
    dim3 watki_Kalkulator_MSE_(ROZMIAR_OBSZARU_PRZESZUKANIA, ROZMIAR_OBSZARU_PRZESZUKANIA, 1);
    dim3 bloki_Kalkulator_MSE_szum_duzy(ROZMIAR_OBSZARU_PRZESZUKANIA, ROZMIAR_OBSZARU_PRZESZUKANIA, ilosc_blokow);
    dim3 watki_Kalkulator_MSE_szum_duzy(ROZMIAR_LATKI, ROZMIAR_LATKI, 1);
    dim3 bloki_najmniejsze_liczby(1, 1, ilosc_blokow);
    dim3 watki_najmniejsze_liczby(ilosc_latek_w_obszarze_przeszukania / 2, 1, 1);
    dim3 watki_Przepisywacz(ROZMIAR_LATKI, ROZMIAR_LATKI, 1);
    dim3 bloki_Przepisywacz(1, 1, ilosc_blokow);
    dim3 bloki_DCT_krok1(1, 1, ilosc_blokow * N_HARD);
    dim3 watki_DCT_krok1(ROZMIAR_LATKI, ROZMIAR_LATKI, 1);
    dim3 bloki_DCT_krok2(1, 1, ilosc_blokow * N_WIEN);
    dim3 watki_DCT_krok2(ROZMIAR_LATKI, ROZMIAR_LATKI, 1);
    dim3 bloki_Walsh(1, 1, ilosc_blokow);
    dim3 watki_Walsh_krok1(ROZMIAR_LATKI * ROZMIAR_LATKI, N_HARD / 2, 1);
    dim3 watki_Walsh_krok2(ROZMIAR_LATKI * ROZMIAR_LATKI, N_WIEN / 2, 1);

    dim3 bloki_Zerowanie(1, 1, ilosc_blokow);
    dim3 watki_Zerowanie(N_HARD * ROZMIAR_LATKI * ROZMIAR_LATKI, 1, 1);
    dim3 bloki_Wien(N_WIEN, 1, ilosc_blokow);
    dim3 watki_Wien(ROZMIAR_LATKI * ROZMIAR_LATKI, 1, 1);
    int dzielenie_macierzy_watki_x = 32;
    int dzielenie_macierzy_watki_y = 32;
    int dzielenie_macierzy_bloki_x = (szerokosc_obrazka_z_marginesami + dzielenie_macierzy_watki_x) / dzielenie_macierzy_watki_x;
    int dzielenie_macierzy_bloki_y = (wysokosc_obrazka_z_marginesami + dzielenie_macierzy_watki_y) / dzielenie_macierzy_watki_y;
    dim3 bloki_dzielnie_Macierzy(dzielenie_macierzy_bloki_x, dzielenie_macierzy_bloki_y, 1);
    dim3 watki_dzielnie_Macierzy(dzielenie_macierzy_watki_x, dzielenie_macierzy_watki_y, 1);

    int start_3 = clock();

    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_Y, wskaznik_host_Macierz_wejsciowa->kanal_Y, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_Cr, wskaznik_host_Macierz_wejsciowa->kanal_Cr, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(wskaznik_dev_Macierz_wejsciowa->kanal_Cb, wskaznik_host_Macierz_wejsciowa->kanal_Cb, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(wskaznik_dev_Szum_YCrCb, wskaznik_host_Szum_YCrCb, 1 * sizeof(Szum_YCrCb), cudaMemcpyHostToDevice);



    for (int i = 0; i < ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania; i += p_hard)
    {
        for (int j = 0; j < ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania; j += p_hard)
        {
            if (sigma > 40)
            {
                tau_hard = 10000;
                Kalkulator_MSE_szum_duzy << <bloki_Kalkulator_MSE_szum_duzy, watki_Kalkulator_MSE_szum_duzy >> > (Dev_Macierz_wejsciowa.kanal_Y, dev_koordynatySOA, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j, sigma);
                //Kalkulator_MSE_ << <bloki_Kalkulator_MSE_, watki_Kalkulator_MSE_ >> > (Dev_Macierz_wejsciowa.kanal_Y, dev_koordynatySOA, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j);
            }
            else
            {
                Kalkulator_MSE_ << <bloki_Kalkulator_MSE_, watki_Kalkulator_MSE_ >> > (Dev_Macierz_wejsciowa.kanal_Y, dev_koordynatySOA, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j);
            }

            ///////////////////////////////////////////////////////////wyszukanie N_HARD najblizszych latek////////////////////////////////////////////

            Najmniejsze_liczby_ << <bloki_najmniejsze_liczby, watki_najmniejsze_liczby >> > (dev_koordynatySOA, Device_tablica_ilosci_pasujacych_latek, N_HARD, tau_hard, false);

            Przepisywacz_do_tablic_transformaty_ << <bloki_Przepisywacz, watki_Przepisywacz >> > (Dev_Macierz_wejsciowa, dev_koordynatySOA, Device_tablica_ilosci_pasujacych_latek, Device_tablice_transformaty_16, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j, N_HARD);

            ////////////////////////////////////////////// pasujêce ³atki znajduj¹ce siê w "device_tablice_transformaty_32_1krok" (tylko tyle z tej tablicy ile spe³nia warunek max dopasowania) poddajemy transformacie cosinusowej 2d (ca³e ³atki), a nastêpnie transformacie 1D walsha-hadamarda "w poprzek" grupy ³atek////////////////////////////////////////                       

            DCT << <bloki_DCT_krok1, watki_DCT_krok1 >> > (Device_tablice_transformaty_16, ROZMIAR_LATKI, ROZMIAR_LATKI, Device_tablica_ilosci_pasujacych_latek, N_HARD);

            int wielkosc_extern_sh_memory = sizeof(float) * (POWIERZCHNIA_LATKI * N_HARD);
            Walsh_1D << <bloki_Walsh, watki_Walsh_krok1, wielkosc_extern_sh_memory >> > (Device_tablice_transformaty_16, Device_tablica_ilosci_pasujacych_latek, N_HARD); // przesuniêcie to indeks elentu zerowego w macierzy transformat dla danego wywo³ania kernela                                                                                                                                                                                       
            /////////////////////////////////////////////////////////////// W przekszta³conych ³atkach zerujemy wspó³czynniki których abs jest mmniejszy ni¿ Lambda_Hard_3D*SIGMA/////////////////////////

            Zerowanie_ << <bloki_Zerowanie, watki_Zerowanie >> > (Device_tablice_transformaty_16, Device_tablica_ilosci_zerowan, Device_tablica_ilosci_pasujacych_latek, wskaznik_dev_Szum_YCrCb, N_HARD);

            ////////////////////////////////////////////////////////////// Odwracamy transformaty w celu uzyskania w³aœciwego obrazu//////////////////////////////////////////////////////


            Walsh_1D << <bloki_Walsh, watki_Walsh_krok1, wielkosc_extern_sh_memory >> > (Device_tablice_transformaty_16, Device_tablica_ilosci_pasujacych_latek, N_HARD);

            DCT_odwrotna << <bloki_DCT_krok1, watki_DCT_krok1 >> > (Device_tablice_transformaty_16, ROZMIAR_LATKI, ROZMIAR_LATKI, Device_tablica_ilosci_pasujacych_latek, N_HARD);

            ///// //////////////////////////teraz trzeba poodk³adaæ l³¹tki w odpowiednie miejsca tablicy wynikowej po 1 kroku, oraz pododawaæ wartoœci iliœci niewyzerowanych w jej dzielniku
            Przepisywacz_z_tablic_transformaty_1krok_ << <bloki_Przepisywacz, watki_Przepisywacz >> > (Device_tablica_ilosci_zerowan, Dev_Obrazek_po_1kroku, Dev_Obrazek_Dzielnik, dev_koordynatySOA, Device_tablica_ilosci_pasujacych_latek, Device_tablice_transformaty_16, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j, N_HARD);

        }
    }

    DzielenieMacierzy << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (Dev_Obrazek_po_1kroku, Dev_Obrazek_Dzielnik, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami);
    Nadpisywanie_marginesow1 << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (Dev_Obrazek_po_1kroku, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego);
    Nadpisywanie_marginesow2 << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (Dev_Obrazek_po_1kroku, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego);
    Zerowanie_Macierzy << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (Dev_Obrazek_Dzielnik, wysokosc_obrazka_z_marginesami, szerokosc_obrazka_z_marginesami);

    if (sigma > 40)
    {
        tau_wien = 2000;
    }

    for (int i = 0; i < (ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania); i += p_wien)
    {
        for (int j = 0; j < (ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania); j += p_wien)
        {

            Kalkulator_MSE_ << <bloki_Kalkulator_MSE_, watki_Kalkulator_MSE_ >> > (Dev_Obrazek_po_1kroku.kanal_Y, dev_koordynatySOA, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j);


            ///////////////////////////////////////////////////////////wyszukanie N_Wien najblizszych latek////////////////////////////////////////////


            Najmniejsze_liczby_ << <bloki_najmniejsze_liczby, watki_najmniejsze_liczby >> > (dev_koordynatySOA, Device_tablica_ilosci_pasujacych_latek, N_WIEN, tau_wien, true);

            //przepisujemy ³atki z tablicy reprezentuj¹cej obrazek wejœciowego do "device_tablice_transformaty_32_1krok": 
            Przepisywacz_do_tablic_transformaty_ << <bloki_Przepisywacz, watki_Przepisywacz >> > (Dev_Macierz_wejsciowa, dev_koordynatySOA, Device_tablica_ilosci_pasujacych_latek, Device_tablice_transformaty_32_1krok, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j, N_WIEN);

            //przepisujemy ³atki z tablicy repezentuj¹cej obrazek wstêpnie odszumiony w 1 kroku do device_tablice_transformaty_32_2krok
            Przepisywacz_do_tablic_transformaty_ << <bloki_Przepisywacz, watki_Przepisywacz >> > (Dev_Obrazek_po_1kroku, dev_koordynatySOA, Device_tablica_ilosci_pasujacych_latek, Device_tablice_transformaty_32_2krok, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j, N_WIEN);

            ////////////////////////////////////////////// pasujêce ³atki znajduj¹ce siê w "device_tablice_transformaty_32_1krok" (tylko tyle z tej tablicy ile spe³nia warunek max dopasowania) poddajemy transformacie cosinusowej 2d (ca³e ³atki), a nastêpnie transformacie 1D walsha-hadamarda "w poprzek" grupy ³atek////////////////////////////////////////

            DCT << <bloki_DCT_krok2, watki_DCT_krok2 >> > (Device_tablice_transformaty_32_1krok, Device_tablice_transformaty_32_2krok, ROZMIAR_LATKI, ROZMIAR_LATKI, Device_tablica_ilosci_pasujacych_latek, N_WIEN);
            int wielkosc_extern_sh_memory = sizeof(float) * (POWIERZCHNIA_LATKI * N_WIEN);
            Walsh_1D << <bloki_Walsh, watki_Walsh_krok2, wielkosc_extern_sh_memory >> > (Device_tablice_transformaty_32_1krok, Device_tablica_ilosci_pasujacych_latek, N_WIEN);
            Walsh_1D << <bloki_Walsh, watki_Walsh_krok2, wielkosc_extern_sh_memory >> > (Device_tablice_transformaty_32_2krok, Device_tablica_ilosci_pasujacych_latek, N_WIEN);
            Filtr_Wienera << <bloki_Wien, watki_Wien >> > (Device_tablica_wartosci_fitru_wienera, Device_tablice_transformaty_32_1krok, Device_tablice_transformaty_32_2krok, Device_tablica_ilosci_pasujacych_latek, wskaznik_dev_Szum_YCrCb, N_WIEN);

            Walsh_1D << <bloki_Walsh, watki_Walsh_krok2, wielkosc_extern_sh_memory >> > (Device_tablice_transformaty_32_2krok, Device_tablica_ilosci_pasujacych_latek, N_WIEN);

            DCT_odwrotna << <bloki_DCT_krok2, watki_DCT_krok2 >> > (Device_tablice_transformaty_32_2krok, ROZMIAR_LATKI, ROZMIAR_LATKI, Device_tablica_ilosci_pasujacych_latek, N_WIEN);

            ///// //////////////////////////teraz trzeba poodk³adaæ l³¹tki w odpowiednie miejsca tablicy wynikowej po 1 kroku, oraz pododawaæ wartoœci iliœci niewyzerowanych w jej dzielniku
            Przepisywacz_z_tablic_transformaty_2krok_ << <bloki_Przepisywacz, watki_Przepisywacz >> > (Device_tablica_wartosci_fitru_wienera, Dev_Obrazek_po_2kroku, Dev_Obrazek_Dzielnik, dev_koordynatySOA, Device_tablica_ilosci_pasujacych_latek, Device_tablice_transformaty_32_2krok, ilosc_blokow_w_boku_x, szerokosc_obrazka_z_marginesami, i, j, N_WIEN);
            //   (device_tablica_wag_fitru_wienera,obrazek_po_kolejnym_kroku, obrazek_po_kolejnym_kroku_dzielnik, sigma, Tablice_koordynatLatek dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, float* device_tablice_transformaty_32_2krok, int ilosc_blokow_w_boku_x, int ilosc_blokow_w_boku_y, int szerokosc, int i, int j, int mnoznik_tablicy_transormat
        }
    }

    DzielenieMacierzy << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (Dev_Obrazek_po_2kroku, Dev_Obrazek_Dzielnik, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami);

    cudaMemcpy(wskaznik_host_Obrazek_po_2kroku->kanal_Y, wskaznik_dev_Obrazek_po_2kroku->kanal_Y, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wskaznik_host_Obrazek_po_2kroku->kanal_Cr, wskaznik_dev_Obrazek_po_2kroku->kanal_Cr, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(wskaznik_host_Obrazek_po_2kroku->kanal_Cb, wskaznik_dev_Obrazek_po_2kroku->kanal_Cb, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat Obrazek_po_2kroku(wielkosc_tablicy_z_marginesami, 1, CV_32FC3);
    
    for (int i = 0; i < wielkosc_tablicy_z_marginesami; i++)
    {
        kanaly_do_przepisania[0] = Host_Obrazek_po_2kroku.kanal_Y[i];
        kanaly_do_przepisania[1] = Host_Obrazek_po_2kroku.kanal_Cr[i];
        kanaly_do_przepisania[2] = Host_Obrazek_po_2kroku.kanal_Cb[i];
        Obrazek_po_2kroku.at<cv::Vec3f>(i, 0) = kanaly_do_przepisania;
    }

    Obrazek_po_2kroku = Obrazek_po_2kroku.reshape(0, wysokosc_obrazka_z_marginesami);

    Obrazek_po_2kroku.convertTo(Obrazek_po_2kroku, CV_8UC3);

    cv::cvtColor(Obrazek_po_2kroku, Obrazek_po_2kroku, cv::COLOR_YCrCb2RGB);

    Obrazek_po_2kroku = Obrazek_po_2kroku(cv::Rect(wielkosc_marginesu_lewego, wielkosc_marginesu_lewego, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));
    Obrazek_odszumiony = Obrazek_po_2kroku;
    
    cudaFree(wskaznik_dev_Macierz_wejsciowa->kanal_Y);
    cudaFree(wskaznik_dev_Macierz_wejsciowa->kanal_Cr);
    cudaFree(wskaznik_dev_Macierz_wejsciowa->kanal_Cb);
    wskaznik_dev_Macierz_wejsciowa->kanal_Y = nullptr;
    wskaznik_dev_Macierz_wejsciowa->kanal_Cr = nullptr;
    wskaznik_dev_Macierz_wejsciowa->kanal_Cb = nullptr;
    cudaFree(wskaznik_dev_Macierz_wejsciowa);

    cudaFree(wskaznik_dev_Obrazek_po_1kroku->kanal_Y);
    cudaFree(wskaznik_dev_Obrazek_po_1kroku->kanal_Cr);
    cudaFree(wskaznik_dev_Obrazek_po_1kroku->kanal_Cb);
    wskaznik_dev_Obrazek_po_1kroku->kanal_Y = nullptr;
    wskaznik_dev_Obrazek_po_1kroku->kanal_Cr = nullptr;
    wskaznik_dev_Obrazek_po_1kroku->kanal_Cb = nullptr;
    cudaFree(wskaznik_dev_Obrazek_po_1kroku);

    cudaFree(wskaznik_dev_Obrazek_po_2kroku->kanal_Y);
    cudaFree(wskaznik_dev_Obrazek_po_2kroku->kanal_Cr);
    cudaFree(wskaznik_dev_Obrazek_po_2kroku->kanal_Cb);
    wskaznik_dev_Obrazek_po_2kroku->kanal_Y = nullptr;
    wskaznik_dev_Obrazek_po_2kroku->kanal_Cr = nullptr;
    wskaznik_dev_Obrazek_po_2kroku->kanal_Cb = nullptr;;
    cudaFree(wskaznik_dev_Obrazek_po_2kroku);

    cudaFree(wskaznik_dev_Obrazek_Dzielnik->kanal_Y);
    cudaFree(wskaznik_dev_Obrazek_Dzielnik->kanal_Cr);
    cudaFree(wskaznik_dev_Obrazek_Dzielnik->kanal_Cb);
    wskaznik_dev_Obrazek_Dzielnik->kanal_Y = nullptr;
    wskaznik_dev_Obrazek_Dzielnik->kanal_Cr = nullptr;
    wskaznik_dev_Obrazek_Dzielnik->kanal_Cb = nullptr;
    cudaFree(wskaznik_dev_Obrazek_Dzielnik);

    cudaFree(wskaznik_device_tablice_transformaty_16->kanal_Y);
    cudaFree(wskaznik_device_tablice_transformaty_16->kanal_Cr);
    cudaFree(wskaznik_device_tablice_transformaty_16->kanal_Cb);
    wskaznik_device_tablice_transformaty_16->kanal_Y = nullptr;
    wskaznik_device_tablice_transformaty_16->kanal_Cr = nullptr;
    wskaznik_device_tablice_transformaty_16->kanal_Cb = nullptr;
    cudaFree(wskaznik_device_tablice_transformaty_16);

    cudaFree(wskaznik_device_tablice_transformaty_32_1krok->kanal_Y);
    cudaFree(wskaznik_device_tablice_transformaty_32_1krok->kanal_Cr);
    cudaFree(wskaznik_device_tablice_transformaty_32_1krok->kanal_Cb);
    wskaznik_device_tablice_transformaty_32_1krok->kanal_Y = nullptr;
    wskaznik_device_tablice_transformaty_32_1krok->kanal_Cr = nullptr;
    wskaznik_device_tablice_transformaty_32_1krok->kanal_Cb = nullptr;
    cudaFree(wskaznik_device_tablice_transformaty_32_1krok);

    cudaFree(wskaznik_device_tablice_transformaty_32_2krok->kanal_Y);
    cudaFree(wskaznik_device_tablice_transformaty_32_2krok->kanal_Cr);
    cudaFree(wskaznik_device_tablice_transformaty_32_2krok->kanal_Cb);
    wskaznik_device_tablice_transformaty_32_2krok->kanal_Y = nullptr;
    wskaznik_device_tablice_transformaty_32_2krok->kanal_Cr = nullptr;
    wskaznik_device_tablice_transformaty_32_2krok->kanal_Cb = nullptr;
    cudaFree(wskaznik_device_tablice_transformaty_32_2krok);

    cudaFree(wskaznik_dev_koordynatySOA->MSE);
    cudaFree(wskaznik_dev_koordynatySOA->koordynata_x);
    cudaFree(wskaznik_dev_koordynatySOA->koordynata_y);
    wskaznik_dev_koordynatySOA->MSE = nullptr;
    wskaznik_dev_koordynatySOA->koordynata_x = nullptr;
    wskaznik_dev_koordynatySOA->koordynata_y = nullptr;
    cudaFree(wskaznik_dev_koordynatySOA);

    cudaFree(wskaznik_device_tablica_ilosci_zerowan->kanal_Y);
    cudaFree(wskaznik_device_tablica_ilosci_zerowan->kanal_Cr);
    cudaFree(wskaznik_device_tablica_ilosci_zerowan->kanal_Cb);
    wskaznik_device_tablica_ilosci_zerowan->kanal_Y = nullptr;
    wskaznik_device_tablica_ilosci_zerowan->kanal_Cr = nullptr;
    wskaznik_device_tablica_ilosci_zerowan->kanal_Cb = nullptr;
    cudaFree(wskaznik_device_tablica_ilosci_zerowan);

    cudaFree(wskaznik_device_tablica_wag_fitru_wienera->kanal_Y);
    cudaFree(wskaznik_device_tablica_wag_fitru_wienera->kanal_Cr);
    cudaFree(wskaznik_device_tablica_wag_fitru_wienera->kanal_Cb);
    wskaznik_device_tablica_wag_fitru_wienera->kanal_Y = nullptr;
    wskaznik_device_tablica_wag_fitru_wienera->kanal_Cr = nullptr;
    wskaznik_device_tablica_wag_fitru_wienera->kanal_Cb = nullptr;
    cudaFree(wskaznik_device_tablica_wag_fitru_wienera);
    
    cudaFree(Device_tablica_ilosci_pasujacych_latek);
    cudaFree(&wskaznik_dev_Szum_YCrCb);
    wskaznik_dev_Szum_YCrCb = nullptr;

    free(wskaznik_host_Macierz_wejsciowa->kanal_Y);
    free(wskaznik_host_Macierz_wejsciowa->kanal_Cr);
    free(wskaznik_host_Macierz_wejsciowa->kanal_Cb);
    wskaznik_host_Macierz_wejsciowa->kanal_Y = nullptr;
    wskaznik_host_Macierz_wejsciowa->kanal_Cr = nullptr;
    wskaznik_host_Macierz_wejsciowa->kanal_Cb = nullptr;

    free(wskaznik_host_Obrazek_po_2kroku->kanal_Y);
    free(wskaznik_host_Obrazek_po_2kroku->kanal_Cr);
    free(wskaznik_host_Obrazek_po_2kroku->kanal_Cb);
    wskaznik_host_Obrazek_po_2kroku->kanal_Y = nullptr;
    wskaznik_host_Obrazek_po_2kroku->kanal_Cr = nullptr;
    wskaznik_host_Obrazek_po_2kroku->kanal_Cb = nullptr;

    free(wskaznik_host_Obrazek_po_1kroku->kanal_Y);
    free(wskaznik_host_Obrazek_po_1kroku->kanal_Cr);
    free(wskaznik_host_Obrazek_po_1kroku->kanal_Cb);
    wskaznik_host_Obrazek_po_1kroku->kanal_Y = nullptr;
    wskaznik_host_Obrazek_po_1kroku->kanal_Cr = nullptr;
    wskaznik_host_Obrazek_po_1kroku->kanal_Cb = nullptr;

}

int main(int argc, char* argv[])
{

    //_putenv_s("CUDA_DEVICE_MAX_CONNECTIONS", "32");
    
    float sigma = SIGMA;
    cv::Mat Obrazek;
    cv::Mat Obrazek_odszumiony;
    int tryb_szybkosci = 1;
    int opcja_obrazka = 1;
    int licznik = 0;
    std::string wpisana_nazwa;

	if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
		{
			std::cout << "Filtr BM3D GPU, filtruje z szumu obrazy kolorowe.\n";
			std::cout << "Uzycie: BM3D_Color_GPU.exe <liczba calkowita>  <liczba calkowita lub zmiennoprzecinkowa>\n ";
			std::cout << "Argumenty:\n";
			std::cout << "  <nazwa pliku>         Nazwa pliku. Mozna podac nazwe i sciezke folderu lub sama nazwe \n";
			std::cout << "                        jezeli znajduje sie w jednym folderze z programem\n";
			std::cout << "                        -zostan¹ przetworzone wszystkie pliki graficzne w folderze\n";
			std::cout << "  <poziom szumu>        Liczba calkowita: 0 do 100\n";
			std::cout << "  <stala filtracji>	Liczba calkowita: sila odzumiania";
			return 0;
		}

    if (argc != 4) 
	{
        std::cerr << "U¿ycie: " << argv[0] << " <nazwa pliku> <poziom szumu> <normalny czy szybki?>\n pomoc: --help lub -h";
        cv::waitKey(0);
        return 1;
    }

    wpisana_nazwa = argv[1];       //nazwa wczytywanego pliku argv[0] to nazwa programu
    sigma = std::atoi(argv[2]);  // drugi arg - poziom szumu
    tryb_szybkosci = std::atoi(argv[3]); // szybkosc 0 co 3, 1 co 4

    int devID = 0;
    initializeCUDA(argc, argv, devID);
    cudaDeviceReset();

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
        funkcja_glowna(Obrazek, Obrazek_odszumiony, sigma, tryb_szybkosci);
        for (std::filesystem::path plik : {std::filesystem::absolute(wpisana_nazwa)})
        {
            std::string nowa_nazwa = plik.stem().string() + "_filtered" + plik.extension().string();
            std::string nowa_nazwa_i_sciezka = plik.parent_path().string() + "/" + nowa_nazwa;
            std::cout << plik.parent_path().string() << std::endl;
            std::cout << nowa_nazwa_i_sciezka << std::endl;
            // Zapis przetworzonego obrazu
            //cv::imshow("Obrazek po 2 kroku", Obrazek_odszumiony);
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
                Obrazek = cv::imread(entry.path().string(), cv::IMREAD_COLOR);

                if (Obrazek.empty())
                {
                    std::cerr << "Nie mozna wczytac obrazka do odszumienia." << std::endl;
                    cv::waitKey(0);
                    return -1;
                }
                funkcja_glowna(Obrazek, Obrazek_odszumiony, sigma, tryb_szybkosci);
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
                            std::cerr << "Utworzono folder: " << nowa_sciezka << std::endl;
                        }
                        else {
                            std::cerr << "Nie udalo siê utworzyc folderu: " << nowa_sciezka << std::endl;
                            return 1; // Zakoñczenie programu z b³êdem
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
        std::cerr << "Podana œcie¿ka lub nazwa pliku  jest bledna" << std::endl;
        return 1;
    }
    time_t czasStop = clock();
    double czas = (double)(czasStop - czasStart) / (double)CLOCKS_PER_SEC;
    std::cout << "Przefiltrowano " << licznik << " obrazow w czasie : " << czas << " s." << std::endl;


    cv::waitKey(0);

    cudaDeviceReset();

    return 0;

}