/*
© Marcin Wodejko 2024.
marwod@interia.pl
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <cuda_texture_types.h>
#include <stdint.h>
#include <string>
#include <time.h>
#include <iostream>
#include <cmath>
#include <conio.h>
#include <math.h>
#include <cmath>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "Naglowek_struktury.h"
#include "transformaty_cuda.cuh"
#include <random>
#include <cstdlib>
#include <filesystem> // musi byc C++17 lub wyzej


//.werrsja po poprawkach 8 maja 2024, bardzo szybka. poprawiono transformate walsha


#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartoœæ w iloœci ³atek i u¿ywanych w¹tków. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru ³atki
#define ROZMIAR_LATKI       8
#define POWIERZCHNIA_LATKI       64
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_£ATKI iloœæ pixeli obszaru przeszukania
#define SIGMA       20.0f
#define LAMBDA2DHARD       0.9f
#define P_HARD  3 //p_Hard krok tworzenia latek, w oryginale 1,2 lub 3, u Lebruna 3
#define P_WIEN  3 //krok tworzenia latek, w oryginale 1, 2 lub 3, u Lebruna 3
#define N_HARD  16 //maks ilosc lek w grupie 3D
#define N_WIEN  32 //maks ilosc lek w grupie 3D
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


__global__ void Najmniejsze_liczby(Tablice_koordynatLatek koordynatySOA, int* device_tablica_ilosci_pasujacych_latek, int ilosc_najmniejszych, int tau_hard, bool krok2) // wykorzystanie algorytmu redykcji u¿ywanego zwykle do sumowania tablicy
{

    int indeks = threadIdx.x;

    int przesuniecie = blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA;
    __shared__ int s_koordynatySOA[1024];
    __shared__ float s_MSE_SOA[1024];
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

    for (int i = 0; i < ilosc_najmniejszych; i++)//ilosc najmniejszych wynosi N_HARD dla pierwszego kroku lub 32 dla drugiego st¹d w pamiêci zarezerwowano miejsce dla 32
    {

        for (int s = 512; s > 0; s >>= 1)//s zmniejszamy dwukrotnie za ka¿d¹ iteracj¹
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
            s_MSE_SOA[s_koordynatySOA[0]] = 10000000000000000000; //aby w nastepnej iteracji wyszukiwania zosta³ zigm=norowana jako kandydat do najmniejszego
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

    if (indeks < 1)
    {

        for (int i = ilosc_najmniejszych; i > 0; i = i / 2)
        {

            if (koordynatySOA.MSE[przesuniecie + i - 1] < tau_hard)
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
                //przpisujemy obszar preszukania (40 pixeli) dla ³atki do pamiêci dzielonej bloku, ze wzglêdu na zmieszczenie siê w dostêpnej w wywo³aniu funkcji iosci w¹tków musia³em zrealizowaæ przypisanie w czterech krokach.
            }

        }
    }
    __syncthreads();
    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI)) //przpisujemy obszar ³atki do ktorej bêdziemy porownywaæ do pamiêci dzielonej bloku
    {
        latka_referencyjna[row_pos * ROZMIAR_LATKI + col_pos] = obszar_preszukana_shared[(row_pos + ofset) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + (col_pos + ofset)];
        // przypisujemy wartoœci dla ³atki referencyjnej (latka o rozmiarze 8*8) dla ka¿dego obszaru przeszukania. Ofset jest potrzebny w zwi¹zku z ró¿n¹ wielkoœcia ³atki i obaszaru przeszukania oraz tym ¿e ³atka referencyjna umieszona jest w œrodku tzn jej lewy gorny róg jest umieszczony w œrodku x=19, y=19.
        //£atka nie jest umieszczona idealnie po œrodu, ale jest to kompromis który zapewnia pokrycie ca³go obszaru i wspó³pracê z 1024 w¹tkami w bloku.
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
                //__syncthreads(); // ¿eby nie by³o pchania siê w¹tków jednoczeœnie do tech samych pikseli slre niestety nie dzia³¹ przez pêtlê jheœli wszystkie w¹tki nie s¹ zatrudnione
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
 
    __shared__ float latka_referencyjna[ROZMIAR_LATKI * ROZMIAR_LATKI];
    __shared__ float latka_porownywana[ROZMIAR_LATKI * ROZMIAR_LATKI];
    __shared__ float Macierz_wynikowa_posrednia[POWIERZCHNIA_LATKI];

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

    if ((threadIdx.y < ROZMIAR_LATKI) && (threadIdx.x < ROZMIAR_LATKI))
    {
        float zmienna_sumowana = ((latka_referencyjna[index2dLatki] - latka_porownywana[index2dLatki]) * (latka_referencyjna[index2dLatki] - latka_porownywana[index2dLatki]));
        latka_referencyjna[index2dLatki] = zmienna_sumowana;

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            float suma = 0;
            for (int i = 0; i < 64; i++)//atomic add dla floatów jest wolniejsze
            {
                suma = suma + latka_referencyjna[i];
            }
            dev_koordynatySOA.MSE[index_elmentu_zero_tablicy_koordynat + (blockIdx.x + blockIdx.y * 32)] = suma / POWIERZCHNIA_LATKI;
            dev_koordynatySOA.koordynata_x[index_elmentu_zero_tablicy_koordynat + blockIdx.x + blockIdx.y * 32] = blockIdx.x;
            dev_koordynatySOA.koordynata_y[index_elmentu_zero_tablicy_koordynat + blockIdx.x + blockIdx.y * 32] = blockIdx.y;
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
    int index_x_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z % ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + i;//przetestowaæ czy blo z czy y czy jeden i drugi!!!!!!!
    int index_y_pixela_gorny_lewy_obszaru_przeszukania = (blockIdx.z / ilosc_blokow_w_boku_x) * RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA + j;
    int index_2d_latki = col_pos + (row_pos * ROZMIAR_LATKI);
    int index_elmentu_zero_tablicy_koordynat = (blockIdx.z * ROZMIAR_OBSZARU_PRZESZUKANIA * ROZMIAR_OBSZARU_PRZESZUKANIA);
    int index_elmentu_zero_tablicy_transformat = (blockIdx.z * ROZMIAR_LATKI * ROZMIAR_LATKI * mnoznik_tablicy_transormat);
    float ilosc_niewyzerowanych = tablica_ilosci_zerowan[blockIdx.z];
    if ((row_pos < ROZMIAR_LATKI) && (col_pos < ROZMIAR_LATKI))
    {
        //sz_Macierz_wspolczynnikow_Kaizerra[threadIdx.y * ROZMIAR_LATKI + threadIdx.x] = Macierz_wspolczynnikow_Kaizerra[threadIdx.y * ROZMIAR_LATKI + threadIdx.x];
        __syncthreads();
        
        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//³atka po ³atce przepsisujemy ³atki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
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
        for (int i = 0; i < device_tablica_ilosci_pasujacych_latek[blockIdx.z]; i++)//³atka po ³atce przepsisujemy ³atki z  tablicy transformat device_tablice_transformaty_32_1krok(dla 1 kroku, dl 2 kroku device_tablice_transformaty_32 do t
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
            device_tablica_wartosci_fitru_wiena[blockIdx.z] = 1.0 / (device_tablica_wartosci_fitru_wiena[blockIdx.z] * 32);
        }

    }
    */

}

__global__ void Nadpisywanie_marginesow1(float* device_obrazek_po1kroku, float* device_obrazek_po1kroku_dzielnik, int szerokosc, int wysokosc, int margines_lewy, int margines_prawy) //doanaie nowych marginesów
{

    int row_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int col_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int margines = margines_lewy + margines_prawy;

    if (row_pos < szerokosc && col_pos < margines_lewy)

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

__global__ void Nadpisywanie_marginesow2(float* device_obrazek_po1kroku, float* device_obrazek_po1kroku_dzielnik, int szerokosc, int wysokosc, int margines_lewy, int margines_prawy) //dzielenie wyiku sumowania zerowanych ³atek zprzez ilosc zerowañ oraz doanaie nowych marginesów
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



__global__ void DzielenieMacierzy(float* device_obrazek_po_n_kroku, float* __restrict__ device_obrazek_po_n_kroku_dzielnik, int szerokosc, int wysokosc, int margines_lewy, int margines_prawy) //dzielenie wyiku sumowania zerowanych ³atek zprzez ilosc zerowañ
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
    double sigma = sigm; // Wartoœæ sigma dla szumu gaussowskiego
    // Generator liczb losowych dla szumu gaussowskiego
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, sigma);
    // Dodaje szum gaussowski do ka¿dego piksela
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

void funkcja_glowna (cv::Mat Obrazek_zaszumiony, cv::Mat &Obrazek_odszumiony, float sigma, int szybkosc)
{
    int p_hard = P_HARD; //przesyniêcie pomidzy ³atkami w kroku 1, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
    int p_wien = P_WIEN;//przesuniêcie pomiêdzy ³atkami w kroku 2, w oryginale 1,2 lub 3, u Lebruna wynosi 3, w orginalnym opisie maksymalnie 4
    int zakladka_obszaru_przeszukania = 2;
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


    int szerokosc_obrazka_oryginalnego = Obrazek_zaszumiony.cols;
    int wysokosc_obrazka_oryginalnego = Obrazek_zaszumiony.rows;
    int wielkosc_marginesu_lewego = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA / 2;
    int wielkosc_marginesu_prawego = RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA;

    cv::copyMakeBorder(Obrazek_zaszumiony, Obrazek_zaszumiony, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego, cv::BORDER_REFLECT_101);
    Obrazek_zaszumiony.convertTo(Obrazek_zaszumiony, CV_32F);
    int szerokosc_obrazka_z_marginesami = szerokosc_obrazka_oryginalnego + wielkosc_marginesu_lewego + wielkosc_marginesu_prawego;
    int wysokosc_obrazka_z_marginesami = wysokosc_obrazka_oryginalnego + wielkosc_marginesu_lewego + wielkosc_marginesu_prawego;
    int wielkosc_tablicy_z_marginesami = szerokosc_obrazka_z_marginesami * wysokosc_obrazka_z_marginesami;

    Obrazek_zaszumiony = Obrazek_zaszumiony.reshape(1, wielkosc_tablicy_z_marginesami);
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


    host_obrazek_poczatkowy = (float*)Obrazek_zaszumiony.data;

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


    ////////////////////////////////////////////przygotowanie i lokowanie w pamiêci tablic pomocniczych////////////////////////////////
    /////////lokujemy je w pamiêci przed rozpoczêciem wykonywania programu przez kartê gdy¿ dynamiczne lokowanie pamiêci przez CUDÊ wielokrotnie spowalnia program///////////////


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
    for (int i = 0; i < (ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania); i += p_hard)
    {
        for (int j = 0; j < (ROZMIAR_OBSZARU_PRZESZUKANIA + zakladka_obszaru_przeszukania); j += p_hard)
        {
            //int wielkosc_tabeli_koordynat = (ROZMIAR_OBSZARU_PRZESZUKANIA) * (ROZMIAR_OBSZARU_PRZESZUKANIA);
            int szerokosc = szerokosc_obrazka_z_marginesami;
            int tau_hard = TAU_HARD_NISKI;
            if (sigma > 40)
            {
                tau_hard = TAU_HARD_WYSOKI * 3;

                Kalkulator_MSE_szum_duzy << <bloki_Kalkulator_MSE_szum_duzy, watki_Kalkulator_MSE_szum_duzy >> > (device_obrazek_poczatkowy, dev_koordynatySOA, ilosc_blokow_w_boku_x, szerokosc, i, j, sigma);

                //Kalkulator_MSE << <bloki_Kalkulator_MSE, watki_Kalkulator_MSE >> > (device_obrazek_poczatkowy, dev_koordynatySOA, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j);

            }
            else
            {
                Kalkulator_MSE << <bloki_Kalkulator_MSE, watki_Kalkulator_MSE >> > (device_obrazek_poczatkowy, dev_koordynatySOA, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j);
            }

            ///////////////////////////////////////////////////////////wyszukanie N_HARD najblizszych latek////////////////////////////////////////////

            Najmniejsze_liczby << <bloki_najmniejsze_liczby, watki_najmniejsze_liczby >> > (dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, N_HARD, tau_hard, false);
            Przepisywacz_do_tabloc_transformaty << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_obrazek_poczatkowy, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_16, ilosc_blokow_w_boku_x, 0, szerokosc, i, j, N_HARD);

            ////////////////////////////////////////////// pasujêce ³atki znajduj¹ce siê w "device_tablice_transformaty_32_1krok" (tylko tyle z tej tablicy ile spe³nia warunek max dopasowania) poddajemy transformacie cosinusowej 2d (ca³e ³atki), a nastêpnie transformacie 1D walsha-hadamarda "w poprzek" grupy ³atek////////////////////////////////////////
            DCT << <bloki_DCT_krok1, watki_DCT_krok1 >> > (device_tablice_transformaty_16, device_tablice_transformaty_32_2krok, ROZMIAR_LATKI, ROZMIAR_LATKI, device_tablica_ilosci_pasujacych_latek, N_HARD, false);
            Walsh1dPojedyncza << <bloki_Walsh, watki_Walsh_krok1 >> > (device_tablice_transformaty_16, device_tablice_transformaty_32_2krok, device_tablica_ilosci_pasujacych_latek, N_HARD, false);
            Zerowanie << <bloki_Zerowanie, watki_Zerowanie >> > (device_tablice_transformaty_16, device_tablica_ilosci_zerowan, device_tablica_ilosci_pasujacych_latek, sigma, N_HARD);

            ////////////////////////////////////////////////////////////// Odwracamy transformaty w celu uzyskania w³aœciwego obrazu//////////////////////////////////////////////////////

            Walsh1dPojedyncza << <bloki_Walsh, watki_Walsh_krok1 >> > (device_tablice_transformaty_16, device_tablice_transformaty_32_2krok, device_tablica_ilosci_pasujacych_latek, N_HARD, false);
            DCT_odwrotna << <bloki_DCT_krok1, watki_DCT_krok1 >> > (device_tablice_transformaty_16, ROZMIAR_LATKI, ROZMIAR_LATKI, device_tablica_ilosci_pasujacych_latek, N_HARD);
            ///// //////////////////////////teraz trzeba poodk³adaæ l³¹tki w odpowiednie miejsca tablicy wynikowej po 1 kroku, oraz pododawaæ wartoœci iliœci niewyzerowanych w jej dzielniku
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
            //przepisujemy ³atki z tablicy reprezentuj¹cej obrazek wejœciowego do "device_tablice_transformaty_32_1krok": 
            Przepisywacz_do_tabloc_transformaty << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_obrazek_poczatkowy, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_32_1krok, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j, N_WIEN);
            //przepisujemy ³atki z tablicy repezentuj¹cej obrazek wstêpnie odszumiony w 1 kroku do device_tablice_transformaty_32_2krok
            Przepisywacz_do_tabloc_transformaty << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_obrazek_po1kroku, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_32_2krok, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j, N_WIEN);
            ////////////////////////////////////////////// pasujêce ³atki znajduj¹ce siê w "device_tablice_transformaty_32_1krok" (tylko tyle z tej tablicy ile spe³nia warunek max dopasowania) poddajemy transformacie cosinusowej 2d (ca³e ³atki), a nastêpnie transformacie 1D walsha-hadamarda "w poprzek" grupy ³atek////////////////////////////////////////
            DCT << <bloki_DCT_krok2, watki_DCT_krok2 >> > (device_tablice_transformaty_32_1krok, device_tablice_transformaty_32_2krok, ROZMIAR_LATKI, ROZMIAR_LATKI, device_tablica_ilosci_pasujacych_latek, N_WIEN, true);
            Walsh1dPojedyncza << <bloki_Walsh, watki_Walsh_krok2 >> > (device_tablice_transformaty_32_1krok, device_tablice_transformaty_32_2krok, device_tablica_ilosci_pasujacych_latek, N_WIEN, true); // przesuniêcie to indeks elentu zerowego w macierzy transformat dla danego wywo³ania kernela                                                                                                                      
            /////////////////////////////////////////////////////////////// W przekszta³conych ³atkach zerujemy wspó³czynniki których abs jest mmniejszy ni¿ Lambda_Hard_3D*SIGMA/////////////////////////

            Filtr_Wiena << <bloki_Wien, watki_Wien >> > (device_tablica_wartosci_fitru_wiena, device_tablice_transformaty_32_1krok, device_tablice_transformaty_32_2krok, device_tablica_ilosci_pasujacych_latek, sigma, N_WIEN);
            Walsh1dPojedyncza << <bloki_Walsh, watki_Walsh_krok2 >> > (device_tablice_transformaty_32_2krok, device_tablice_transformaty_32_1krok, device_tablica_ilosci_pasujacych_latek, N_WIEN, false);
            cudaFuncSetCacheConfig(DCT_odwrotna, cudaFuncCachePreferShared);
            DCT_odwrotna << <bloki_DCT_krok2, watki_DCT_krok2 >> > (device_tablice_transformaty_32_2krok, ROZMIAR_LATKI, ROZMIAR_LATKI, device_tablica_ilosci_pasujacych_latek, N_WIEN);

            Przepisywacz_z_tablic_transformaty_2krok << <bloki_Przepisywacz, watki_Przepisywacz >> > (device_tablica_wartosci_fitru_wiena, device_obrazek_po2kroku, device_obrazek_po2kroku_dzielnik, sigma, dev_koordynatySOA, device_tablica_ilosci_pasujacych_latek, device_tablice_transformaty_32_2krok, ilosc_blokow_w_boku_x, ilosc_blokow_w_boku_y, szerokosc, i, j, N_WIEN);
        }
    }


    DzielenieMacierzy << <bloki_dzielnie_Macierzy, watki_dzielnie_Macierzy >> > (device_obrazek_po2kroku, device_obrazek_po2kroku_dzielnik, szerokosc_obrazka_z_marginesami, wysokosc_obrazka_z_marginesami, wielkosc_marginesu_lewego, wielkosc_marginesu_prawego);
    cudaMemcpy(host_obrazek_po2kroku, device_obrazek_po2kroku, wielkosc_tablicy_z_marginesami * sizeof(float), cudaMemcpyDeviceToHost);

   cv::Mat testDataMat2(wysokosc_obrazka_z_marginesami, szerokosc_obrazka_z_marginesami, CV_32F, host_obrazek_po2kroku);
  
    testDataMat2.convertTo(testDataMat2, CV_8U);
    Obrazek_odszumiony = testDataMat2(cv::Rect(wielkosc_marginesu_lewego, wielkosc_marginesu_lewego, szerokosc_obrazka_oryginalnego, wysokosc_obrazka_oryginalnego));



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
;
}

int main(int argc, char** argv)
{
    float sigma = SIGMA;
    cv::Mat Obrazek;
    cv::Mat Obrazek_odszumiony;
    int tryb_szybkosci = 1;
    int opcja_obrazka=1;
    int licznik = 0;
    std::string wpisana_nazwa;
	
	if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
    {
        std::cout << "Filtr NLM GPU, filtruje z szumu obrazy w skali szarosci.\n";
        std::cout << "Uzycie: BM3D_Gray_GPU.exe <liczba calkowita>  <liczba calkowita lub zmiennoprzecinkowa>\n ";
        std::cout << "Argumenty:\n";
        std::cout << "  <nazwa pliku>         Nazwa pliku. Mozna podac nazwe i sciezke folderu lub sama nazwe \n";
        std::cout << "                        jezeli znajduje sie w jednym folderze z programem\n";
        std::cout << "                        -zostan¹ przetworzone wszystkie pliki graficzne w folderze\n";
        std::cout << "  <poziom szumu>        Liczba calkowita: 0 do 100\n";
        std::cout << "  <stala filtracji>	Liczba calkowita: sila odzumiania";
        return 0;
    }

    if (argc != 4) {
        std::cerr << "U¿ycie: " << argv[0] << " <nazwa pliku> <poziom szumu> <normalny czy szybki?>\n pomoc: --help lub -h";
        cv::waitKey(0);
        return 1;
    }

    wpisana_nazwa = argv[1];       //nazwa wczytywanego pliku argv[0] to nazwa programu
    sigma = std::atoi(argv[2]);  // drugi arg - poziom szumu
    tryb_szybkosci = std::atoi(argv[3]); // Argument float

    int devID = 0;
    initializeCUDA(argc, argv, devID);
    cudaDeviceReset();
    
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
        funkcja_glowna(Obrazek, Obrazek_odszumiony, sigma, tryb_szybkosci);
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
                Obrazek = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

                if (Obrazek.empty())
                {
                    std::cerr << "Nie mozna wczytac obrazka do odszumienia." << std::endl;
                    cv::waitKey(0);
                    return -1;
                }
                funkcja_glowna(Obrazek, Obrazek_odszumiony, sigma, tryb_szybkosci);
                for (std::filesystem::path plik : {std::filesystem::absolute(std::filesystem::path(entry))})
                {
                    std::string nowa_nazwa = plik.stem().string() + "_filtered" + plik.extension().string();
                    std::string nowa_sciezka = plik.parent_path().string() + "/" +"filtered";
                    std::string nowa_nazwa_i_sciezka = nowa_sciezka + "/" + nowa_nazwa;

                    if (!std::filesystem::exists(nowa_sciezka)) //sprawdza czy istnieje folder do apisania wynikowych obrazow
                    {
                        if (std::filesystem::create_directories(nowa_sciezka)) {
                            std::cout << "Utworzono folder: " << nowa_sciezka << std::endl;
                        }
                        else {
                            std::cerr << "Nie uda³o siê utworzyæ folderu: " << nowa_sciezka << std::endl;
                            return 1;
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
    //getch();

    return 0;

}