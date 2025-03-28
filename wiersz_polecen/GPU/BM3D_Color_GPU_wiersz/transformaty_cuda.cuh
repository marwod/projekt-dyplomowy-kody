#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <math.h>
#include <cmath>
#include "naglowek_struktury.h"

#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartosc w ilosci latek i uzywanych watk0w. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru latki
#define ROZMIAR_LATKI       8
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_lATKI ilosc pixeli obszaru przeszukania
#define POWIERZCHNIA_LATKI       64







__global__ void DCT(Obrazek_YCrCb Macierz1, int rozmiar_latki_x, int rozmiar_latki_y, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat);
__global__ void DCT(Obrazek_YCrCb Macierz1, Obrazek_YCrCb Macierz2, int rozmiar_latki_x, int rozmiar_latki_y, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat);// wersja przeciazona dla 2 kroku
__global__ void DCT_odwrotna(Obrazek_YCrCb Macierz, int x, int y, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat);
__global__ void Walsh_1D(Obrazek_YCrCb  Tablice_Latek_transformowanych, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transormat);
