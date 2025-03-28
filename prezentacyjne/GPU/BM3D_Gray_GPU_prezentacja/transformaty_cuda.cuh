#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"

#include <math.h>
#include <cmath>

#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartoœc w iloœci latek i uzywanych watkow. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru latki
#define ROZMIAR_LATKI       8
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_LATKI iloœc pixeli obszaru przeszukania
#define POWIERZCHNIA_LATKI       64
#define N_WIEN     32



__global__ void DCT(float* Macierz1, float* Macierz2, int x, int y, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat, bool krok2);
__global__ void DCT_odwrotna(float* Macierz, int x, int y, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transformat);
__global__ void Walsh1dPojedyncza(float* macierz1, float* macierz2, int* device_tablica_ilosci_pasujacych_latek, int mnoznik_tablicy_transormat, bool krok2);
