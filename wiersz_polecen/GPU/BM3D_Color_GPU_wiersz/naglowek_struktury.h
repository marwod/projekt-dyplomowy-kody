#pragma once
#ifndef naglowek_struktury_h
#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartosc w ilosci latek i uzywanych watk0w. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru latki
#define ROZMIAR_LATKI       8
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 


struct Tablice_koordynatLatek
{
    float* MSE;
    int* koordynata_x;
    int* koordynata_y;
};
struct Obrazek_YCrCb
{
    float* kanal_Y;
    float* kanal_Cr;
    float* kanal_Cb;
};
struct Tablice_ilosci
{
    int* kanal_Y;
    int* kanal_Cr;
    int* kanal_Cb;
};
struct Szum_YCrCb
{
    float szum_kanal_Y;
    float szum_kanal_Cr;
    float szum_kanal_Cb;
};

#endif