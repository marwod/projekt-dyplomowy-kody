#pragma once
#ifndef Naglowek_struktury_h
#define Naglowek_struktury_h
#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //wartoœæ w iloœci ³atek i u¿ywanych w¹tków. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru ³atki
#define ROZMIAR_LATKI       8
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_£ATKI iloœæ pixeli obszaru przeszukania


struct Tablice_koordynatLatek
{
    float* MSE;
    int* koordynata_x;
    int* koordynata_y;
};


void wyswietlProbny();

#endif