#pragma once
#ifndef Naglowek_struktury_h
#define Naglowek_struktury_h
#define ROZMIAR_OBSZARU_PRZESZUKANIA       32 //warto�� w ilo�ci �atek i u�ywanych w�tk�w. rozmiar w pixelax wyniesie 40 (po dodaniu rozmiaru �atki
#define ROZMIAR_LATKI       8
#define RZECZYWISTY_ROZMIAR_OBSZARU_PRZESZUKANIA 40 // ROZMIAR_PRZESZUKANIA +ROZMIAR_�ATKI ilo�� pixeli obszaru przeszukania


struct Tablice_koordynatLatek
{
    float* MSE;
    int* koordynata_x;
    int* koordynata_y;
};


void wyswietlProbny();

#endif