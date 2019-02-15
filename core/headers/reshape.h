//
// Created by c on 2019/2/14.
//
#include "iostream"
#ifndef ASR_RESHAPE_H
#define ASR_RESHAPE_H
using namespace std;
void reshape(double **mfcc39, int maxlen, int dim, int frameNum, double **x_test);
void trimming(vector<int> &newlist, vector<int> &copylist, int *dict_out, int length, int hopStep);
#endif //ASR_RESHAPE_H
