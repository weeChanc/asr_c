#include "iostream"
#include "math.h"
#include "stdlib.h"
#include <string>
#include <fstream>
#include <stdio.h>

#ifndef MODEL_H
#define MODEL_H
using namespace std;

#define time_steps 100   //ʱ�䲽
 int *model(double (**input),int fram_old,int fram_new,int (*label));
#endif
