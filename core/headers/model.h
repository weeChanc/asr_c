#ifndef MODEL_H
#define MODEL_H
#include "iostream"
#include "math.h"
#include "stdlib.h"
#include <string>
#include <fstream>
#include <stdio.h>
using namespace std;


#define time_steps 100   //ʱ�䲽
int *model(double (**input_data),int fram_old,int fram_new,int (*label),double(**G_W_Z),
           double(**G_U_Z), double(**G_W_R), double(**G_U_R), double(**G_W_H), double(**G_U_H),
           double(*G_B_Z), double(*G_B_R), double(*G_B_H),double(**F_W_Z), double(**F_U_Z),
           double(**F_W_R), double(**F_U_R), double(**F_W_H), double(**F_U_H), double(*F_B_Z),
           double(*F_B_R), double(*F_B_H),double(**B_W_Z), double(**B_U_Z), double(**B_W_R),
           double(**B_U_R), double(**B_W_H), double(**B_U_H), double(*B_B_Z), double(*B_B_R),
           double(*B_B_H),double(**gru_kernel), double(**gru_r_kernel), double(*gru_bias),
           double(**fw_gru_kernel), double(**fw_gru_r_kernel), double(*fw_gru_bias),
           double(**bw_gru_kernel), double(**bw_gru_r_kernel), double(*bw_gru_bias));

#endif
