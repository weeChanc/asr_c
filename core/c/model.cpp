#include "iostream"
#include "math.h"
#include "stdlib.h"
#include <string>
#include <fstream>
#include <stdio.h>
#include<time.h>
#include "model.h"
using namespace std;

#define time_steps 100   //时间步

//激活函数
double sigmoid(double x){
	return 1.0 / (1.0 + exp(-x));
}

double hard_sigmoid(double x) {
	double y;
	y = (x + 1.0) / 2.0;
	if (y >= 1.0) {
		y = 1.0;
	}
	else if(y<=0.0) {
		y = 0.0;
	}
	return y;
}
//求数组最大值索引
int max(double arr[], int n) {
	int i;
	int index=0;
	double *p;
	for (i = 0, p = arr; i < n; i++) {
		if (*p < arr[i]) {
			p = arr + i;
			index = i;
		}

	}
	return index;
}

int release_weights(int innode, int hidenode, double(**W_Z), double(**U_Z), double(**W_R), double(**U_R), double(**W_H), double(**U_H), double(*B_Z), double(*B_R), double(*B_H), double(**gru_kernel), double(**gru_r_kernel), double(*gru_bias) ) {
	//释放内存
//======================================================
	for (int i = 0; i < innode; i++) {
		free(gru_kernel[i]);
		free(W_Z[i]);
		free(W_R[i]);
		free(W_H[i]);
	}
	free(gru_kernel);
	free(W_Z);
	free(W_R);
	free(W_H);
	//---------------------------------------------------------------
	for (int i = 0; i < hidenode; i++) {
		free(gru_r_kernel[i]);
		free(U_Z[i]);
		free(U_R[i]);
		free(U_H[i]);
	}
	free(gru_r_kernel);
	free(U_Z);
	free(U_R);
	free(U_H);
	//-------------------------------------------------------------------
	free(gru_bias);
	free(B_Z);
	free(B_R);
	free(B_H);
	return 0;
}

double *gru(int e,double(**outstack),int step,int innode, int hidenode, double(**W_Z), double(**U_Z), double(**W_R), double(**U_R), double(**W_H), double(**U_H), double(*B_Z), double(*B_R), double(*B_H), double *x)
{
	/**********************************************/
	double *out_pre = new double[hidenode];
	if (step%100 == 0)
	{
		for (int i = 0; i < hidenode; i++)  //输出值初始化
		{
			out_pre[i] = 0.0;
		}
	}
	else
	{
		for (int i = 0; i < hidenode; i++) {
			out_pre[i] = outstack[step - 1][i];
		}
	}
	//out_vector.push_back(out);    //将h放到h_vector的最后面
	//}

	//正向传播
	double *z_gate = new double[hidenode];     //更新门
	double *r_gate = new double[hidenode];     //重置门
	double *h_gate = new double[hidenode];     //新记忆门
	double *h_hat = new double[hidenode];

	for (int j = 0; j < hidenode; j++)
	{
		//输入层转播到隐层
		double zGate = 0.0;   //更新门单元初始化
		double rGate = 0.0;  //重置门单元初始化
		//double *out_pre = out_vector.back();  //h(t-1)取out_vector中最后面的值

		for (int m = 0; m < innode; m++) //输入乘以权重
		{
			zGate += x[m] * W_Z[m][j];
			rGate += x[m] * W_R[m][j];
		}

		for (int m = 0; m < hidenode; m++)  //h(t-1)乘以权重并加上输出乘以权重的值
		{
			zGate += out_pre[m] * U_Z[m][j] ;
			rGate += out_pre[m] * U_R[m][j] ;
		}

		z_gate[j] = sigmoid(zGate + B_Z[j]);  //更新门单元更新
		r_gate[j] = sigmoid(rGate + B_R[j]);  //重置门单元更新
	}

	if (hidenode != 39) {
		for (int j = 0; j < hidenode; j++) {
			double hGate = 0.0;  //记忆单元初始化
			//double *out_pre = out_vector.back();

			for (int m = 0; m < innode; m++) {
				hGate += x[m] * W_H[m][j];
			}
			for (int m = 0; m < hidenode; m++) {
				hGate += out_pre[m] * r_gate[m] * U_H[m][j];
			}
			h_gate[j] = tanh(hGate + B_H[j]);
			outstack[step][j] = (z_gate[j])*out_pre[j] + (1-z_gate[j]) * h_gate[j];
		}
	}
	else {
		for (int j = 0; j < hidenode; j++) {
			double hGate = 0.0;  //记忆单元初始化
			//double *out_pre = out_vector.back();

			for (int m = 0; m < innode; m++) {
				hGate += x[m] * W_H[m][j];
			}
			for (int m = 0; m < hidenode; m++) {
				hGate += out_pre[m] * r_gate[m] * U_H[m][j];
			}
			h_gate[j] = hGate + B_H[j];
		}
		int index = max(h_gate, 39);
		double sum_soft = 0.0;
		for (int j = 0; j < hidenode; j++) {
			sum_soft += exp(h_gate[j]-h_gate[index]);
			//sum_soft += exp(h_gate[j]);
		}
		double sum = 0;
		for (int j = 0; j < hidenode; j++) {
			h_hat[j] = exp(h_gate[j]-h_gate[index]) / sum_soft;
			outstack[step][j] = z_gate[j]*out_pre[j] +(1- z_gate[j]) * h_hat[j];
			sum += h_hat[j];
		}
	}


	delete z_gate;     //更新门
	delete r_gate;     //重置门
	delete h_gate;     //新记忆门
	//delete out;
	delete out_pre;
	return outstack[step];
}

int *model(double (**input_data),int fram_old,int fram_new,int (*label),double(**G_W_Z),
		   double(**G_U_Z), double(**G_W_R), double(**G_U_R), double(**G_W_H), double(**G_U_H),
		   double(*G_B_Z), double(*G_B_R), double(*G_B_H),double(**F_W_Z), double(**F_U_Z),
		   double(**F_W_R), double(**F_U_R), double(**F_W_H), double(**F_U_H), double(*F_B_Z),
		   double(*F_B_R), double(*F_B_H),double(**B_W_Z), double(**B_U_Z), double(**B_W_R),
		   double(**B_U_R), double(**B_W_H), double(**B_U_H), double(*B_B_Z), double(*B_B_R),
		   double(*B_B_H),double(**gru_kernel), double(**gru_r_kernel), double(*gru_bias),
		   double(**fw_gru_kernel), double(**fw_gru_r_kernel), double(*fw_gru_bias),
		   double(**bw_gru_kernel), double(**bw_gru_r_kernel), double(*bw_gru_bias)) {

	int gru_innode = 512;
	int gru_hidenode = 39;
	int bi_gru_innode = 39;
	int bi_gru_hidenode = 256;
	double **out1;
	double **out2;
	double **biout;
	double **out;


	//输入值以及中间输出的内存空间创建
	//-------------------------------------------------------------------------------------------------------------------------
	int fram_num = fram_new;
	out1 = (double**)malloc(sizeof(double*) * fram_num);
	out2 = (double**)malloc(sizeof(double*) * fram_num);
	out = (double**)malloc(sizeof(double*) * fram_num);
	biout = (double**)malloc(sizeof(double*) * fram_num);
	for (int i = 0; i < fram_num; i++) {
		out1[i] = (double*)malloc(sizeof(double) * 256);
		out2[i] = (double*)malloc(sizeof(double) * 256);
		out[i] = (double*)malloc(sizeof(double) * 39);
		biout[i] = (double*)malloc(sizeof(double) * 512);
	}
	clock_t mode_startTime=clock();
	int n = (fram_num-1) / 100;
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++前向计算+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	for (int e = 0; e <=  n; e++) {
		if (e == n) {
			int r = fram_num - e * 100;
			for (int i = 0; i < r; i++) {
				out1[i+e*100] = gru(e,out1, i+e*100, bi_gru_innode, bi_gru_hidenode, F_W_Z, F_U_Z, F_W_R, F_U_R, F_W_H, F_U_H, F_B_Z, F_B_R, F_B_H, input_data[i + e * 100]);
				out2[i+e*100] = gru(e,out2, i+e*100, bi_gru_innode, bi_gru_hidenode, B_W_Z, B_U_Z, B_W_R, B_U_R, B_W_H, B_U_H, B_B_Z, B_B_R, B_B_H, input_data[r - 1 - i+e*100]);
			}
			for (int i = 0; i < r; i++) {
				for (int c = 0; c < 256; c++) {
					biout[i+e*100][c] = out1[i+e*100][c];
					biout[i+e*100][c + 256] = out2[r - 1 - i+e*100][c];
				}  //对每个时间步的双向GRU的输出进行拼接
			}
			for (int i = 0; i < r; i++) {
				out[i+e*100] = gru(e,out, i+e*100, gru_innode, gru_hidenode, G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H, G_B_Z, G_B_R, G_B_H, biout[i+e*100]);
				//cout << "帧数: " << i+e*100+1 << "    标记" << ":" << max(out[i+e*100], 39) << "\n";
			}
		}
		else {
			for (int i = 0; i < time_steps; i++) {
				out1[i + e * 100] = gru(e,out1, i+e*100, bi_gru_innode, bi_gru_hidenode, F_W_Z, F_U_Z, F_W_R, F_U_R, F_W_H, F_U_H, F_B_Z, F_B_R, F_B_H, input_data[i + e * 100]);
				out2[i + e * 100] = gru(e,out2, i+e*100, bi_gru_innode, bi_gru_hidenode, B_W_Z, B_U_Z, B_W_R, B_U_R, B_W_H, B_U_H, B_B_Z, B_B_R, B_B_H, input_data[time_steps - 1 - i + e * 100]);
			}
			for (int i = 0; i < time_steps; i++) {
				for (int c = 0; c < 256; c++) {
					biout[i + e * 100][c] = out1[i + e * 100][c];
					biout[i + e * 100][c +256] = out2[time_steps - 1 - i + e * 100][c];
				}  //对每个时间步的双向GRU的输出进行拼接
			}
			for (int i = 0; i < time_steps; i++) {
				out[i + e * 100] = gru(e,out, i+e*100, gru_innode, gru_hidenode, G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H, G_B_Z, G_B_R, G_B_H, biout[i + e * 100]);
				//cout << "帧数: " << i + e * 100+1 << "    标记" << ":" << max(out[i + e * 100], 39) << "\n";
				/*	for (int n = 0; n < 39; n++) {
                        cout << n<<":   "<<out[i+e*100][n] << '\n';
                    }*/
			}
		}
	}
	//================================================================================================================================
	int remain =fram_old - (n+1)*50;
	for (int i = 0; i <= n; i++) {    //考虑滑动窗，将帧数恢复到原始数据的帧数
		if(i==0){
			for (int j = 0; j < 100; j++) {
				label[j] = max(out[j], 39);
			}
		}
		else if(i==n){
			for (int j = 0; j < remain; j++) {
				label[j + (i+1) * 50] = max(out[j + i * 100 + 50], 39);
			}
		}
		else{
			for (int j = 0; j < 50; j++) {
				label[j + (i+1) * 50] = max(out[j + i * 100 + 50], 39);
			}
		}
	}

	for (int i = 0; i < fram_old; i++) {
		//cout <<"帧数"<<i+1<<"：  "<< label[i]<<'\n';
	}
	//====================================================================================================


	for (int k = 0; k < fram_num; k++) {
		//free(input_data[k]);
		free(out1[k]);
		free(out2[k]);
		free(out[k]);
		free(biout[k]);
	}
	//free(input_data);
	free(out1);
	free(out2);
	free(out);
	free(biout);
//	release_weights(gru_innode, gru_hidenode,G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H, G_B_Z, G_B_R, G_B_H,gru_kernel,gru_r_kernel,gru_bias);
//	release_weights(bi_gru_innode, bi_gru_hidenode, F_W_Z, F_U_Z, F_W_R, F_U_R, F_W_H, F_U_H, F_B_Z, F_B_R, F_B_H, fw_gru_kernel,fw_gru_r_kernel,fw_gru_bias);
//	release_weights(bi_gru_innode, bi_gru_hidenode, B_W_Z, B_U_Z, B_W_R, B_U_R, B_W_H, B_U_H, B_B_Z, B_B_R, B_B_H, bw_gru_kernel, bw_gru_r_kernel, bw_gru_bias);

	//getchar();
	clock_t mode_endTime=clock();
	cout<<"mode运行时间为：:"<<mode_endTime-mode_startTime<<"ms"<<endl;
	return label;
}
