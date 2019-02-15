#include "iostream"
#include "math.h"
#include "stdlib.h"
#include <string>
#include <fstream>
#include <stdio.h>
#include "constant.h"
using namespace std;

#define time_steps 100   //ʱ�䲽

//�����
double sigmoid(double x){
	return 1.0 / (1.0 + exp(-x));
}

//���������ֵ����
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

int init_weights(int innode, int hidenode, double(**W_Z), double(**U_Z), double(**W_R), double(**U_R), double(**W_H), double(**U_H), double(*B_Z), double(*B_R), double(*B_H), double(**gru_kernel),double(**gru_r_kernel),double(*gru_bias),string path_gru_bias, string path_gru_kernel, string path_gru_r_kernel) {
	//����Ȩ��
	ifstream gru_w;//�����ȡ�ļ���������ڳ�����˵��in
	gru_w.open(path_gru_kernel);//���ļ�
	for (int i = 0; i < innode; i++)
	{
		for (int j = 0; j < hidenode * 3; j++)
		{
			gru_w >> gru_kernel[i][j];  //��������ֲ�
		}
	}
	gru_w.close();//��ȡ���֮��ر��ļ�

	//--------------------------------------------------------------------
	ifstream gru_r_w;//�����ȡ�ļ���������ڳ�����˵��in
	gru_r_w.open(path_gru_r_kernel);//���ļ�
	for (int i = 0; i < hidenode; i++)
	{
		for (int j = 0; j < hidenode * 3; j++)
		{
			gru_r_w >> gru_r_kernel[i][j];  //��������ֲ�
		}
	}
	gru_r_w.close();//��ȡ���֮��ر��ļ�

	//-------------------------------------------------------------------------
	ifstream gru_b;//�����ȡ�ļ���������ڳ�����˵��in
	gru_b.open(path_gru_bias);//���ļ�
	for (int i = 0; i < hidenode * 3; i++)
	{
		gru_b >> gru_bias[i];
	}
	gru_b.close();//��ȡ���֮��ر��ļ�
	//--------------------------------------------------------------------
	//�����Ӧ����Ȩ�س�ʼ��
	for (int i = 0; i < innode; i++) {
		for (int j = 0; j < hidenode; j++) {
			W_Z[i][j] = gru_kernel[i][j];
			W_R[i][j] = gru_kernel[i][j + hidenode];
			W_H[i][j] = gru_kernel[i][j + hidenode * 2];
		}
	}

	//���ض�Ӧ����Ȩ�س�ʼ��
	for (int i = 0; i < hidenode; i++) {
		for (int j = 0; j < hidenode; j++) {
			U_Z[i][j] = gru_r_kernel[i][j];
			U_R[i][j] = gru_r_kernel[i][j + hidenode];
			U_H[i][j] = gru_r_kernel[i][j + hidenode * 2];
		}
	}

	//ƫ�ó�ʼ��
	for (int i = 0; i < hidenode; i++) {
		B_Z[i] = gru_bias[i];
		B_R[i] = gru_bias[i + hidenode];
		B_H[i] = gru_bias[i + hidenode * 2];
	}
	return 0;
}

int release_weights(int innode, int hidenode, double(**W_Z), double(**U_Z), double(**W_R), double(**U_R), double(**W_H), double(**U_H), double(*B_Z), double(*B_R), double(*B_H), double(**gru_kernel), double(**gru_r_kernel), double(*gru_bias) ) {
	//�ͷ��ڴ�
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

	//vector<double*> Z_vector;      //������
	//vector<double*> R_vector;      //������
	//vector<double*> H_vector;      //�¼���
	//vector<double*> out_vector;      //���ֵ

	//��0ʱ����û��֮ǰ��������ģ����Գ�ʼ��һ��ȫΪ0��
	//double *out = new double[hidenode];     //���ֵ
	double *out_pre = new double[hidenode];
	if (step%100 == 0)
	{
		for (int i = 0; i < hidenode; i++)  //���ֵ��ʼ��
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
	//out_vector.push_back(out);    //��h�ŵ�h_vector�������
	//}

	//���򴫲�
	double *z_gate = new double[hidenode];     //������
	double *r_gate = new double[hidenode];     //������
	double *h_gate = new double[hidenode];     //�¼�����
	double *h_hat = new double[hidenode];

	for (int j = 0; j < hidenode; j++)
	{
		//�����ת��������
		double zGate = 0.0;   //�����ŵ�Ԫ��ʼ��
		double rGate = 0.0;  //�����ŵ�Ԫ��ʼ��
		//double *out_pre = out_vector.back();  //h(t-1)ȡout_vector��������ֵ

		for (int m = 0; m < innode; m++) //�������Ȩ��
		{
			zGate += x[m] * W_Z[m][j];
			rGate += x[m] * W_R[m][j];
		}

		for (int m = 0; m < hidenode; m++)  //h(t-1)����Ȩ�ز������������Ȩ�ص�ֵ
		{
			zGate += out_pre[m] * U_Z[m][j] ;
			rGate += out_pre[m] * U_R[m][j] ;
		}

		z_gate[j] = sigmoid(zGate + B_Z[j]);  //�����ŵ�Ԫ����
		r_gate[j] = sigmoid(rGate + B_R[j]);  //�����ŵ�Ԫ����
	}

	if (hidenode != 39) {
		for (int j = 0; j < hidenode; j++) {
			double hGate = 0.0;  //���䵥Ԫ��ʼ��
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
			double hGate = 0.0;  //���䵥Ԫ��ʼ��
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
		//cout << sum;
	}

	//�������
	//outstack[step] = out;
	//out_vector.push_back(out);
	//cout << "Phone label " << "ʱ�䲽" << ":" << max(out,hidenode) << "\n";

	delete z_gate;     //������
	delete r_gate;     //������
	delete h_gate;     //�¼�����
	//delete out;
	delete out_pre;
	return outstack[step];
}

int *model(double (**input_data),int fram_old,int fram_new,int (*label)) {
	string path_gru_bias = Constant::ASR_BASE_PATH + "/gru_bias.txt";
	string path_gru_kernel =Constant::ASR_BASE_PATH + "/gru_kernel.txt";
	string path_gru_r_kernel =Constant::ASR_BASE_PATH + "/gru_r_kernel.txt";
	//-----------------------------------------------------------------------------------------------------
	string path_fw_gru_bias =Constant::ASR_BASE_PATH + "/fw_gru_bias.txt";
	string path_fw_gru_kernel =Constant::ASR_BASE_PATH + "/fw_gru_kernel.txt";
	string path_fw_gru_r_kernel =Constant::ASR_BASE_PATH + "/fw_gru_r_kernel.txt";
	//---------------------------------------------------------------------------------------------------
	string path_bw_gru_bias =Constant::ASR_BASE_PATH + "/bw_gru_bias.txt";
	string path_bw_gru_kernel =Constant::ASR_BASE_PATH + "/bw_gru_kernel.txt";
	string path_bw_gru_r_kernel =Constant::ASR_BASE_PATH + "/bw_gru_r_kernel.txt";
	//------------------------------------------------------------------------------------------------------
	int gru_innode = 512;
	int gru_hidenode = 39;
	int bi_gru_innode = 39;
	int bi_gru_hidenode = 256;
	//double **input_data;
	double **out1;
	double **out2;
	double **biout;
	double **out;
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	double **G_W_Z;  //����ĸ�����Ȩ��
	double **G_U_Z;  //������ĸ�����Ȩ��
	double **G_W_R;  //�������������Ȩ��
	double **G_U_R;  //�������������Ȩ��
	double **G_W_H;  //�����Ȩ��
	double **G_U_H;  //�ϸ�ʱ�䲽�����Ȩ��
	double *G_B_Z;
	double *G_B_R;
	double *G_B_H;
	double **gru_kernel;
	double **gru_r_kernel;
	double *gru_bias;
	//---------------------------------
	double **F_W_Z;  //����ĸ�����Ȩ��
	double **F_U_Z;  //������ĸ�����Ȩ��
	double **F_W_R;  //�������������Ȩ��
	double **F_U_R;  //�������������Ȩ��
	double **F_W_H;  //�����Ȩ��
	double **F_U_H;  //�ϸ�ʱ�䲽�����Ȩ��
	double *F_B_Z;
	double *F_B_R;
	double *F_B_H;
	double **fw_gru_kernel;
	double **fw_gru_r_kernel;
	double *fw_gru_bias;
	//------------------------------------
	double **B_W_Z;  //����ĸ�����Ȩ��
	double **B_U_Z;  //������ĸ�����Ȩ��
	double **B_W_R;  //�������������Ȩ��
	double **B_U_R;  //�������������Ȩ��
	double **B_W_H;  //�����Ȩ��
	double **B_U_H;  //�ϸ�ʱ�䲽�����Ȩ��
	double *B_B_Z;
	double *B_B_R;
	double *B_B_H;
	double **bw_gru_kernel;
	double **bw_gru_r_kernel;
	double *bw_gru_bias;
	//+++++++++++++++++++++++++++++++++++++++++++++++

	//��ʼ�������ŵ�Ȩ�ؾ���
	gru_kernel = (double**)malloc(sizeof(double*) * gru_innode);
	G_W_Z = (double**)malloc(sizeof(double*)*gru_innode);
	G_W_R = (double**)malloc(sizeof(double*)*gru_innode);
	G_W_H = (double**)malloc(sizeof(double*)*gru_innode);
	for (int i = 0; i < gru_innode; i++) {
		gru_kernel[i] = (double*)malloc(sizeof(double) * 3 * gru_hidenode);
		G_W_Z[i] = (double*)malloc(sizeof(double)*gru_hidenode);
		G_W_R[i] = (double*)malloc(sizeof(double)*gru_hidenode);
		G_W_H[i] = (double*)malloc(sizeof(double)*gru_hidenode);
	}
	gru_r_kernel = (double**)malloc(sizeof(double*) * gru_hidenode);
	G_U_Z = (double**)malloc(sizeof(double*)*gru_hidenode);
	G_U_R = (double**)malloc(sizeof(double*)*gru_hidenode);
	G_U_H = (double**)malloc(sizeof(double*)*gru_hidenode);
	for (int i = 0; i < gru_hidenode; i++) {
		gru_r_kernel[i] = (double*)malloc(sizeof(double) * 3 * gru_hidenode);
		G_U_Z[i] = (double*)malloc(sizeof(double)*gru_hidenode);
		G_U_R[i] = (double*)malloc(sizeof(double)*gru_hidenode);
		G_U_H[i] = (double*)malloc(sizeof(double)*gru_hidenode);
	}
	gru_bias = (double*)malloc(sizeof(double) * 3 * gru_hidenode);
	G_B_Z = (double*)malloc(sizeof(double)*gru_hidenode);
	G_B_R = (double*)malloc(sizeof(double)*gru_hidenode);
	G_B_H = (double*)malloc(sizeof(double)*gru_hidenode);
	//--------------------------------------------------------------------------------
	fw_gru_kernel = (double**)malloc(sizeof(double*) * bi_gru_innode);
	F_W_Z = (double**)malloc(sizeof(double*)*bi_gru_innode);
	F_W_R = (double**)malloc(sizeof(double*)*bi_gru_innode);
	F_W_H = (double**)malloc(sizeof(double*)*bi_gru_innode);
	for (int i = 0; i < bi_gru_innode; i++) {
		fw_gru_kernel[i] = (double*)malloc(sizeof(double) * 3 * bi_gru_hidenode);
		F_W_Z[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
		F_W_R[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
		F_W_H[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	}
	fw_gru_r_kernel = (double**)malloc(sizeof(double*) * bi_gru_hidenode);
	F_U_Z = (double**)malloc(sizeof(double*)*bi_gru_hidenode);
	F_U_R = (double**)malloc(sizeof(double*)*bi_gru_hidenode);
	F_U_H = (double**)malloc(sizeof(double*)*bi_gru_hidenode);
	for (int i = 0; i < bi_gru_hidenode; i++) {
		fw_gru_r_kernel[i] = (double*)malloc(sizeof(double) * 3 * bi_gru_hidenode);
		F_U_Z[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
		F_U_R[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
		F_U_H[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	}
	fw_gru_bias = (double*)malloc(sizeof(double) * 3 * bi_gru_hidenode);
	F_B_Z = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	F_B_R = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	F_B_H = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	//-------------------------------------------------------------------
	bw_gru_kernel = (double**)malloc(sizeof(double*) * bi_gru_innode);
	B_W_Z = (double**)malloc(sizeof(double*)*bi_gru_innode);
	B_W_R = (double**)malloc(sizeof(double*)*bi_gru_innode);
	B_W_H = (double**)malloc(sizeof(double*)*bi_gru_innode);
	for (int i = 0; i < bi_gru_innode; i++) {
		bw_gru_kernel[i] = (double*)malloc(sizeof(double) * 3 * bi_gru_hidenode);
		B_W_Z[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
		B_W_R[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
		B_W_H[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	}
	bw_gru_r_kernel = (double**)malloc(sizeof(double*) * bi_gru_hidenode);
	B_U_Z = (double**)malloc(sizeof(double*)*bi_gru_hidenode);
	B_U_R = (double**)malloc(sizeof(double*)*bi_gru_hidenode);
	B_U_H = (double**)malloc(sizeof(double*)*bi_gru_hidenode);
	for (int i = 0; i < bi_gru_hidenode; i++) {
		bw_gru_r_kernel[i] = (double*)malloc(sizeof(double) * 3 * bi_gru_hidenode);
		B_U_Z[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
		B_U_R[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
		B_U_H[i] = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	}
	bw_gru_bias = (double*)malloc(sizeof(double) * 3 * bi_gru_hidenode);
	B_B_Z = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	B_B_R = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	B_B_H = (double*)malloc(sizeof(double)*bi_gru_hidenode);
	//------------------------------------------------------------------------------Ȩ�س�ʼ��
	init_weights(gru_innode, gru_hidenode,G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H, G_B_Z, G_B_R, G_B_H,gru_kernel,gru_r_kernel,gru_bias, path_gru_bias,path_gru_kernel,path_gru_r_kernel);
	init_weights(bi_gru_innode, bi_gru_hidenode, F_W_Z, F_U_Z, F_W_R, F_U_R, F_W_H, F_U_H, F_B_Z, F_B_R, F_B_H, fw_gru_kernel,fw_gru_r_kernel,fw_gru_bias,path_fw_gru_bias, path_fw_gru_kernel, path_fw_gru_r_kernel);
	init_weights(bi_gru_innode, bi_gru_hidenode, B_W_Z, B_U_Z, B_W_R, B_U_R, B_W_H, B_U_H, B_B_Z, B_B_R, B_B_H, bw_gru_kernel, bw_gru_r_kernel, bw_gru_bias, path_bw_gru_bias, path_bw_gru_kernel, path_bw_gru_r_kernel);
	//����ֵ�Լ��м�������ڴ�ռ䴴��
	//-------------------------------------------------------------------------------------------------------------------------
	int fram_num = fram_new;
//	input_data = (double**)malloc(sizeof(double*) * fram_num);
//	for (int i = 0; i < fram_num; i++) {
//		input_data[i] = (double*)malloc(sizeof(double) * 39);
//	}
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
	//+++++++++++++++++++++++++++++++++++����ֵ����+++++++++++++++++++++++++++++++++++++++++++++++++++

//	ifstream mfcctxt;//�����ȡ�ļ���������ڳ�����˵��in
//	mfcctxt.open("./sa1_gengxin.txt");//���ļ�
//	//{int count =0;
//    for (int i = 0; i <fram_num ; i++)
//    {
//        for (int j = 0; j < 39 ; j++)
//        {
//            mfcctxt>> input_data[i][j]; //��������
//            //cout<<count++;
//        }
//    }
//    mfcctxt.close();

	int n = (fram_num-1) / 100;
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ǰ�����+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
				}  //��ÿ��ʱ�䲽��˫��GRU���������ƴ��
			}
			for (int i = 0; i < r; i++) {
				out[i+e*100] = gru(e,out, i+e*100, gru_innode, gru_hidenode, G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H, G_B_Z, G_B_R, G_B_H, biout[i+e*100]);
				//cout << "֡��: " << i+e*100+1 << "    ���" << ":" << max(out[i+e*100], 39) << "\n";
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
				}  //��ÿ��ʱ�䲽��˫��GRU���������ƴ��
			}
			for (int i = 0; i < time_steps; i++) {
				out[i + e * 100] = gru(e,out, i+e*100, gru_innode, gru_hidenode, G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H, G_B_Z, G_B_R, G_B_H, biout[i + e * 100]);
				//cout << "֡��: " << i + e * 100+1 << "    ���" << ":" << max(out[i + e * 100], 39) << "\n";
			/*	for (int n = 0; n < 39; n++) {
					cout << n<<":   "<<out[i+e*100][n] << '\n';
				}*/
			}
		}
	}
	//================================================================================================================================
    int remain =fram_old - (n+1)*50;
    for (int i = 0; i <= n; i++) {    //���ǻ���������֡���ָ���ԭʼ���ݵ�֡��
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
        //cout <<"֡��"<<i+1<<"��  "<< label[i]<<'\n';
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
	release_weights(gru_innode, gru_hidenode,G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H, G_B_Z, G_B_R, G_B_H,gru_kernel,gru_r_kernel,gru_bias);
	release_weights(bi_gru_innode, bi_gru_hidenode, F_W_Z, F_U_Z, F_W_R, F_U_R, F_W_H, F_U_H, F_B_Z, F_B_R, F_B_H, fw_gru_kernel,fw_gru_r_kernel,fw_gru_bias);
	release_weights(bi_gru_innode, bi_gru_hidenode, B_W_Z, B_U_Z, B_W_R, B_U_R, B_W_H, B_U_H, B_B_Z, B_B_R, B_B_H, bw_gru_kernel, bw_gru_r_kernel, bw_gru_bias);

	//getchar();
    return label;
}
//����ģ��1.7��
//100ʱ�䲽��ǰ�����0.3��
