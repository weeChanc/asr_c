#include<iostream>
#include<fstream>
#include <math.h>
#include<complex>
#include<vector>

using namespace std;

void reshape(double **mfcc39, int maxlen, int dim, int frameNum, double **x_test) {
    int overlap = maxlen / 2;
    int num_100 = ceil(frameNum / (overlap * 1.0)) - 1;
    int test_num = num_100 * maxlen;
    double mean[39] = {0};
    double stdd[39] = {0};
    //标准化
    for (int i = 0; i < 39; i++) {
        double sum = 0, sum_std = 0;
        //均值
        for (int j = 0; j < frameNum; j++) {
            sum = mfcc39[j][i] + sum;
        }
        mean[i] = sum / (frameNum * 1.0);
        //标准差
        for (int t = 0; t < frameNum; t++) {
            sum_std = pow(mfcc39[t][i] - mean[i], 2) + sum_std;
        }
        stdd[i] = sqrt(sum_std / (frameNum * 1.0));
        //
        for (int y = 0; y < frameNum; y++) {
            mfcc39[y][i] = (mfcc39[y][i] - mean[i]) / stdd[i];
        }
    }
//    double d=mfcc39[326][38];
    int padlen = static_cast<int>(ceil(frameNum / (overlap * 1.0))) * overlap - frameNum;
    //定义输出数组（test_num，39）

    int start = 0, endd = 0, frame100 = 0;
    for (int i = 0; i < (num_100 - 1); i++) {
        start = i * overlap;
        endd = start + maxlen;
        for (int j = start; j < endd; j++) {
            for (int y = 0; y < 39; y++) {
                x_test[j + frame100][y] = mfcc39[j][y];
            }

        }
        frame100 += overlap;
    }
    //最后一个不足maxlen的赋值
    //double a=x_test[300][38];
    int lastnum = (num_100 - 1) * overlap;
    int lastnum1 = maxlen * (num_100 - 1);
    for (int i = 0; i < (frameNum - lastnum); i++) {
        for (int j = 0; j < 39; j++) {
            x_test[lastnum1 + i][j] = mfcc39[lastnum + i][j];
        }
    }
//    ofstream filex_test("D:\\CodeBlocks\\projext\\mfcc_zhou\\zhou\\bin\\x_test.dat");
//    for (int j = 0; j < test_num; j++)//write DCT
//	{
//	    for (int i = 0; i < 39; i++)
//		    filex_test << x_test[j][i] << " ";
//		filex_test << endl;
//	}
    //   return x_test;

}

/////////对输出序列进行处理//////
void trimming(vector<int> &newlist, vector<int> &copylist, int *dict_out, int length, int hopStep) {
//	int length = sizeof(dict_out)/sizeof(int);
//	int *copylist=new int[length];//计算所占帧数

    int num = 0, lastnum = 100;
//	int newlist[length]={0};//输出音素序列
    for (int j = 0, i = 0; i < length; i++) {
        if (lastnum == dict_out[i]) {
            num++;
        } else {
            newlist.push_back(dict_out[i]);
            copylist.push_back(i * hopStep);//元素开始的帧数
            lastnum = dict_out[i];
        }
    }
}
