#include<iostream>
#include<fstream>
#include<complex>
#include"readwav.h"
#include"model.h"
#include<vector>
//#include <constant.h>

#include "constant.h"

#include <jni.h>


using namespace std;


const int filterNum = 26;//若改需要改f[];
int sampleRate = 16000;
int maxlen = 100;
int dim = 39;
#define Win_Time 0.025//把25ms里的所有点作为一个点分析
#define Hop_Time 0.010//每隔10ms分一次帧
//#define frameSize 512//补零后的长度
#define Pi 3.1415927
int hopStep = Hop_Time * sampleRate;//隔hopStep个样本点分一次帧

extern void reshape(double **mfcc39, int maxlen, int dim, int frameNum, double **x_test);

extern void
trimming(vector<int> &newlist, vector<int> &copylist, int *dict_out, int frameNum, int hopStep);


//1.预加重
void pre_emphasizing(double *sample, int len, double factor, double *Sample) {
    Sample[0] = sample[0];
    for (int i = 1; i < len; i++) {
        //预加重过程
        Sample[i] = sample[i] - factor * sample[i - 1];
    }
}

void Hamming(double *hamWin, int hamWinSize) {
    for (int i = 0; i < hamWinSize; i++) {
        hamWin[i] = (double) (0.54 - 0.46 * cos(2 * Pi * (double) i / ((double) hamWinSize - 1)));
    }
}

//计算每一帧的功率谱
void mfccFFT(double *frameSample, double *FFTSample, int frameSize, int pos) {
    //对分帧加窗后的各帧信号进行FFT变换得到各帧的频谱
    //并对语音信号的频谱取模平方得到语音信号的功率谱
    double dataR[frameSize];
    double dataI[frameSize];
    for (int i = 0; i < frameSize; i++) {
        dataR[i] = frameSample[i + pos];
        dataI[i] = 0.0f;
    }

    int x0, x1, x2, x3, x4, x5, x6, xx, x7, x8;
    int i, j, k, b, p, L;
    float TR, TI, temp;
    /********** following code invert sequence ************/
    for (i = 0; i < frameSize; i++) {
        x0 = x1 = x2 = x3 = x4 = x5 = x6 = x7 = x8 = 0;
        x0 = i & 0x01;
        x1 = (i / 2) & 0x01;
        x2 = (i / 4) & 0x01;
        x3 = (i / 8) & 0x01;
        x4 = (i / 16) & 0x01;
        x5 = (i / 32) & 0x01;
        x6 = (i / 64) & 0x01;
        x7 = (i / 128) & 0x01;
        x8 = (i / 256) & 0x01;
        xx = x0 * 256 + x1 * 128 + x2 * 64 + x3 * 32 + x4 * 16 + x5 * 8 + x6 * 4 + x7 * 2 + x8;
        dataI[xx] = dataR[i];
    }
    for (i = 0; i < frameSize; i++) {
        dataR[i] = dataI[i];
        dataI[i] = 0;
    }

    /************** following code FFT *******************/
    for (L = 1; L <= 9; L++) { /* for(1) */
        b = 1;
        i = L - 1;
        while (i > 0) {
            b = b * 2;
            i--;
        } /* b= 2^(L-1) */
        for (j = 0; j <= b - 1; j++) /* for (2) */
        {
            p = 1;
            i = 9 - L;
            while (i > 0) /* p=pow(2,7-L)*j; */
            {
                p = p * 2;
                i--;
            }
            p = p * j;
            for (k = j; k < 512; k = k + 2 * b) /* for (3) */
            {
                TR = dataR[k];
                TI = dataI[k];
                temp = dataR[k + b];
                dataR[k] = dataR[k] + dataR[k + b] * cos(2 * Pi * p / frameSize) +
                           dataI[k + b] * sin(2 * Pi * p / frameSize);
                dataI[k] = dataI[k] - dataR[k + b] * sin(2 * Pi * p / frameSize) +
                           dataI[k + b] * cos(2 * Pi * p / frameSize);
                dataR[k + b] = TR - dataR[k + b] * cos(2 * Pi * p / frameSize) -
                               dataI[k + b] * sin(2 * Pi * p / frameSize);
                dataI[k + b] = TI + temp * sin(2 * Pi * p / frameSize) -
                               dataI[k + b] * cos(2 * Pi * p / frameSize);
            } /* END for (3) */
        } /* END for (2) */
    } /* END for (1) */
    for (i = 0; i < frameSize / 2 + 1; i++) {
        FFTSample[i + pos] = (dataR[i] * dataR[i] + dataI[i] * dataI[i]) / frameSize;
    }

}

//参数说明：frameSample为处理之后的数组，Sample为对样本进行预加重之后的数组
//          len为Sample的长度，frameSize为每帧的样本点个数，frameSampleLen为处理之后的长度
double *mfccFrame(double *frameSample, double *Sample, int *len, int frameSize, int &frameSampleLen,
                  int frameNum) {
    double *hamWin;
    int hamWinSize = sampleRate * Win_Time;//16000*0.025
    hamWin = new double[hamWinSize];
    Hamming(hamWin, hamWinSize);//计算hamming窗

//	int hopStep = Hop_Time * sampleRate;
//    int frameNum = 1+ceil((double(*len)-400)/ hopStep);//计算一共会有多少
//    int frameNum00 = ceil(double(*len) / hopStep);//计算一共会有多少帧
    int frameNum00 = frameNum;

    frameSampleLen = frameNum00 * frameSize;//经过处理之后的长度
    frameSample = new double[frameSampleLen];
    for (int i = 0; i < frameSampleLen; i++)
        frameSample[i] = 0;

    double *FFTSample = new double[frameSampleLen];
    for (int i = 0; i < frameSampleLen; i++)
        FFTSample[i] = 0;
    for (int i = 0; i < frameNum00; i++)//分帧
    {
        for (int j = 0; j < frameSize; j++) {
            if (j < hamWinSize && i * hopStep + j < *len) {
                frameSample[i * frameSize + j] = Sample[i * hopStep + j] * hamWin[j];
            } else
                frameSample[i * frameSize + j] = 0;//补0
        }

        mfccFFT(frameSample, FFTSample, frameSize, i * frameSize);
    }

//	ofstream fileFrame("D:\\CodeBlocks\\projext\\mfcc_zhou\\zhou\\bin\\Frame.txt");
//	for(int j = 0; j < frameNum; j++)
//    {
//        for(int i = 0; i < frameSize; i++)
//            fileFrame << frameSample[j * frameSize + i] << " ";
//        fileFrame<<endl;
//    }
    delete[]hamWin;
    return FFTSample;
}


void DCT(double **c, int frameNum) {
    for (int k = 0; k < frameNum; k++) {
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < filterNum; j++) {
                c[k][i] += 2 * c[k][j + 13] * cos(Pi * i * (2 * j + 1) / (2 * filterNum));
                //if(k == 0 && i ==0)
                //cout << c[0][0] << endl;
            }

            if (i == 0) {
                c[k][i] = c[k][i] * sqrt(1 / (4 * 26.0));
            } else {
                c[k][i] = c[k][i] * sqrt(1 / (2 * 26.0));
            }
        }
    }
    //cout << "c[0][0] = " << c[0][0] << endl;
}

void computeMel(double **mel, int sampleRate, double *FFTSample, int frameNum, int frameSize) {

    double f[filterNum + 2] = {0.0, 2.0, 4.0, 7.0, 10.0, 13.0, 16.0, 20.0, 24.0, 29.0, 34.0, 40.0, 46.0, 53.0, 60.0,
                               68.0, 77.0, 87.0, 97.0, 109.0, 122.0, 136.0, 152.0, 169.0, 188.0, 209.0, 231.0, 256.0};


    //计算出每个三角滤波器的输出: 对每一帧进行处理
    for (int i = 0; i < frameNum; i++) {
        for (int j = 1; j <= filterNum; j++) {
            double temp = 0;
            for (int z = 0; z < frameSize; z++) {
                if (z < f[j - 1])
                    temp = 0;
                else if (z >= f[j - 1] && z <= f[j])
                    temp = (z - f[j - 1]) / (f[j] - f[j - 1]);
                else if (z >= f[j] && z <= f[j + 1])
                    temp = (f[j + 1] - z) / (f[j + 1] - f[j]);
                else if (z > f[j + 1])
                    temp = 0;
                mel[i][j - 1 + 13] += FFTSample[i * frameSize + z] * temp;
            }
        }
    }
//    double meltest=mel[0][2];
//    cout << "mel[0][13] = " << mel[0][13] << endl;
//	ofstream fileMel("./Mel.dat");
//	for(int j = 0; j < frameN
//    um; j++)
//    {
//        for(int i = 0; i < filterNum; i++)
//            fileMel << mel[j][i] << " ";
//        fileMel<<endl;
//    }
//    for(int i = 0; i <= filterNum; i++)
//		fileMel << mel[0][i] << endl;

    //取对数
    for (int i = 0; i < frameNum; i++) {
        for (int j = 0; j < filterNum; j++) {
            if (mel[i][j + 13] >= 0.00000000001)
                mel[i][j + 13] = log(mel[i][j + 13]);
            else
                mel[i][j + 13] = 2.22e-16;
        }
    }
    //cout << "mel[0][13] = " << mel[0][13] << endl;
}

void writeToFile(int frameNum, int frameSize, double **DCT) {
    ofstream fileDCT("./DCT.dat");

    for (int j = 0; j < frameNum; j++)//write DCT
    {
        for (int i = 0; i < 39; i++)
            fileDCT << DCT[j][i] << " ";
        fileDCT << endl;
    }
}

//MFCC
void MFCC(double *sample, int len, double **mfcc39, int frameNum) {

    double factor = 0.97;//预加重参数
    double *Sample = new double[len];
    //1.预加重
    pre_emphasizing(sample, len, factor, Sample);
    //1ms

    //Sample[len],相等
//    double yujz=Sample[10];
    //计算出每帧有多少个点，然后算出最接近点的个数的2^k，使得每帧的点的个数为2^k，以便进行补0
    int frameSize = (int) pow(2, ceil(log(Win_Time * sampleRate) / log(2.0)));
    double *frameSample = NULL, *FFTSample = NULL;
    int frameSampleLen;

    //分帧、加窗、功率谱
    FFTSample = mfccFrame(frameSample, Sample, &len, frameSize, frameSampleLen, frameNum);

    delete[]Sample;
    delete[]frameSample;

    computeMel(mfcc39, sampleRate, FFTSample, frameNum, frameSize);
    delete[]FFTSample;


    DCT(mfcc39, frameNum);
    //   delete mel;
    //   writeToFile(frameNum, frameSize, mfcc39);
    //return c;
}

int
init_weights(int innode, int hidenode, double(**W_Z), double(**U_Z), double(**W_R), double(**U_R),
             double(**W_H), double(**U_H), double(*B_Z), double(*B_R), double(*B_H),
             double(**gru_kernel), double(**gru_r_kernel), double(*gru_bias), string path_gru_bias,
             string path_gru_kernel, string path_gru_r_kernel) {
    //载入权重
    ifstream gru_w;//定义读取文件流，相对于程序来说是in
    gru_w.open(path_gru_kernel);//打开文件
    for (int i = 0; i < innode; i++) {
        for (int j = 0; j < hidenode * 3; j++) {
            gru_w >> gru_kernel[i][j];  //均匀随机分布
        }
    }
    gru_w.close();//读取完成之后关闭文件

    //--------------------------------------------------------------------
    ifstream gru_r_w;//定义读取文件流，相对于程序来说是in
    gru_r_w.open(path_gru_r_kernel);//打开文件
    for (int i = 0; i < hidenode; i++) {
        for (int j = 0; j < hidenode * 3; j++) {
            gru_r_w >> gru_r_kernel[i][j];  //均匀随机分布
        }
    }
    gru_r_w.close();//读取完成之后关闭文件

    //-------------------------------------------------------------------------
    ifstream gru_b;//定义读取文件流，相对于程序来说是in
    gru_b.open(path_gru_bias);//打开文件
    for (int i = 0; i < hidenode * 3; i++) {
        gru_b >> gru_bias[i];
    }
    gru_b.close();//读取完成之后关闭文件
    //--------------------------------------------------------------------
    //输入对应的门权重初始化
    for (int i = 0; i < innode; i++) {
        for (int j = 0; j < hidenode; j++) {
            W_Z[i][j] = gru_kernel[i][j];
            W_R[i][j] = gru_kernel[i][j + hidenode];
            W_H[i][j] = gru_kernel[i][j + hidenode * 2];
        }
    }

    //隐藏对应的门权重初始化
    for (int i = 0; i < hidenode; i++) {
        for (int j = 0; j < hidenode; j++) {
            U_Z[i][j] = gru_r_kernel[i][j];
            U_R[i][j] = gru_r_kernel[i][j + hidenode];
            U_H[i][j] = gru_r_kernel[i][j + hidenode * 2];
        }
    }

    //偏置初始化
    for (int i = 0; i < hidenode; i++) {
        B_Z[i] = gru_bias[i];
        B_R[i] = gru_bias[i + hidenode];
        B_H[i] = gru_bias[i + hidenode * 2];
    }
    return 0;
}


namespace good {
    int gru_innode = 512;
    int gru_hidenode = 39;
    int bi_gru_innode = 39;
    int bi_gru_hidenode = 256;
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    double **G_W_Z;  //输入的更新门权重
    double **G_U_Z;  //隐含层的更新门权重
    double **G_W_R;  //隐含层的重置门权重
    double **G_U_R;  //隐含层的重置门权重
    double **G_W_H;  //输入的权重
    double **G_U_H;  //上个时间步输出的权重
    double *G_B_Z;
    double *G_B_R;
    double *G_B_H;
    double **gru_kernel;
    double **gru_r_kernel;
    double *gru_bias;
    //---------------------------------
    double **F_W_Z;  //输入的更新门权重
    double **F_U_Z;  //隐含层的更新门权重
    double **F_W_R;  //隐含层的重置门权重
    double **F_U_R;  //隐含层的重置门权重
    double **F_W_H;  //输入的权重
    double **F_U_H;  //上个时间步输出的权重
    double *F_B_Z;
    double *F_B_R;
    double *F_B_H;
    double **fw_gru_kernel;
    double **fw_gru_r_kernel;
    double *fw_gru_bias;
    //------------------------------------
    double **B_W_Z;  //输入的更新门权重
    double **B_U_Z;  //隐含层的更新门权重
    double **B_W_R;  //隐含层的重置门权重
    double **B_U_R;  //隐含层的重置门权重
    double **B_W_H;  //输入的权重
    double **B_U_H;  //上个时间步输出的权重
    double *B_B_Z;
    double *B_B_R;
    double *B_B_H;
    double **bw_gru_kernel;
    double **bw_gru_r_kernel;
    double *bw_gru_bias;
}

using namespace good;

void load() {
    string path_gru_bias = Constant::ASR_BASE_PATH + "/model/gru_bias.txt";
    string path_gru_kernel = Constant::ASR_BASE_PATH + "/model/gru_kernel.txt";
    string path_gru_r_kernel = Constant::ASR_BASE_PATH + "/model/gru_r_kernel.txt";
    //-----------------------------------------------------------------------------------------------------
    string path_fw_gru_bias = Constant::ASR_BASE_PATH + "/model/fw_gru_bias.txt";
    string path_fw_gru_kernel = Constant::ASR_BASE_PATH + "/model/fw_gru_kernel.txt";
    string path_fw_gru_r_kernel = Constant::ASR_BASE_PATH + "/model/fw_gru_r_kernel.txt";
    //---------------------------------------------------------------------------------------------------
    string path_bw_gru_bias = Constant::ASR_BASE_PATH + "/model/bw_gru_bias.txt";
    string path_bw_gru_kernel = Constant::ASR_BASE_PATH + "/model/bw_gru_kernel.txt";
    string path_bw_gru_r_kernel = Constant::ASR_BASE_PATH + "/model/bw_gru_r_kernel.txt";
    //------------------------------------------------------------------------------------------------------

    //+++++++++++++++++++++++++++++++++++++++++++++++

    //初始化各个门的权重矩阵
    gru_kernel = (double **) malloc(sizeof(double *) * gru_innode);
    G_W_Z = (double **) malloc(sizeof(double *) * gru_innode);
    G_W_R = (double **) malloc(sizeof(double *) * gru_innode);
    G_W_H = (double **) malloc(sizeof(double *) * gru_innode);
    for (int i = 0; i < gru_innode; i++) {
        gru_kernel[i] = (double *) malloc(sizeof(double) * 3 * gru_hidenode);
        G_W_Z[i] = (double *) malloc(sizeof(double) * gru_hidenode);
        G_W_R[i] = (double *) malloc(sizeof(double) * gru_hidenode);
        G_W_H[i] = (double *) malloc(sizeof(double) * gru_hidenode);
    }
    gru_r_kernel = (double **) malloc(sizeof(double *) * gru_hidenode);
    G_U_Z = (double **) malloc(sizeof(double *) * gru_hidenode);
    G_U_R = (double **) malloc(sizeof(double *) * gru_hidenode);
    G_U_H = (double **) malloc(sizeof(double *) * gru_hidenode);
    for (int i = 0; i < gru_hidenode; i++) {
        gru_r_kernel[i] = (double *) malloc(sizeof(double) * 3 * gru_hidenode);
        G_U_Z[i] = (double *) malloc(sizeof(double) * gru_hidenode);
        G_U_R[i] = (double *) malloc(sizeof(double) * gru_hidenode);
        G_U_H[i] = (double *) malloc(sizeof(double) * gru_hidenode);
    }
    gru_bias = (double *) malloc(sizeof(double) * 3 * gru_hidenode);
    G_B_Z = (double *) malloc(sizeof(double) * gru_hidenode);
    G_B_R = (double *) malloc(sizeof(double) * gru_hidenode);
    G_B_H = (double *) malloc(sizeof(double) * gru_hidenode);
    //--------------------------------------------------------------------------------
    fw_gru_kernel = (double **) malloc(sizeof(double *) * bi_gru_innode);
    F_W_Z = (double **) malloc(sizeof(double *) * bi_gru_innode);
    F_W_R = (double **) malloc(sizeof(double *) * bi_gru_innode);
    F_W_H = (double **) malloc(sizeof(double *) * bi_gru_innode);
    for (int i = 0; i < bi_gru_innode; i++) {
        fw_gru_kernel[i] = (double *) malloc(sizeof(double) * 3 * bi_gru_hidenode);
        F_W_Z[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
        F_W_R[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
        F_W_H[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    }
    fw_gru_r_kernel = (double **) malloc(sizeof(double *) * bi_gru_hidenode);
    F_U_Z = (double **) malloc(sizeof(double *) * bi_gru_hidenode);
    F_U_R = (double **) malloc(sizeof(double *) * bi_gru_hidenode);
    F_U_H = (double **) malloc(sizeof(double *) * bi_gru_hidenode);
    for (int i = 0; i < bi_gru_hidenode; i++) {
        fw_gru_r_kernel[i] = (double *) malloc(sizeof(double) * 3 * bi_gru_hidenode);
        F_U_Z[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
        F_U_R[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
        F_U_H[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    }
    fw_gru_bias = (double *) malloc(sizeof(double) * 3 * bi_gru_hidenode);
    F_B_Z = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    F_B_R = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    F_B_H = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    //-------------------------------------------------------------------
    bw_gru_kernel = (double **) malloc(sizeof(double *) * bi_gru_innode);
    B_W_Z = (double **) malloc(sizeof(double *) * bi_gru_innode);
    B_W_R = (double **) malloc(sizeof(double *) * bi_gru_innode);
    B_W_H = (double **) malloc(sizeof(double *) * bi_gru_innode);
    for (int i = 0; i < bi_gru_innode; i++) {
        bw_gru_kernel[i] = (double *) malloc(sizeof(double) * 3 * bi_gru_hidenode);
        B_W_Z[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
        B_W_R[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
        B_W_H[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    }
    bw_gru_r_kernel = (double **) malloc(sizeof(double *) * bi_gru_hidenode);
    B_U_Z = (double **) malloc(sizeof(double *) * bi_gru_hidenode);
    B_U_R = (double **) malloc(sizeof(double *) * bi_gru_hidenode);
    B_U_H = (double **) malloc(sizeof(double *) * bi_gru_hidenode);
    for (int i = 0; i < bi_gru_hidenode; i++) {
        bw_gru_r_kernel[i] = (double *) malloc(sizeof(double) * 3 * bi_gru_hidenode);
        B_U_Z[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
        B_U_R[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
        B_U_H[i] = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    }
    bw_gru_bias = (double *) malloc(sizeof(double) * 3 * bi_gru_hidenode);
    B_B_Z = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    B_B_R = (double *) malloc(sizeof(double) * bi_gru_hidenode);
    B_B_H = (double *) malloc(sizeof(double) * bi_gru_hidenode);

    //初始化权重
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    init_weights(gru_innode, gru_hidenode, G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H, G_B_Z, G_B_R,
                 G_B_H, gru_kernel, gru_r_kernel, gru_bias, path_gru_bias, path_gru_kernel,
                 path_gru_r_kernel);
    init_weights(bi_gru_innode, bi_gru_hidenode, F_W_Z, F_U_Z, F_W_R, F_U_R, F_W_H, F_U_H, F_B_Z,
                 F_B_R, F_B_H, fw_gru_kernel, fw_gru_r_kernel, fw_gru_bias, path_fw_gru_bias,
                 path_fw_gru_kernel, path_fw_gru_r_kernel);
    init_weights(bi_gru_innode, bi_gru_hidenode, B_W_Z, B_U_Z, B_W_R, B_U_R, B_W_H, B_U_H, B_B_Z,
                 B_B_R, B_B_H, bw_gru_kernel, bw_gru_r_kernel, bw_gru_bias, path_bw_gru_bias,
                 path_bw_gru_kernel, path_bw_gru_r_kernel);
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    //main函数到这为载入网络权重

}


extern "C" {
JNIEXPORT void JNICALL
Java_analyze_Analyze_analyze(JNIEnv *env, jclass klass, jstring jpath) {
    //输入WAV文件地址
    string path = (env)->GetStringUTFChars(jpath, nullptr);
    unsigned long wavdata_l;
    double *sample = NULL;

    sample = readwav(path, &wavdata_l);

    int frameNum = 1 + ceil((double(wavdata_l) - Win_Time * sampleRate) / hopStep);//计算所有样本点一共有多少帧
    int frame_new = (ceil(frameNum / (maxlen / 2 * 1.0)) - 1) * maxlen;
    double **mfcc39 = new double *[frameNum];
    for (int i = 0; i < frameNum; i++) {
        mfcc39[i] = new double[39];
        for (int j = 0; j < 39; j++)
            mfcc39[i][j] = 0;
    }
//9ms
    clock_t startTime = clock();
    MFCC(sample, wavdata_l, mfcc39, frameNum);
    clock_t endTime = clock();
    //741ms

    delete[]sample;
    int overlap = maxlen / 2;
    int num_100 = ceil(frameNum / (overlap * 1.0)) - 1;
    int test_num = num_100 * maxlen;
    double **xtest = new double *[test_num];
    for (int i = 0; i < test_num; i++) {
        xtest[i] = new double[39];
        for (int j = 0; j < 39; j++)
            xtest[i][j] = 0;
    }
    reshape(mfcc39, maxlen, dim, frameNum, xtest);
    //释放
    for (int i = 0; i < frameNum; i++) {
        delete[] mfcc39[i];
    }
    delete[] mfcc39;


    //==========================================
    int *label = new int[frameNum];

    //前向计算部分
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    label = model(xtest, frameNum, frame_new, label, G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H,
                  G_B_Z,
                  G_B_R, G_B_H, F_W_Z, F_U_Z, F_W_R, F_U_R, F_W_H, F_U_H, F_B_Z, F_B_R, F_B_H,
                  B_W_Z,
                  B_U_Z, B_W_R, B_U_R, B_W_H, B_U_H, B_B_Z, B_B_R, B_B_H, gru_kernel, gru_r_kernel,
                  gru_bias,
                  fw_gru_kernel, fw_gru_r_kernel, fw_gru_bias, bw_gru_kernel, bw_gru_r_kernel,
                  bw_gru_bias
    );
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    //释放
    for (int i = 0; i < test_num; i++) {
        delete[] xtest[i];
    }
    delete[] xtest;


    //=================================
    cout << wavdata_l << endl;
//	cout<<xtest[0][0];
    vector<int> newlist;//输出的序列
    vector<int> pointlist;//记录每个音素从第几个采样点开始
    trimming(newlist, pointlist, label, frameNum, hopStep);
    //10611ms

    delete[]label;
    int phone_num = newlist.size();
    //把输出写到txt文件中
    ofstream fileout(Constant::ASR_BASE_PATH + "/output.txt");
    for (int i = 0; i < phone_num; i++) {
        fileout << newlist[i] << ',';
        fileout << pointlist[i] << endl;
    }
    fileout.close();

    newlist.clear();
    pointlist.clear();
    //10919ms
}
}

extern "C" {
JNIEXPORT void JNICALL
Java_analyze_Analyze_init(JNIEnv *env, jclass klass, jstring path) {
    load();
    Constant::ASR_BASE_PATH = env->GetStringUTFChars(path, nullptr);
}
}

void tt(string);

int main() {
    load();
    tt( "/home/c/CLionProjects/untitled/sound/1551168085597_record.pcm.wav");

}


void tt(string path) {
//输入WAV文件地址
    unsigned long wavdata_l;
    double *sample = NULL;

    sample = readwav(path, &wavdata_l);

    int frameNum = 1 + ceil((double(wavdata_l) - Win_Time * sampleRate) / hopStep);//计算所有样本点一共有多少帧
    int frame_new = (ceil(frameNum / (maxlen / 2 * 1.0)) - 1) * maxlen;
    double **mfcc39 = new double *[frameNum];
    for (int i = 0; i < frameNum; i++) {
        mfcc39[i] = new double[39];
        for (int j = 0; j < 39; j++)
            mfcc39[i][j] = 0;
    }
//9ms
    clock_t startTime = clock();
    MFCC(sample, wavdata_l, mfcc39, frameNum);
    clock_t endTime = clock();
//741ms

    delete[]sample;
    int overlap = maxlen / 2;
    int num_100 = ceil(frameNum / (overlap * 1.0)) - 1;
    int test_num = num_100 * maxlen;
    double **xtest = new double *[test_num];
    for (int i = 0; i < test_num; i++) {
        xtest[i] = new double[39];
        for (int j = 0; j < 39; j++)
            xtest[i][j] = 0;
    }
    reshape(mfcc39, maxlen, dim, frameNum, xtest);
//释放
    for (int i = 0; i < frameNum; i++) {
        delete[] mfcc39[i];
    }
    delete[] mfcc39;


//==========================================
    int *label = new int[frameNum];

//前向计算部分
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    label = model(xtest, frameNum, frame_new, label, G_W_Z, G_U_Z, G_W_R, G_U_R, G_W_H, G_U_H,
                  G_B_Z,
                  G_B_R, G_B_H, F_W_Z, F_U_Z, F_W_R, F_U_R, F_W_H, F_U_H, F_B_Z, F_B_R, F_B_H,
                  B_W_Z,
                  B_U_Z, B_W_R, B_U_R, B_W_H, B_U_H, B_B_Z, B_B_R, B_B_H, gru_kernel, gru_r_kernel,
                  gru_bias,
                  fw_gru_kernel, fw_gru_r_kernel, fw_gru_bias, bw_gru_kernel, bw_gru_r_kernel,
                  bw_gru_bias
    );
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//释放
    for (int i = 0; i < test_num; i++) {
        delete[] xtest[i];
    }
    delete[] xtest;


//=================================
    cout << wavdata_l << endl;
//	cout<<xtest[0][0];
    vector<int> newlist;//输出的序列
    vector<int> pointlist;//记录每个音素从第几个采样点开始
    trimming(newlist, pointlist, label, frameNum, hopStep);
//10611ms

    delete[]label;
    int phone_num = newlist.size();
//把输出写到txt文件中
    ofstream fileout(Constant::ASR_BASE_PATH + "/output.txt");
    for (int i = 0; i < phone_num; i++) {
        fileout << newlist[i] << ',';
        fileout << pointlist[i] << endl;
    }
    fileout.close();

    newlist.clear();
    pointlist.clear();
//10919ms
}


