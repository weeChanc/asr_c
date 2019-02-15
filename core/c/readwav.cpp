
#include <iostream>
#include <fstream>
#include <string.h>
#include<math.h>
#include<cmath>
#include<stdlib.h>
#include <bitset>
#include <iomanip>
//要在intmain()的前面加上函数的声明，因为你的函数写在main函数的后面
int hex_char_value(char ss);
int hex_to_decimal(const char* s);
//string hex to  binary(char* szHex);
using namespace std;

struct wav_struct
{
    unsigned long file_size;
    unsigned short channel;
    unsigned long frequency;
    unsigned long Bps;
    unsigned short sample_num_bit;
    unsigned long data_size;
    unsigned char *data;
};

double * readwav(string path,long unsigned *data_len)
{
    fstream fs;
    wav_struct WAV;
    fs.open(path, ios::binary | ios::in);
    fs.seekg(0, ios::end);
    WAV. file_size = fs.tellg();
    fs.seekg(0x14);
    fs.read((char*)&WAV.channel, sizeof(WAV.channel));
    fs.seekg(0x18);
    fs.read((char*)&WAV.frequency, sizeof(WAV.frequency));
    fs.seekg(0x1c);
    fs.read((char*)&WAV.Bps, sizeof(WAV.Bps));
    fs.seekg(0x22);
    fs.read((char*)&WAV.sample_num_bit, sizeof(WAV.sample_num_bit));
    fs.seekg(0x28);
    fs.read((char*)&WAV.data_size, sizeof(WAV.data_size));
    WAV. data = new unsigned char [WAV.data_size];
    fs.seekg(0x2c);
    fs.read((char *)WAV.data,sizeof(char)*WAV.data_size);
	 double* wavdata=( double *)malloc(WAV.data_size*sizeof(double));

    for (unsigned long i =0; i<WAV.data_size; i = i + 2)
    {
        //右边为大端
        unsigned long data_low = WAV.data[i];
        unsigned long data_high = WAV.data[i + 1];
        double data_true = data_high * 256 + data_low;
        //printf("%d ",data_true);
        long data_complement = 0;
        //取大端的最高位（符号位）
        int my_sign = (int)(data_high / 128);
        //printf("%d ", my_sign);
        if (my_sign == 1)
        {
            data_complement = data_true - 65536;
        }
        else
        {
            data_complement = data_true;
        }
        //printf("%d ", data_complement);
        setprecision(4);
        long double float_data = (double)(data_complement/(double)32768);

        wavdata[i/2]=float_data;
        //printf("%f ",wavdata[i/2]);

    }
    fs.close();

    delete[] WAV.data;
	*data_len=WAV.data_size/2;
	return wavdata;
}

int hex_char_value(char c)
{
    if (c>='0'&&c<='9')
        return c-'0';
    else if (c>='a'&&c<='f')
        return (c-'a'+10);
    else if (c>='A'&&c<='F')
        return(c-'A'+10);
    return 0;
}
int hex_to_decimal(char* szHex)
{
    int len =2;
    int result=0;
    for (int i=0;i<len;i++)
    {
         result += (int)pow((float)16, (int)len - i - 1) * hex_char_value(szHex[i]);
    }
    return result;
}


