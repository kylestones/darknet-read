#include "im2col.h"
#include <stdio.h>


// 根据 image 的索引得到其 pixel value
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    // 实际卷积的时候会有 padding ，而填充的值并不会存储在 im 中，
    // 将 row 和 col 减去 pad 便可以得到在 im 中的索引
    row -= pad;
    col -= pad;

    // 此时对应的是 padding 的值
    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;

    // im 是一个一维数组，
    // 将所有通道的二维图像按顺序依次拼接成一个一维数组
    // 每个通道内部逐行拼接
    // im = [R0 R1 R2 R3 G0 G1 G2 G3 B0 B1 B2 B3]
    //
    // channel*height*width 找到指定 channel 的开头
    // row*width + col 为在指定 channel 内，根据 (row, col) 得到指定的 pixel
    // 作者这里的索引是从小到大来计算的 col row channel
    return im[col + width*(row + height*channel)];
}

// Optimizing Conv in Caffe : Converting convolution to GEMM
// 需要两部来完成
// 1. 使用 im2col convert the image to a matrix
// 2. 调用 GEMM 完成实际的矩阵乘法运行
// 优点：逻辑简单，易于实现
// 缺点：消耗大量内存，并且没有很好利用 flops/parm advantage TODO
//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
//
// 由于卷积操作需要多次重复使用 image 中的 pixel ,不方便并行处理；
// 这里希望 *将卷积操作转换成矩阵乘法*
// matrix 是一个 (channels*ksize*ksize) x (height_col*width_col)的二维数组【行数 x 列数】
//
// 一个卷积核的大小是 channels*ksize*ksize
// 在一次卷积中，将 image 中与卷积核进行相乘的所有元素展开拼接成一列
// 输出一个 feature map 共需要 [(height + 2*pad - ksize) / stride + 1]^2 次卷积
// 所以共需要上述多次卷积生成一个 feature map
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;

    // height_col width_col 是经过卷积之后 feature map 的大小 spatial 
    // 将 image 转换成 matrix 的行数就是 height_col*width_col
    // 输出一个 feature map 共需要这么多次卷积
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    // 在一次卷积中，将 image 中与卷积核进行相乘的所有元素展开成一列
    // 而一个卷积核的大小就是 channels*ksize*ksize ，所以一列元素的个数为 channels*ksize*ksize
    // 将 image 展开成 matrix ，行数为卷积核大小的平方乘以通道数 channels*ksize*ksize
    int channels_col = channels * ksize * ksize;


    // 下面三个循环一起完成对 data_col 的填充：索引 col_index 从 0 开始逐渐增大到最大
    // 将 col_index 转换到 image 对应的索引上，获取响应的 pixel value
    
    // 对 data_col 逐行遍历
    for (c = 0; c < channels_col; ++c) {

        // 一个卷积核一个 channel 上的水平和垂直坐标
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;

        // channel
        int c_im = c / ksize / ksize;

        // 下面两层循环一起完成对 data_col 按列遍历
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {

                // 将数据在 data_col 中的索引转换到 im 中对应的索引
                // h 和 w 与 stride 相乘表示垂直和水平方向将卷积核移动长度
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;

                // col_index 是生成的 matrix 的索引
                // 这里的 matrix 并没有使用二维数组来存储，仍然使用了一维数组来保存
                // 不过依然可以认为 matrix 是一个 (channels*ksize*ksize) x (height_col*width_col)的二维数组【行数 x 列数】
                int col_index = (c * height_col + h) * width_col + w;

                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}


// 理解参考 
// https://www.zhihu.com/question/28385679
// https://blog.csdn.net/qq_29381089/article/details/80290111
