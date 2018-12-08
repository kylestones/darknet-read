#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer make_dropout_layer(int batch, int inputs, float probability)
{
    dropout_layer l = {0};
    l.type = DROPOUT;

    // 阈值
    l.probability = probability;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;

    // 输入的每个 pixel 都对应一个随机值
    l.rand = calloc(inputs*batch, sizeof(float));

    // inverted dropout
    l.scale = 1./(1.-probability);

    l.forward = forward_dropout_layer;
    l.backward = backward_dropout_layer;
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void resize_dropout_layer(dropout_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
}

void forward_dropout_layer(dropout_layer l, network net)
{
    int i;

    // Inference 不用 dropout
    if (!net.train) return;

    for(i = 0; i < l.batch * l.inputs; ++i){
        // 这个概率每次都会随机生成，此处为什么要保存该随机值呢？ 反向传播的时候需要使用
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        if(r < l.probability) net.input[i] = 0;
        else net.input[i] *= l.scale;
    }
}

void backward_dropout_layer(dropout_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch * l.inputs; ++i){
        float r = l.rand[i];
        // 被 dropout 的节点不产生误差
        if(r < l.probability) net.delta[i] = 0;
        else net.delta[i] *= l.scale;
    }
}

