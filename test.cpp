#include "layers.h"
#include<stdio.h>
#include <sys/time.h>
#include "utils.h"

int main(int argc, char **argv) {

    Image<float> data(3, 3, 1, 2);
    Image<float> data_1(3, 3, 1, 2);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            data(i, j, 0, 0) = i + 1;
            data(i, j, 0, 1) = (i + 1) * 0.1;
            data_1(i, j, 0, 0) = (i + 1) * 1;
            data_1(i, j, 0, 1) = (i + 1) * 10;
        }
    }

    Image<float> dx(3, 3, 1, 2);
    DataLayer * d_layer = new DataLayer(3, 3, 1, 2, data);
    BatchNorm * bat  = new BatchNorm(d_layer);
    Var x, y, z, w;
    Func back;
    back(x, y, z, w) = data_1(x, y, z, w);
    bat->back_propagate(back);

    init_constant(bat->params[1], 0.0);
    init_constant(bat->params[0], 1.0);

    std::vector<Func> train_outs;
    train_outs.push_back(bat->f_param_grads[0]);
    train_outs.push_back(bat->f_param_grads[1]);
    train_outs.push_back(bat->f_in_grad);
    Pipeline train(train_outs);

    back.realize(data_1);
    //d_layer->forward.realize(data);
    //bat->forward.compute_root();
    //bat->forward.trace_stores();
    //bat->forward.realize(data);

    //bat->back_propagate.realize();
    train.realize({bat->param_grads[0], bat->param_grads[1], data});

}




