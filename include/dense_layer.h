// dense_layer.h - Dense layer logic.
//
// Dense layer is treated as a convolution layer with kernel shape == input shape.
//

#ifndef DENSE_LAYER_H_
#define DENSE_LAYER_H_

#include "common.h"
#include "conv_layer.h"
#include "layer.h"

#include <stdlib.h>

ERROR_CODE init_dense_layer(CONV_LAYER* layer, int in_h, int in_w, int in_d,
                            int out_h, int out_w, int out_d,
                            ACTIVATION_TYPE activation_type);

#endif // DENSE_LAYER_H_
