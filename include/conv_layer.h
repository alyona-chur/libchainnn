// conv_layer.h - Convolution layer logic.
//
// Gradients are updated on backward pass and applied on update.
//

#ifndef CONV_LAYER_H_
#define CONV_LAYER_H_

#include "common.h"
#include "layer.h"

#include <stdlib.h>

// Convolution layer
//

typedef struct {
	LAYER as_layer;

	// Kernel
	ll* weights;
	int w_len;

	int ker_w;
	int ker_h;
	int ker_d;
	int ker_len;

	// Activation
	ll (*activation)(ll);
    ll (*activation_derivative)(ll);

	// Bias value for each neuron
	ll* biases;
	int b_len; // == out_len;

	// Data only used on training
	// Gradients are computed over mini-batch
    // and then applied to weights and biases
	ll* sum_res; // Weighted sum results

	ll* w_grad_sum;
	ll* w_prev_delta;
	ll* b_cur_grad_sum;
	ll* b_prev_delta;
} CONV_LAYER;

static inline ERROR_CODE init_layer_activation(CONV_LAYER* layer, ACTIVATION_TYPE activation_type) {
	if (activation_type == SIGMOID) {
		layer->activation = &sigmoid;
		layer->activation_derivative = &sigmoid_derivative;
	}
	else {
		return ACTIVATION_NOT_EXIST;
	}
	return SUCCESS;
}

ERROR_CODE init_conv_layer(CONV_LAYER* layer, int in_h, int in_w, int in_d,
                           int ker_cnt, int ker_size, int stride,
                           ACTIVATION_TYPE activation_type);

ERROR_CODE pass_forwards_conv_layer(const CONV_LAYER* layer,
                                    const ll* input, ll* output);

void prepare_conv_layer_for_training(CONV_LAYER* layer);
ERROR_CODE pass_backwards_conv_layer(CONV_LAYER* layer,
                                     const ll* input_deltas, ll* output_deltas);
ERROR_CODE update_conv_layer(CONV_LAYER* layer, ll lr, ll moment);
void free_conv_layer_after_training(CONV_LAYER* layer);
void free_conv_layer(CONV_LAYER* layer);

void print_conv_layer(CONV_LAYER* layer);

// Testing
void set_layer_weights(CONV_LAYER* layer, ll* weights);
void set_layer_biases(CONV_LAYER* layer, ll* biases);

// Convolution
//

static void convolve(const ll* input, const ll* weights,
                     const ll* biases, const SYNAPSE* synapses,
                     ll* output, int ker_len, int out_pos);

#endif // CONV_LAYER_H_
