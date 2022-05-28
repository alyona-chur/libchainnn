// layer.h - Synapse, base layer, activation logic.
//

#ifndef LAYER_H_
#define LAYER_H_

#include "common.h"

#include <math.h>
#include <stdlib.h>

// Synapse
//

typedef struct {
	// Position of the synapse start and end neuron in the layer
	int in_pos;
	int out_pos;

    // Position of the synapse weight and bias
    // in layer's weights and biases array
    // (set to NULL for not weighted layers)
	ll w_pos;
	ll b_pos;
} SYNAPSE;

// Layer
//

typedef enum {
	MAX_POOLING = 0,  // Weighted layers have positive layer types
	DENSE = 1,
	CONV = 2
} LAYER_TYPE;

struct Layer {
	LAYER_TYPE type;

	// Synapses for each neuron
	SYNAPSE* synapses;
	int syn_len;

	// Input and output indexing
	int in_w;
	int in_h;
	int in_d;
	int in_len;

	int out_w;
	int out_h;
	int out_d;
	int out_len;

	// Methods
	ERROR_CODE (*pass_forwards)(const LAYER*, ll*, ll*);

    ERROR_CODE (*prepare_for_training)(LAYER*);
    ERROR_CODE (*free_after_training)(LAYER*);
    ERROR_CODE (*pass_backwards)(LAYER*, ll*, ll*);
    ERROR_CODE (*update)(LAYER*, ll*, ll*);

	ERROR_CODE (*print)(const LAYER*);
	ERROR_CODE (*write)(const LAYER*);
	ERROR_CODE (*read)(const LAYER*);

    void (*free)(LAYER*);
};

void free_layer_after_training(LAYER* layer);
void free_layer(LAYER* layer);
void print_layer(LAYER* layer);
// ERROR_CODE read_layer(const LAYER* layer, int path_len, const char* filepath);
// ERROR_CODE write_layer(const LAYER* layer, int path_len, const char* filepath);

// Activation
//

typedef enum {
	SIGMOID = 0
} ACTIVATION_TYPE;

ll sigmoid(ll x);
ll sigmoid_derivative(ll x);

#endif  // LAYER_H_
