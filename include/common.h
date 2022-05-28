// common.h - Common utilities.
//
// All data is stored in 1D arrays.
//

#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>

#define long long ll

// Indexing
//

#define DIM2_TO_DIM1(i, j, w) (i * w + j)
#define DIM3_TO_DIM1(i, j, k, w, d) (i * w * d + j * d + k)
#define LEN_DIM2(h, w) (h * w)
#define LEN_DIM3(h, w, d) (h * w * d)
#define FILL_ARRAY(array, val, len) for (int a = 0; a < len; ++a) array[a] = val;

// Error codes
//

typedef enum {
	LOSS_TYPE_NOT_EXIST = -1,
	SUCCESS = 0,
	ACTIVATION_NOT_EXIST = 1,
	LAYER_TYPE_NOT_EXIST = 2,
	NOT_IMPLEMENTED = 3,
	WRONG_LAYER_TYPE = 4,
	SIZE_ERROR = 5
} ERROR_CODE;

#endif // COMMON_H_
