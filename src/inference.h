/*
 * digit_touch — C interface of the inference engine (TFLM, implemented in C++).
 */

#ifndef INFERENCE_H_
#define INFERENCE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_IMAGE_SIZE  (28 * 28)
#define INFERENCE_NUM_CLASSES 10

/* int8 <-> real conversion rule: real = scale * (q - zero_point) */
struct inference_quant {
	float in_scale;
	int in_zero_point;
	float out_scale;
	int out_zero_point;
};

/* Load the model and reserve its memory in the arena. 0 on success. */
int inference_init(void);

/* image: 28x28 int8 (-128 = black, 127 = white)
 * scores: 10 int8 outputs (quantized probabilities). 0 on success.
 */
int inference_run(const int8_t *image, int8_t *scores);

/* Arena bytes actually used (known after inference_init) */
size_t inference_arena_used(void);
size_t inference_arena_size(void);

const struct inference_quant *inference_get_quant(void);

/* Time spent in one operation during the last inference */
struct inference_layer_time {
	const char *op;   /* TFLite name: "CONV_2D", "MAX_POOL_2D"... */
	uint32_t cycles;
};

/* Copy the profile of the last inference (CONFIG_APP_LAYER_PROFILING).
 * Returns the number of operations, 0 if profiling is disabled.
 */
int inference_last_profile(struct inference_layer_time *out, int max);

/* Arena address, to know which memory it lives in */
const void *inference_arena_addr(void);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H_ */
