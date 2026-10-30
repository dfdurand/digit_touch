/*
 * digit_touch — preprocessing: 252x252 drawing -> MNIST-style 28x28 image.
 */

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include <stdint.h>

#include "inference.h"
#include "ui.h"

#define MNIST_SIZE 28
#define MNIST_BOX  20 /* MNIST: the digit fits in a 20x20 box */

/* Crop, scale down and center the drawing like the MNIST digits.
 * ink  : DRAW_SIZE x DRAW_SIZE bytes (0 = black, 255 = white)
 * bbox : bounding box of the ink
 * out  : 28x28 grayscale image 0..255
 */
void preprocess_drawing(const uint8_t *ink, const struct ink_bbox *bbox,
			uint8_t out[MNIST_SIZE * MNIST_SIZE]);

/* 0..255 -> int8 according to the model's input conversion rule */
void preprocess_quantize(const uint8_t in[MNIST_SIZE * MNIST_SIZE],
			 int8_t out[MNIST_SIZE * MNIST_SIZE], const struct inference_quant *q);

/* Debug: print the 28x28 image as ASCII on the console, then a line
 * "IMG:<784 bytes in hex>" readable by train/view_digits.py.
 */
void preprocess_dump(const uint8_t img[MNIST_SIZE * MNIST_SIZE], int pred);

#endif /* PREPROCESS_H_ */
