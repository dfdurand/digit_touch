/*
 * digit_touch — preprocessing: 252x252 drawing -> MNIST-style 28x28 image.
 *
 * The model has only seen MNIST digits, all prepared the same way:
 *   1. the digit is scaled down to fit in a 20x20 pixel box, keeping its
 *      aspect ratio, with anti-aliasing (grayscale edges);
 *   2. this box is placed in a 28x28 image so that the CENTER OF MASS
 *      (center of gravity of the ink) falls at the center of the image.
 * We reproduce exactly these two operations on the finger drawing.
 */

#include "preprocess.h"

#include <math.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

/* Value of a drawing pixel, 0 (black) outside the area */
static inline uint32_t ink_at(const uint8_t *ink, int x, int y)
{
	if (x < 0 || y < 0 || x >= DRAW_SIZE || y >= DRAW_SIZE) {
		return 0;
	}
	return ink[y * DRAW_SIZE + x];
}

/* Step 1: bounding box -> 20x20 box.
 *
 * We take a SQUARE whose side = the longest side of the bounding box,
 * centered on it: a "1" (tall and narrow) stays narrow instead of being
 * stretched horizontally. This square is split into 20x20 cells; each output
 * pixel is the AVERAGE of the pixels in its cell ("box" filter). A 17 px
 * stroke scaled down by a factor of ~7 gives ~2-3 px with gray edges: that
 * is MNIST's anti-aliasing.
 */
static void fit_box(const uint8_t *ink, const struct ink_bbox *bb,
		    uint8_t box[MNIST_BOX * MNIST_BOX])
{
	int w = bb->x_max - bb->x_min + 1;
	int h = bb->y_max - bb->y_min + 1;
	int side = MAX(w, h);
	/* Top-left corner of the square centered on the bounding box */
	int x0 = bb->x_min + (w - side) / 2;
	int y0 = bb->y_min + (h - side) / 2;

	for (int oy = 0; oy < MNIST_BOX; oy++) {
		int sy0 = y0 + oy * side / MNIST_BOX;
		int sy1 = MAX(y0 + (oy + 1) * side / MNIST_BOX, sy0 + 1);

		for (int ox = 0; ox < MNIST_BOX; ox++) {
			int sx0 = x0 + ox * side / MNIST_BOX;
			int sx1 = MAX(x0 + (ox + 1) * side / MNIST_BOX, sx0 + 1);
			uint32_t sum = 0;

			for (int y = sy0; y < sy1; y++) {
				for (int x = sx0; x < sx1; x++) {
					sum += ink_at(ink, x, y);
				}
			}
			box[oy * MNIST_BOX + ox] = sum / ((sx1 - sx0) * (sy1 - sy0));
		}
	}
}

/* Step 2: place the 20x20 box in the 28x28 image, centered on its center
 * of mass.
 *
 * Center of mass = average of the pixel positions, weighted by their
 * intensity. Example: a "7" has a lot of ink at the top (the bar), so its
 * center of mass is higher than the center of its box. Bringing it back to
 * the center moves the 7 down a little, exactly as in MNIST.
 */
static void center_of_mass(const uint8_t box[MNIST_BOX * MNIST_BOX],
			   uint8_t out[MNIST_SIZE * MNIST_SIZE])
{
	uint32_t total = 0, sx = 0, sy = 0;
	int dx, dy;

	for (int y = 0; y < MNIST_BOX; y++) {
		for (int x = 0; x < MNIST_BOX; x++) {
			uint32_t v = box[y * MNIST_BOX + x];

			total += v;
			sx += v * x;
			sy += v * y;
		}
	}

	if (total == 0) {
		dx = dy = (MNIST_SIZE - MNIST_BOX) / 2;
	} else {
		/* Offset that brings the center of mass to the center (14, 14) */
		dx = (int)lroundf(MNIST_SIZE / 2.0f - (float)sx / total);
		dy = (int)lroundf(MNIST_SIZE / 2.0f - (float)sy / total);
		/* Keep the box entirely inside the image */
		dx = CLAMP(dx, 0, MNIST_SIZE - MNIST_BOX);
		dy = CLAMP(dy, 0, MNIST_SIZE - MNIST_BOX);
	}

	memset(out, 0, MNIST_SIZE * MNIST_SIZE);
	for (int y = 0; y < MNIST_BOX; y++) {
		memcpy(&out[(y + dy) * MNIST_SIZE + dx], &box[y * MNIST_BOX], MNIST_BOX);
	}
}

void preprocess_drawing(const uint8_t *ink, const struct ink_bbox *bbox,
			uint8_t out[MNIST_SIZE * MNIST_SIZE])
{
	uint8_t box[MNIST_BOX * MNIST_BOX];

	fit_box(ink, bbox, box);
	center_of_mass(box, out);
}

void preprocess_quantize(const uint8_t in[MNIST_SIZE * MNIST_SIZE],
			 int8_t out[MNIST_SIZE * MNIST_SIZE], const struct inference_quant *q)
{
	/* During training: real pixel = byte / 255.
	 * int8 conversion: q = round(real / scale) + zero_point.
	 * With scale = 1/255 and zero_point = -128, this amounts to byte - 128.
	 */
	for (int i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
		int v = (int)lroundf(in[i] / 255.0f / q->in_scale) + q->in_zero_point;

		out[i] = (int8_t)CLAMP(v, -128, 127);
	}
}

/* With CONFIG_LOG_PRINTK, each printk() becomes a log message: so each whole
 * line is built before being sent in a single call (one printk per
 * character overflowed the log buffer).
 */
void preprocess_dump(const uint8_t img[MNIST_SIZE * MNIST_SIZE], int pred)
{
	static const char shades[] = " .:-=+*#%@";
	static const char hex[] = "0123456789abcdef";
	static char line[2 * MNIST_SIZE * MNIST_SIZE + 1];

	printk("+----------------------------------------------------------+\n");
	for (int y = 0; y < MNIST_SIZE; y++) {
		for (int x = 0; x < MNIST_SIZE; x++) {
			char c = shades[img[y * MNIST_SIZE + x] * (sizeof(shades) - 2) / 255];

			/* 2 characters per pixel: the image is not squashed horizontally */
			line[2 * x] = c;
			line[2 * x + 1] = c;
		}
		line[2 * MNIST_SIZE] = '\0';
		printk("| %s |\n", line);
	}
	printk("+----------------------------------------------------------+\n");

	for (int i = 0; i < MNIST_SIZE * MNIST_SIZE; i++) {
		line[2 * i] = hex[img[i] >> 4];
		line[2 * i + 1] = hex[img[i] & 0xf];
	}
	line[2 * MNIST_SIZE * MNIST_SIZE] = '\0';
	printk("IMG:%d:%s\n", pred, line);
}
