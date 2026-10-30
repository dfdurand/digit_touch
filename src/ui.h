/*
 * digit_touch — LVGL UI: drawing area + Clear button.
 */

#ifndef UI_H_
#define UI_H_

#include <stdbool.h>
#include <stdint.h>

#include "inference.h"

/* Square drawing area: 252 = 9 x 28, 8-bit grayscale
 * (0 = black, 255 = white), stored row by row.
 */
#define DRAW_SIZE 252

/* Bounding box of the ink (inclusive bounds) */
struct ink_bbox {
	int16_t x_min, y_min, x_max, y_max;
};

/* Delay without a new stroke after which the drawing is considered finished.
 * Allows drawing a 4 or a crossed 7 in several strokes.
 */
#define DRAWING_DONE_DELAY_MS 500

/* If a result has been displayed for at least this delay, touching the
 * screen starts a new digit: the drawing area is cleared first.
 * Below it, the stroke is added to the current drawing (e.g. the bar of a 7
 * added right after the result).
 */
#define AUTO_CLEAR_DELAY_MS 1000

/* Called (in the LVGL thread) when the drawing is finished */
typedef void (*ui_drawing_done_cb_t)(void);

/* Create the screen. Call once, from the thread that runs LVGL. */
void ui_init(ui_drawing_done_cb_t done_cb);

/* Display the result of a recognition.
 * permille: probability of each digit, in thousandths (0..1000)
 * best, second: the two most likely digits
 * sure: false if the confidence is too low -> displays "?"
 */
void ui_show_result(const int permille[INFERENCE_NUM_CLASSES], int best, int second, bool sure);

/* Drawn image: DRAW_SIZE x DRAW_SIZE bytes */
const uint8_t *ui_drawing(void);

/* false if nothing is drawn; otherwise fills *bbox */
bool ui_drawing_bbox(struct ink_bbox *bbox);

#endif /* UI_H_ */
