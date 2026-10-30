/*
 * digit_touch — recognition of handwritten digits drawn with a finger.
 *
 * At startup: model self-test on fixed MNIST images (step 5), then the
 * drawing UI (step 6). When the drawing is finished: 28x28 preprocessing
 * (step 7), inference, then display of the result and the confidence bars
 * (step 8). The 28x28 image is also sent to the console (debug, and data
 * collection for step 9).
 */

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/display.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <lvgl.h>

#include "inference.h"
#include "preprocess.h"
#include "selftest.h"
#include "ui.h"

LOG_MODULE_REGISTER(app, LOG_LEVEL_INF);

/* Below 75 % for the most likely digit, "?" is displayed.
 * The model has no "this is not a digit" class: it always answers one of
 * the 10. A low confidence signals an ambiguous drawing.
 * Tuned from testing: 60 % let through a "∠" read as 4 at 73 % and a
 * French-style "1" read as 1 at 64 % (vs 7 at 34 %). The correct answers
 * were all above 93 %.
 */
#define CONFIDENCE_THRESHOLD_PERMILLE 750

static bool model_ready;

/* Called by the UI when the drawing is finished */
static void on_drawing_done(void)
{
	static uint8_t img[MNIST_SIZE * MNIST_SIZE];
	static int8_t input[MNIST_SIZE * MNIST_SIZE];
	const struct inference_quant *q = inference_get_quant();
	int8_t scores[INFERENCE_NUM_CLASSES];
	struct ink_bbox bbox;
	uint32_t t0, t1, t2;
	int order[INFERENCE_NUM_CLASSES];
	int permille[INFERENCE_NUM_CLASSES];
	bool sure;

	if (!model_ready || !ui_drawing_bbox(&bbox)) {
		return;
	}

	t0 = k_cycle_get_32();
	preprocess_drawing(ui_drawing(), &bbox, img);
	preprocess_quantize(img, input, q);
	t1 = k_cycle_get_32();
	if (inference_run(input, scores) != 0) {
		LOG_ERR("inference failed");
		return;
	}
	t2 = k_cycle_get_32();

	/* Classes sorted by decreasing score (insertion sort, 10 elements) */
	for (int i = 0; i < INFERENCE_NUM_CLASSES; i++) {
		int j = i;

		while (j > 0 && scores[order[j - 1]] < scores[i]) {
			order[j] = order[j - 1];
			j--;
		}
		order[j] = i;
	}

	/* Probabilities in thousandths: real = scale * (q - zero_point) */
	for (int i = 0; i < INFERENCE_NUM_CLASSES; i++) {
		permille[i] = (int)((scores[i] - q->out_zero_point) * q->out_scale * 1000.0f + 0.5f);
	}
	sure = permille[order[0]] >= CONFIDENCE_THRESHOLD_PERMILLE;

	ui_show_result(permille, order[0], order[1], sure);
	preprocess_dump(img, order[0]);

	LOG_INF("prediction: %d (%d.%d %%)   then %d (%d.%d %%), %d (%d.%d %%)%s", order[0],
		permille[order[0]] / 10, permille[order[0]] % 10, order[1],
		permille[order[1]] / 10, permille[order[1]] % 10, order[2],
		permille[order[2]] / 10, permille[order[2]] % 10, sure ? "" : "  -> shown \"?\"");
	LOG_INF("ink %dx%d px | preprocessing %u us, inference %u us",
		bbox.x_max - bbox.x_min + 1, bbox.y_max - bbox.y_min + 1,
		k_cyc_to_us_near32(t1 - t0), k_cyc_to_us_near32(t2 - t1));
}

int main(void)
{
	const struct device *display_dev = DEVICE_DT_GET(DT_CHOSEN(zephyr_display));

	if (inference_init() == 0) {
		model_selftest();
		model_ready = true;
	} else {
		LOG_ERR("model initialization failed");
	}

	if (!device_is_ready(display_dev)) {
		LOG_ERR("display not ready");
		return 0;
	}

	ui_init(on_drawing_done);
	lv_timer_handler();
	display_blanking_off(display_dev);

	while (1) {
		uint32_t sleep_ms = lv_timer_handler();

		k_msleep(MIN(sleep_ms, 10));
	}

	return 0;
}
