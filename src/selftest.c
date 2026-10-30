/*
 * digit_touch — model self-test on fixed MNIST images.
 *
 * For each image in test_images.c:
 *  - run inference on the board;
 *  - compare the 10 int8 scores with those computed on the PC (they must be
 *    byte-for-byte identical: same model, same inputs, same integer
 *    arithmetic);
 * then measure the average time of one inference.
 */

#include "selftest.h"
#include "inference.h"
#include "test_images.h"

#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(selftest, LOG_LEVEL_INF);

#define BENCH_RUNS 100

static int argmax(const int8_t *scores)
{
	int best = 0;

	for (int i = 1; i < INFERENCE_NUM_CLASSES; i++) {
		if (scores[i] > scores[best]) {
			best = i;
		}
	}
	return best;
}

/* Probability in tenths of a %: real = (q - zero_point) * scale, here scale = 1/256 */
static int score_permille(int8_t q, const struct inference_quant *qp)
{
	return (int)((q - qp->out_zero_point) * qp->out_scale * 1000.0f + 0.5f);
}

int model_selftest(void)
{
	const struct inference_quant *qp = inference_get_quant();
	int8_t scores[INFERENCE_NUM_CLASSES];
	int differences = 0;
	uint32_t start, cycles;

	LOG_INF("--- self-test: %d MNIST images ---", TEST_IMAGE_COUNT);

	for (int i = 0; i < TEST_IMAGE_COUNT; i++) {
		if (inference_run(test_images[i], scores) != 0) {
			LOG_ERR("image %d: inference failed", i);
			differences++;
			continue;
		}

		int pred = argmax(scores);
		int pm = score_permille(scores[pred], qp);
		bool same = memcmp(scores, test_expected[i], sizeof(scores)) == 0;

		if (!same) {
			differences++;
		}

		LOG_INF("image %2d: true %u, predicted %d (%3d.%d %%)  %s%s", i, test_labels[i], pred,
			pm / 10, pm % 10, same ? "== PC" : "!= PC",
			pred == test_labels[i] ? "" : "  [model error]");
	}

	/* Timing: average over BENCH_RUNS inferences.
	 * k_cycle_get_32() counts CPU cycles (216 MHz on the F746).
	 */
	start = k_cycle_get_32();
	for (int r = 0; r < BENCH_RUNS; r++) {
		inference_run(test_images[0], scores);
	}
	cycles = (k_cycle_get_32() - start) / BENCH_RUNS;

	LOG_INF("config: %s, CMSIS-NN %s, arena in %s (%p)",
		IS_ENABLED(CONFIG_SPEED_OPTIMIZATIONS) ? "-O2" : "-Os",
		IS_ENABLED(CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN_KERNELS) ? "yes" : "no",
		IS_ENABLED(CONFIG_APP_ARENA_IN_DTCM) ? "DTCM" : "SRAM", inference_arena_addr());
	LOG_INF("inference time: %u cycles = %u us (average over %d)", cycles,
		(uint32_t)k_cyc_to_us_near32(cycles), BENCH_RUNS);
	LOG_INF("arena: %u / %u bytes used", (unsigned)inference_arena_used(),
		(unsigned)inference_arena_size());

	/* Profile of the last benchmark inference: where does the time go? */
	struct inference_layer_time layers[16];
	int n = inference_last_profile(layers, ARRAY_SIZE(layers));
	uint32_t total = 0;

	for (int i = 0; i < n; i++) {
		total += layers[i].cycles;
	}
	for (int i = 0; i < n; i++) {
		LOG_INF("  layer %d %-16s %8u cycles %6u us %3u %%", i, layers[i].op,
			layers[i].cycles, (uint32_t)k_cyc_to_us_near32(layers[i].cycles),
			total ? (uint32_t)((uint64_t)layers[i].cycles * 100 / total) : 0);
	}
	LOG_INF("--- result: %d/%d outputs identical to PC ---",
		TEST_IMAGE_COUNT - differences, TEST_IMAGE_COUNT);

	return differences;
}
