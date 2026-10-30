/*
 * digit_touch — TensorFlow Lite Micro inference engine.
 *
 * TFLM uses no dynamic allocation: all computed tensors (activations) are
 * placed in a static array, the "tensor arena".
 * The weights stay in flash, in g_model_data.
 */

#include "inference.h"
#include "model_data.h"

#include <errno.h>
#include <string.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_profiler_interface.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <zephyr/kernel.h>
#include <zephyr/linker/section_tags.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(inference, LOG_LEVEL_INF);

#if defined(CONFIG_APP_ARENA_IN_DTCM)
/* DTCM: core-coupled RAM (0x20000000), no wait states, no cache */
#define ARENA_SECTION __dtcm_noinit_section
#else
#define ARENA_SECTION
#endif

namespace {

/* Measured on the board: 9924 B used with CMSIS-NN, 9812 B without.
 * 12 KB leaves ~20 % headroom.
 */
constexpr size_t kArenaSize = 12 * 1024;
uint8_t tensor_arena[kArenaSize] __attribute__((aligned(16))) ARENA_SECTION;

#if defined(CONFIG_APP_LAYER_PROFILING)
/* TFLM calls BeginEvent()/EndEvent() around each operation of the model.
 * We simply measure the elapsed CPU cycles there.
 */
class CycleProfiler : public tflite::MicroProfilerInterface {
public:
	static constexpr int kMaxEvents = 16;

	uint32_t BeginEvent(const char *tag) override
	{
		if (count_ >= kMaxEvents) {
			return kMaxEvents;
		}
		events_[count_].op = tag;
		start_[count_] = k_cycle_get_32();
		return count_++;
	}

	void EndEvent(uint32_t handle) override
	{
		if (handle < kMaxEvents) {
			events_[handle].cycles = k_cycle_get_32() - start_[handle];
		}
	}

	void Reset() { count_ = 0; }

	int Copy(struct inference_layer_time *out, int max) const
	{
		int n = MIN(count_, max);

		memcpy(out, events_, n * sizeof(events_[0]));
		return n;
	}

private:
	struct inference_layer_time events_[kMaxEvents];
	uint32_t start_[kMaxEvents];
	int count_ = 0;
};

CycleProfiler profiler;
#define PROFILER (&profiler)
#else
#define PROFILER nullptr
#endif

/* Only the 5 operations of the model (see train/out/quant.json):
 * the linker does not pull in the code of the others.
 */
using OpResolver = tflite::MicroMutableOpResolver<5>;

tflite::MicroInterpreter *interpreter;
TfLiteTensor *input;
TfLiteTensor *output;
struct inference_quant quant;

} /* namespace */

extern "C" int inference_init(void)
{
	const tflite::Model *model = tflite::GetModel(g_model_data);

	if (model->version() != TFLITE_SCHEMA_VERSION) {
		LOG_ERR("schema version %lu, expected %d",
			(unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
		return -EINVAL;
	}

	static OpResolver resolver;
	resolver.AddConv2D();
	resolver.AddMaxPool2D();
	resolver.AddReshape();
	resolver.AddFullyConnected();
	resolver.AddSoftmax();

	static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
							   kArenaSize, nullptr, PROFILER);
	interpreter = &static_interpreter;

	if (interpreter->AllocateTensors() != kTfLiteOk) {
		LOG_ERR("AllocateTensors() failed: %u B arena too small?",
			(unsigned)kArenaSize);
		return -ENOMEM;
	}

	input = interpreter->input(0);
	output = interpreter->output(0);

	/* Check that the model is the one the rest of the code expects */
	if (input->type != kTfLiteInt8 || input->bytes != INFERENCE_IMAGE_SIZE ||
	    output->type != kTfLiteInt8 || output->bytes != INFERENCE_NUM_CLASSES) {
		LOG_ERR("unexpected input/output shape or type");
		return -EINVAL;
	}

	quant.in_scale = input->params.scale;
	quant.in_zero_point = input->params.zero_point;
	quant.out_scale = output->params.scale;
	quant.out_zero_point = output->params.zero_point;

	return 0;
}

extern "C" int inference_run(const int8_t *image, int8_t *scores)
{
	memcpy(input->data.int8, image, INFERENCE_IMAGE_SIZE);

#if defined(CONFIG_APP_LAYER_PROFILING)
	profiler.Reset();
#endif
	if (interpreter->Invoke() != kTfLiteOk) {
		return -EIO;
	}

	memcpy(scores, output->data.int8, INFERENCE_NUM_CLASSES);
	return 0;
}

extern "C" size_t inference_arena_used(void)
{
	return interpreter->arena_used_bytes();
}

extern "C" size_t inference_arena_size(void)
{
	return kArenaSize;
}

extern "C" const struct inference_quant *inference_get_quant(void)
{
	return &quant;
}

extern "C" int inference_last_profile(struct inference_layer_time *out, int max)
{
#if defined(CONFIG_APP_LAYER_PROFILING)
	return profiler.Copy(out, max);
#else
	ARG_UNUSED(out);
	ARG_UNUSED(max);
	return 0;
#endif
}

extern "C" const void *inference_arena_addr(void)
{
	return tensor_arena;
}
