/*
 * digit_touch — LVGL UI: drawing area + Clear button.
 *
 * Drawing is done directly into an 8-bit array (ink[]) that we fully
 * control: white stroke on a black background, like MNIST. LVGL only
 * displays it, through an L8 (8-bit luminance) canvas that uses this same
 * array as its buffer. Preprocessing reads ink[] directly, without
 * going through LVGL.
 */

#include "ui.h"

#include <stdlib.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <lvgl.h>

LOG_MODULE_REGISTER(ui, LOG_LEVEL_INF);

/* Brush radius in screen pixels. A digit drawn ~180 px tall with a ~17 px
 * stroke roughly keeps the MNIST proportions (2-3 px stroke for a 20 px
 * digit).
 */
#define BRUSH_RADIUS 8
#define CANVAS_X     10
#define CANVAS_Y     10

/* Results panel, to the right of the drawing area (480x272 screen) */
#define PANEL_X   278
#define BARS_Y    82
#define BAR_PITCH 14
#define BAR_W     120
#define BAR_H     8

static uint8_t ink[DRAW_SIZE * DRAW_SIZE] __aligned(4);

static lv_obj_t *canvas;
static lv_obj_t *digit_label;
static lv_obj_t *pct_label;
static lv_obj_t *status_label;
static lv_obj_t *bars[INFERENCE_NUM_CLASSES];
static lv_obj_t *bar_pct[INFERENCE_NUM_CLASSES];

/* Time the last result was displayed (automatic clearing) */
static bool result_shown;
static int64_t result_time_ms;

/* End of drawing: timer restarted on every stroke */
static lv_timer_t *done_timer;
static ui_drawing_done_cb_t drawing_done_cb;

/* State of the current stroke */
static bool stroke_active;
static int16_t last_x, last_y;
static uint32_t stroke_count;

/* Bounding box of all the ink */
static bool has_ink;
static struct ink_bbox bbox;

/* ------------------------------------------------------------------------ */
/* Drawing into ink[]                                                       */
/* ------------------------------------------------------------------------ */

/* Filled disk of radius BRUSH_RADIUS centered at (cx, cy), clipped at the edges */
static void stamp_disk(int cx, int cy)
{
	for (int dy = -BRUSH_RADIUS; dy <= BRUSH_RADIUS; dy++) {
		int y = cy + dy;

		if (y < 0 || y >= DRAW_SIZE) {
			continue;
		}
		for (int dx = -BRUSH_RADIUS; dx <= BRUSH_RADIUS; dx++) {
			int x = cx + dx;

			if (x >= 0 && x < DRAW_SIZE &&
			    dx * dx + dy * dy <= BRUSH_RADIUS * BRUSH_RADIUS) {
				ink[y * DRAW_SIZE + x] = 255;
			}
		}
	}
}

/* The touch panel only reports one point every ~10-20 ms: on a fast
 * gesture, two successive points can be 20-30 px apart. So the points are
 * joined by stamping the brush at every pixel along the segment, otherwise
 * the stroke would be a series of separate dots.
 */
static void draw_segment(int x0, int y0, int x1, int y1)
{
	int dx = x1 - x0;
	int dy = y1 - y0;
	int steps = MAX(abs(dx), abs(dy));
	lv_area_t c, area;

	if (steps == 0) {
		stamp_disk(x0, y0);
	} else {
		for (int i = 0; i <= steps; i++) {
			stamp_disk(x0 + dx * i / steps, y0 + dy * i / steps);
		}
	}

	/* Total ink, clamped to the drawing area */
	int xmin = CLAMP(MIN(x0, x1) - BRUSH_RADIUS, 0, DRAW_SIZE - 1);
	int xmax = CLAMP(MAX(x0, x1) + BRUSH_RADIUS, 0, DRAW_SIZE - 1);
	int ymin = CLAMP(MIN(y0, y1) - BRUSH_RADIUS, 0, DRAW_SIZE - 1);
	int ymax = CLAMP(MAX(y0, y1) + BRUSH_RADIUS, 0, DRAW_SIZE - 1);

	if (!has_ink) {
		bbox = (struct ink_bbox){xmin, ymin, xmax, ymax};
		has_ink = true;
	} else {
		bbox.x_min = MIN(bbox.x_min, xmin);
		bbox.y_min = MIN(bbox.y_min, ymin);
		bbox.x_max = MAX(bbox.x_max, xmax);
		bbox.y_max = MAX(bbox.y_max, ymax);
	}

	/* Redraw on screen only the small modified area (screen coordinates) */
	lv_obj_get_coords(canvas, &c);
	area.x1 = c.x1 + xmin;
	area.y1 = c.y1 + ymin;
	area.x2 = c.x1 + xmax;
	area.y2 = c.y1 + ymax;
	lv_obj_invalidate_area(canvas, &area);
}

/* Empty results panel: prompt to draw, bars at zero */
static void clear_result(void)
{
	result_shown = false;
	lv_label_set_text(digit_label, "-");
	lv_obj_set_style_text_color(digit_label, lv_palette_main(LV_PALETTE_GREY), 0);
	lv_label_set_text(pct_label, "");
	lv_label_set_text(status_label, "Draw a digit");

	for (int i = 0; i < INFERENCE_NUM_CLASSES; i++) {
		lv_bar_set_value(bars[i], 0, LV_ANIM_ON);
		lv_obj_set_style_bg_color(bars[i], lv_palette_main(LV_PALETTE_BLUE_GREY),
					  LV_PART_INDICATOR);
		lv_label_set_text(bar_pct[i], "");
	}
}

static void clear_drawing(void)
{
	memset(ink, 0, sizeof(ink));
	has_ink = false;
	stroke_active = false;
	stroke_count = 0;
	lv_timer_pause(done_timer);
	lv_obj_invalidate(canvas);
	clear_result();
}

void ui_show_result(const int permille[INFERENCE_NUM_CLASSES], int best, int second, bool sure)
{
	/* Green if the model is confident, orange otherwise */
	lv_color_t accent = lv_palette_main(sure ? LV_PALETTE_GREEN : LV_PALETTE_ORANGE);

	result_shown = true;
	result_time_ms = k_uptime_get();

	if (sure) {
		lv_label_set_text_fmt(digit_label, "%d", best);
		lv_label_set_text(status_label, "Recognized");
	} else {
		lv_label_set_text(digit_label, "?");
		lv_label_set_text_fmt(status_label, "Not sure: %d or %d?", best, second);
	}
	lv_obj_set_style_text_color(digit_label, accent, 0);
	lv_label_set_text_fmt(pct_label, "%d %%", (permille[best] + 5) / 10);

	for (int i = 0; i < INFERENCE_NUM_CLASSES; i++) {
		lv_bar_set_value(bars[i], permille[i], LV_ANIM_ON);
		lv_obj_set_style_bg_color(bars[i],
					  i == best ? accent
						    : lv_palette_main(LV_PALETTE_BLUE_GREY),
					  LV_PART_INDICATOR);
		lv_label_set_text_fmt(bar_pct[i], "%d %%", (permille[i] + 5) / 10);
	}
}

/* ------------------------------------------------------------------------ */
/* Events                                                                   */
/* ------------------------------------------------------------------------ */

static void canvas_event_cb(lv_event_t *e)
{
	lv_event_code_t code = lv_event_get_code(e);
	lv_point_t p;
	lv_area_t c;

	/* New digit started while the previous one is displayed: clear */
	if (code == LV_EVENT_PRESSED && result_shown &&
	    k_uptime_get() - result_time_ms >= AUTO_CLEAR_DELAY_MS) {
		clear_drawing();
		LOG_INF("new digit: drawing cleared automatically");
	}

	switch (code) {
	case LV_EVENT_PRESSED:
	case LV_EVENT_PRESSING:
		lv_indev_get_point(lv_indev_active(), &p);
		lv_obj_get_coords(canvas, &c);
		/* Coordinates relative to the drawing area. No clamping: a finger
		 * that leaves the area and comes back keeps a continuous stroke,
		 * stamp_disk() simply clips whatever lies outside.
		 */
		p.x -= c.x1;
		p.y -= c.y1;

		if (code == LV_EVENT_PRESSED || !stroke_active) {
			/* New stroke: the drawing is not finished */
			lv_timer_pause(done_timer);
			stroke_active = true;
			stroke_count++;
			draw_segment(p.x, p.y, p.x, p.y);
		} else {
			draw_segment(last_x, last_y, p.x, p.y);
		}
		last_x = p.x;
		last_y = p.y;
		break;

	case LV_EVENT_RELEASED:
	case LV_EVENT_PRESS_LOST:
		if (stroke_active) {
			stroke_active = false;
			LOG_INF("stroke %u done, ink x=[%d..%d] y=[%d..%d]", stroke_count,
				bbox.x_min, bbox.x_max, bbox.y_min, bbox.y_max);
			/* If no stroke starts within DRAWING_DONE_DELAY_MS, it is finished */
			lv_timer_reset(done_timer);
			lv_timer_resume(done_timer);
		}
		break;

	default:
		break;
	}
}

static void done_timer_cb(lv_timer_t *t)
{
	lv_timer_pause(t);
	if (drawing_done_cb != NULL && has_ink) {
		drawing_done_cb();
	}
}

static void clear_btn_cb(lv_event_t *e)
{
	ARG_UNUSED(e);
	clear_drawing();
	LOG_INF("drawing cleared");
}

/* ------------------------------------------------------------------------ */
/* Screen construction                                                      */
/* ------------------------------------------------------------------------ */

void ui_init(ui_drawing_done_cb_t done_cb)
{
	lv_obj_t *scr = lv_screen_active();
	lv_obj_t *btn, *btn_label;

	drawing_done_cb = done_cb;
	done_timer = lv_timer_create(done_timer_cb, DRAWING_DONE_DELAY_MS, NULL);
	lv_timer_pause(done_timer);

	lv_obj_set_style_bg_color(scr, lv_color_hex(0x202020), 0);
	lv_obj_set_scrollable(scr, false);

	/* Drawing area: L8 canvas whose buffer is ink[] */
	if (lv_draw_buf_width_to_stride(DRAW_SIZE, LV_COLOR_FORMAT_L8) != DRAW_SIZE) {
		LOG_ERR("unexpected L8 stride: check CONFIG_LV_DRAW_BUF_STRIDE_ALIGN");
	}
	canvas = lv_canvas_create(scr);
	lv_canvas_set_buffer(canvas, ink, DRAW_SIZE, DRAW_SIZE, LV_COLOR_FORMAT_L8);
	lv_obj_set_pos(canvas, CANVAS_X, CANVAS_Y);
	lv_obj_set_style_outline_width(canvas, 2, 0);
	lv_obj_set_style_outline_color(canvas, lv_palette_main(LV_PALETTE_GREY), 0);
	/* A canvas (image) is not clickable by default */
	lv_obj_set_clickable(canvas, true);
	lv_obj_add_event_cb(canvas, canvas_event_cb, LV_EVENT_ALL, NULL);

	/* Right panel: result */
	digit_label = lv_label_create(scr);
	lv_obj_set_style_text_font(digit_label, &lv_font_montserrat_48, 0);
	lv_obj_set_pos(digit_label, PANEL_X + 4, 0);

	pct_label = lv_label_create(scr);
	lv_obj_set_style_text_font(pct_label, &lv_font_montserrat_24, 0);
	lv_obj_set_pos(pct_label, PANEL_X + 60, 14);

	status_label = lv_label_create(scr);
	lv_obj_set_style_text_color(status_label, lv_palette_lighten(LV_PALETTE_GREY, 2), 0);
	lv_obj_set_pos(status_label, PANEL_X + 4, 54);

	/* One confidence bar per digit: "N [=====    ] xx %" */
	for (int i = 0; i < INFERENCE_NUM_CLASSES; i++) {
		int y = BARS_Y + i * BAR_PITCH;
		lv_obj_t *name = lv_label_create(scr);

		lv_obj_set_style_text_color(name, lv_color_white(), 0);
		lv_label_set_text_fmt(name, "%d", i);
		lv_obj_set_pos(name, PANEL_X + 4, y - 4);

		bars[i] = lv_bar_create(scr);
		lv_bar_set_range(bars[i], 0, 1000);
		lv_obj_set_size(bars[i], BAR_W, BAR_H);
		lv_obj_set_pos(bars[i], PANEL_X + 20, y);
		lv_obj_set_style_bg_color(bars[i], lv_color_hex(0x404040), LV_PART_MAIN);
		lv_obj_set_style_bg_opa(bars[i], LV_OPA_COVER, LV_PART_MAIN);
		lv_obj_set_style_anim_duration(bars[i], 250, 0);

		bar_pct[i] = lv_label_create(scr);
		lv_obj_set_style_text_color(bar_pct[i], lv_palette_lighten(LV_PALETTE_GREY, 2), 0);
		lv_obj_set_width(bar_pct[i], 44);
		lv_obj_set_style_text_align(bar_pct[i], LV_TEXT_ALIGN_RIGHT, 0);
		lv_obj_set_pos(bar_pct[i], PANEL_X + 20 + BAR_W + 2, y - 4);
	}

	btn = lv_button_create(scr);
	lv_obj_set_size(btn, 185, 40);
	lv_obj_set_pos(btn, PANEL_X + 4, 227);
	lv_obj_add_event_cb(btn, clear_btn_cb, LV_EVENT_CLICKED, NULL);
	btn_label = lv_label_create(btn);
	lv_obj_set_style_text_font(btn_label, &lv_font_montserrat_24, 0);
	lv_label_set_text(btn_label, "Clear");
	lv_obj_center(btn_label);

	clear_drawing();
}

const uint8_t *ui_drawing(void)
{
	return ink;
}

bool ui_drawing_bbox(struct ink_bbox *out)
{
	if (has_ink) {
		*out = bbox;
	}
	return has_ink;
}
