/*
 * digit_touch — model self-test on fixed MNIST images.
 */

#ifndef SELFTEST_H_
#define SELFTEST_H_

/* Compare the board's outputs with the PC's and measure the inference time.
 * Returns the number of images whose output differs from the PC.
 */
int model_selftest(void);

#endif /* SELFTEST_H_ */
