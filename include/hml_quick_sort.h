/*****************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.               *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_QUICK_SORT_H_INCLUDED_
#define HML_QUICK_SORT_H_INCLUDED_

/* this file contains the prototypes for the routines that are used to
 * perform thread-safe quick-sort by allowing an additional contextual
 * parameter called 'arg' in qsort.
 * This header file must be included before any other
 * header files such that the _GNU_SOURCE macro is properly '#define'ed
 */

#if (defined __GNU__ || defined __linux__ || \
  defined __linux__ ) || defined(__CYGWIN__)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

/* make prototypes usable from C++ */
#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

/* macro HML_QSORT finds the appropriate qsort_{r,s} functions
 * depending on the OS
 */
#if (defined _GNU_SOURCE || defined __GNU__ || \
  defined __linux__ ) || defined(__CYGWIN__)
#define HML_QSORT(base, num, width, compare, arg)\
  qsort_r(base, num, width, compare, arg)
#elif (defined __APPLE__ || defined __MACH__ || defined __DARWIN__ || \
defined __FREEBSD__ || defined __BSD__ || \
defined OpenBSD3_1 || defined OpenBSD3_9)
#define HML_QSORT(base, num, width, compare, arg)\
  qsort_r(base, num, width, arg, compare)
#elif (defined _WIN32 || defined _WIN64 || defined __WINDOWS__)
#define HML_QSORT(base, num, width, compare, arg)\
  qsort_s(base, num, width, compare, arg)
#else
#error Cannot detect operating system
#endif

/* macro HML_QSORT_COMPARE_FUNC should be used to define the
 * arguments of the comparison function used in qsort_{r,s}
 * the reason for needing this macro is that different OSes
 * have different orders for these arguments
 * Example:
 * int HML_QSORT_COMPARE_FUNC(myIntCompare, a, b, arg) {
 *   int *array = (int *)arg;
 *   return array[*(const int *)a] - array[*(const int *)b];
 * }
 */
#if (defined _GNU_SOURCE || defined __GNU__ ||  \
  defined __linux__ ) || defined(__CYGWIN__)
#define HML_QSORT_COMPARE_FUNC(f, a, b, arg)\
  f(const void *a, const void *b, void *arg)
#elif (defined __APPLE__ || defined __MACH__ || defined __DARWIN__ || \
  defined __FREEBSD__ || defined __BSD__ || \
  defined OpenBSD3_1 || defined OpenBSD3_9)
#define HML_QSORT_COMPARE_FUNC(f, a, b, arg)\
  f(void *arg, const void *a, const void *b)
#elif (defined _WIN32 || defined _WIN64 || defined __WINDOWS__)
#define HML_QSORT_COMPARE_FUNC(f, a, b, arg)\
  f(void *arg, const void *a, const void *b)
#else
#error Cannot detect operating system
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  /* HML_QUICK_SORT_H_INCLUDED_ */
