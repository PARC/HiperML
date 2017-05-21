/*****************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

/*
 * hml_err_utils.h:  Error utility routines for HiperML
 */

#ifndef HML_ERR_UTILS_H_INCLUDED_
#define HML_ERR_UTILS_H_INCLUDED_

/* similar to vsnprintf() on Linux, except that the maximum
 * returned value <= 'count' and the equality happens if and only if
 * buffer 'str' is not big enough to hold the result.
 * This function works the same on both Linux and Windows.
 * Note: Both Linux and Windows' native vsnprintf() implementations are
 *       slightly different from hmlVsnprintf()
 */
int32_t
hmlVsnprintf(char *str, size_t count, char const *format, va_list args);

/* exactly same as snprintf() on Linux, but works also on Windows now
 * Note: Windows' native snprintf() implementation is
 *       slightly different from that of Linux
 */
int32_t
hmlSnprintf(char *str, size_t count, char const *format, ...);

HmlErrCode
hmlPrintErrorInfo(char const *filename, int32_t lineNum, HmlErrCode error);

HmlErrCode
hmlPrintErrorInfoExt(char const *filename,
                     int32_t     lineNum,
                     HmlErrCode  error,
                     char const *extra);

HmlErrCode
hmlSilentErrorInfo(const char *filename, int32_t lineNum, HmlErrCode error);

HmlErrCode
hmlPrintf(const char *format, ...);

#endif /* HML_ERR_UTILS_H_INCLUDED_ */
