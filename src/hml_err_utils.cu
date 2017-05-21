/*****************************************************************
 *  Copyright (c) 2017 Palo Alto Research Center.                *
 *  All rights reserved.                                         *
 *****************************************************************/

/*
 *  hml_err_utils.cu:
 */
#include "hml_common.h"
#include <stdarg.h>

/* similar to vsnprintf() on Linux, except that the maximum
 * returned value <= 'count' and the equality happens if and only if
 * buffer 'str' is not big enough to hold the result.
 * This function works the same on both Linux and Windows.
 * Note: Both Linux and Windows' native vsnprintf() implementations are
 *       slightly different from hmlVsnprintf()
 */
int32_t
hmlVsnprintf(char *str, size_t count, char const *format, va_list args) {
  int32_t len;

  len = vsnprintf(str, count, format, args);
#ifdef _WIN32
  /* Windows does NOT put the null-terminator if strlen(str) >= count */
  if(len < 0 || len == (int32_t)count) {
    str[count - 1] = '\0';
    len = (int32_t)count;
  }
#endif
  /* Linux always put the null-terminator in the end, but it may return
   * a value greater than count, if the actual string is longer than 'str'
   */
  if(len > (int32_t)count) {
    len = (int32_t)count;
  }
  return len;
}

/* exactly same as snprintf() on Linux, but works also on Windows now
 * Note: Windows' native snprintf() implementation is
 *       slightly different from that of Linux
 */
int32_t
hmlSnprintf(char *str, size_t count, char const *format, ...) {
  int32_t   len;
  va_list args;

  va_start(args, format);
  len = hmlVsnprintf(str, count, format, args);
  va_end(args);
  return len;
}

HmlErrCode
hmlPrintErrorInfo(char const *filename, int32_t lineNum, HmlErrCode error) {
  HML_ERR_PROLOGUE;
  char  errorString[1024];

  hmlSnprintf(errorString, sizeof(errorString),
              "--HiperML Error-- File:%s Line:%ld "
              "Error: %ld\n",
              filename, (long int)lineNum, (long int)error);
  fprintf(stderr, "%s", errorString);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPrintErrorInfoExt(char const *filename,
                     int32_t     lineNum,
                     HmlErrCode  error,
                     char const *extra) {
  HML_ERR_PROLOGUE;
  char  errorString[1024];

  hmlSnprintf(errorString, sizeof(errorString),
              "--HiperML Error-- File:%s Line:%ld "
              "Error: %ld with %s\n",
              filename, (long int)lineNum, (long int)error, extra);
  fprintf(stderr, "%s", errorString);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlSilentErrorInfo(const char *filename, int32_t lineNum, HmlErrCode error) {
  HML_ERR_PROLOGUE;

  /* avoid compiler warnings */
  (void) filename;
  (void) lineNum;
  (void) error;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlPrintf(const char *format, ...) {
  HML_ERR_PROLOGUE;
  char        errorString[1024];
  va_list     args;

  va_start(args, format);
  vsprintf(&(errorString[0]), format, args);
  va_end(args);
  fprintf(stderr, "%s", errorString);

  HML_NORMAL_RETURN;
}
