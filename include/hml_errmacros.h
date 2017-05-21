/*****************************************************************
 *  Copyright (c) 2012, Palo Alto Research Center.               *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_ERRMACROS_H_INCLUDED_
#define HML_ERRMACROS_H_INCLUDED_

/*
 * hml_errmacros.h
 *
 *    This file contains the standard Graph Analytics error-handling
 *    macros.  The overall scheme for error handling (sometimes called
 *    "exception handling") is to use function return values as the
 *    conduit for the commumication of error information between
 *    procedures.  All Graph Analytics procedures should return the
 *    standard data type "HmlErrCode", and all calls to procedures should
 *    check error code returns from procedures and handle them
 *    appropriately. The macros defined in this file are designed to
 *    assist in the generation and handling of errors.
 */

#include "hml_types.h"
#include "hml_errors.h"
#include "hml_err_utils.h"

/* $Date: 2010-07-13 11:42:01 -0700 (Tue, 13 Jul 2010) $ */

/*! Declare and initialize the local error variable
 *
 *   A function prologue. It just defines the "__error__" variable, and
 *   initializes it. The only reason for initializing it is that most compilers
 *   whine about unused variables with no way to shut them up.
 */

#ifndef HML_ERR_PROLOGUE
#define HML_ERR_PROLOGUE \
    HmlErrCode __error__ = cHmlErrSuccess
#endif

/*! Check for an error condition.
 *
 *  Generate an error after testing a condition. "cond" is some
 *  condition (probably, it's just tested for non-zero) and "err" is
 *  the error code to generate
 */
#ifndef HML_ERR_GEN
#define HML_ERR_GEN(cond, err)\
do { \
    if (cond) \
    {\
        hmlPrintErrorInfo(__FILE__, __LINE__, err);\
        return err;\
    }\
} while(0)
#endif  /* HML_ERR_GEN defined */

/*! Check for an error condition.
 *
 *  Generate an error after testing a condition. "cond" is some
 *  condition (probably, it's just tested for non-zero), "err" is
 *  the error code to generate, and "ext" is the extra information
 *  of "char const *" type (i.e., a string) that is associated
 *  with the error
 */
#ifndef HML_ERR_GEN_EXT
#define HML_ERR_GEN_EXT(cond, err, ext)\
do { \
    if (cond) \
    {\
        hmlPrintErrorInfoExt(__FILE__, __LINE__, err, ext);\
        return err;\
    }\
} while(0)
#endif  /* HML_ERR_GEN defined */

/*! Check for an error condition.
 *
 *  Generate an error with NO error message after testing a condition.
 * "cond" is some condition (probably, it's just tested for non-zero) and
 * "err" is the error code to generate
 */
#ifndef HML_ERR_GEN_SILENT
#define HML_ERR_GEN_SILENT(cond, err)\
do { \
    if (cond) \
    {\
        hmlSilentErrorInfo(__FILE__, __LINE__, err);\
        return err;\
    }\
} while(0)
#endif  /* HML_ERR_GEN defined */

/*! Check for an error condition; execute cleanup code when the error occurs
 *
 *   Error generation: instead of returning, as in HML_ERR_GEN, go to a cleanup
 *   action at the end of the function.
 */

#ifndef HML_ERR_GEN_CLEANUP
#define HML_ERR_GEN_CLEANUP(cond, err)\
do {\
    if (cond)\
    {\
        hmlPrintErrorInfo(__FILE__, __LINE__, err);\
        __error__ = err;\
        goto ERR_CLEANUP;\
    }\
} while(0)
#endif  /* HML_ERR_GEN_CLEANUP defined */


/*! Just complain about an error.  used when returning is difficult.
 *
 *   Error generation: instead of returning, as in HML_ERR_GEN, go to a cleanup
 *   action at the end of the function.
 */

#ifndef HML_ERR_GEN_REPORT
#define HML_ERR_GEN_REPORT(cond, err)\
do {\
    if (cond)\
    {\
        hmlPrintErrorInfo(__FILE__, __LINE__, err);\
    }\
} while(0)
#endif  /* HML_ERR_GEN_REPORT defined */


/*! Check for error; execute following block if error occurs
 *
 *  Generate an error from a condition.  Unlike HML_ERR_GEN, this macro
 *  expects a statement or block to follow it, and will execute that
 *  statement or block if an error ocurred in the function call.
 */

#ifndef HML_ERR_GEN_CHECK
#define HML_ERR_GEN_CHECK(cond, err)\
    if (cond) \
    {\
        __error__ = err;\
        hmlPrintErrorInfo(__FILE__, __LINE__, err);\
    }\
    if (cond)
#endif  /* HML_ERR_GEN_CHECK defined */

/*! Call a routine, check return code for errors
 *
 *   This calls a function and checks the return code for errors.  It
 *   assumes that the function returns an error code, and if the error
 *   code is non-zero, returns from the function (after possibly
 *   printing an error message).
 */

#ifndef HML_ERR_PASS
#define HML_ERR_PASS(functionCall)\
do {\
    __error__ = functionCall;\
    if (__error__ != cHmlErrSuccess)\
    {\
        hmlPrintErrorInfo(__FILE__, __LINE__, __error__);\
        return __error__;\
    }\
} while (0)
#endif  /* HML_ERR_PASS defined */

/*! Call a routine, check return code for errors
*
*   This calls a function and checks the return code for errors.  It
*   assumes that the function returns an error code, and if the error
*   code is non-zero, returns from the function without
*   printing an error message.
*/

#ifndef HML_ERR_PASS_SILENT
#define HML_ERR_PASS_SILENT(functionCall)\
do {\
    __error__ = functionCall;\
    if (__error__ != cHmlErrSuccess)\
    {\
        return __error__;\
    }\
} while (0)
#endif  /* HML_ERR_PASS_SILENT defined */

/*! Call a routine, check return code for errors, branch to cleanup
 *  code if the error occurs.
 *
 *   This is an error check on a function call. It is like HML_ERR_PASS except
 *   that rather than returning from the function, it sets the local
 *   "__error__" variable, and does a "goto" to some cleanup code. The
 *   cleanup code is labeled "ERR_CLEANUP". The cleanup should have an
 *   invocation of the "ERROR_RETURN" macro at the end of it.
 */

#ifndef HML_ERR_PASS_CLEANUP
#define HML_ERR_PASS_CLEANUP(functionCall)\
do {\
    __error__ = functionCall;\
    if (__error__ != cHmlErrSuccess)\
    {\
        hmlPrintErrorInfo(__FILE__, __LINE__, __error__);\
        goto ERR_CLEANUP;\
    }\
} while (0)
#endif  /* HML_ERR_PASS_CLEANUP defined */

/*! Call a routine, check for error, always continue on.
 *
 *   This is an error check on a function call. It is like HML_ERR_PASS except
 *   that rather than returning from the function, it sets the local
 *   "__error__" variable, and invokes hmlPrintErrorInfo if there
 *   was a problem, but does NOT return.  This is used when immediately
 *   returning or going to a cleanup routine isn't sensible.  Examples of
 *   this are when we're already in a cleanup routine or we're in a
 *   routine that does not return an HmlErrCode.   This is also used to make
 *   static code analysis tools happy - they don't like to see function
 *   call return values ignored.
 */

#ifndef HML_ERR_REPORT
#define HML_ERR_REPORT(functionCall)\
do {\
    __error__ = functionCall;\
    if (__error__ != cHmlErrSuccess)\
    {\
        hmlPrintErrorInfo(__FILE__, __LINE__, __error__);\
    }\
} while (0)
#endif  /* HML_ERR_REPORT defined */

/*! Call a routine; print out error messages; execute following block
 *   if error occurs.
 *
 *   This is a simpler variant of an error check on a function call.
 *   It expects a statement or block to follow it, and will execute
 *   that statement or block if an error ocurred in the function call.
 */

#ifndef HML_ERR_CHECK
#define HML_ERR_CHECK(functionCall)\
    __error__ = functionCall;\
    if (__error__ != cHmlErrSuccess)\
        hmlPrintErrorInfo(__FILE__, __LINE__, __error__);\
    if (__error__ != cHmlErrSuccess)
#endif  /* HML_ERR_CHECK defined */

/*! Call a routine, ignore what's returned
 *
 *   This is used whenever the function call returns a void or some
 *   other value that is not an HmlErrCode.  This keeps any scripts that
 *   check to make sure that all function calls are wrapped in error
 *   macros from complaining.
 */
#ifndef HML_ERR_IGNORE
#define HML_ERR_IGNORE(functionCall) functionCall
#endif  /* HML_ERR_IGNORE defined */

/*! Return from a function after cleanup code is executed
 *
 *   This is used to return the error code at the end of a cleanup
 *   routine. The macro seems simple here, but could change, e.g.
 *   to add checking for memory leaks by local allocators, to add tracing
 *   code, etc., so we suggest that everyone use it rigorously.
 */

#ifndef HML_ERR_RETURN
#define HML_ERR_RETURN return __error__
#endif  /* HML_ERR_RETURN defined */

/*! Return from a function normally
 *
 *   This is used to return from a function after the function completes
 *   successfully. The macro seems simple here, but could change, e.g.
 *   to add checking for memory leaks by local allocators, to add tracing
 *   code, etc., so we suggest that everyone use it rigorously.
 *   The weird "if (1)" is there to make Forte 6.0 happy.  If it isn't
 *   there, the compiler complains that the code at the end of the
 *   loop isn't reached, which is the point after all.
 *   On the other hand, gcc 4.0.0 whines about not reaching the end of
 *   non-void functions if you have the "if (1)".
 */

#ifndef HML_NORMAL_RETURN
#ifdef __GNUC__
#define HML_NORMAL_RETURN do { return __error__;} while (0)
#else
#define HML_NORMAL_RETURN do { if (1) return __error__;} while (0)
#endif
#endif  /* HML_NORMAL_RETURN defined */

/*! Check a condition and return NULL if condition is false
 *  HML_NULL_CHECK is the opposite of HML_NULL_GEN
 */

#ifndef HML_NULL_CHECK
#define HML_NULL_CHECK(cond)\
do {\
    if (!(cond))\
    {\
        return NULL;\
    }\
} while (0)
#endif  /* HML_NULL_CHECK defined */

/*! Check a condition and return NULL if condition is true
 *  HML_NULL_GEN is the opposite of HML_NULL_CHECK
 */

#ifndef HML_NULL_GEN
#define HML_NULL_GEN(cond)\
do {\
    if (cond)\
    {\
        return NULL;\
    }\
} while (0)
#endif  /* HML_NULL_GEN defined */

/*! Check a condition and go to ERR_CLEANUP if condition is false
 *  HML_NULL_CHECK_CLEANUP is the opposite of HML_CHECK_CLEANUP
 */

#ifndef HML_NULL_CHECK_CLEANUP
#define HML_NULL_CHECK_CLEANUP(cond)\
do {\
    if (!(cond))\
    {\
        goto ERR_CLEANUP;\
    }\
} while (0)
#endif  /* HML_NULL_CHECK_CLEANUP defined */

/*! Check for an error condition; execute cleanup code when the error occurs
 *  This macro does NOT need __error__ variable to be declared
 *   instead of generating an error, as in HML_ERR_GEN, go to a cleanup
 *   action at the end of the function.
 */

#ifndef HML_CHECK_CLEANUP
#define HML_CHECK_CLEANUP(cond)\
do {\
    if (cond)\
    {\
        goto ERR_CLEANUP;\
    }\
} while(0)
#endif  /* HML_CHECK_CLEANUP defined */

/*! Check a condition and return ZERO if condition is false
 *  HML_ZERO_CHECK is the opposite of HML_ZERO_GEN
 */

#ifndef HML_ZERO_CHECK
#define HML_ZERO_CHECK(cond)\
do {\
    if (!(cond))\
    {\
        return 0;\
    }\
} while (0)
#endif  /* HML_ZERO_CHECK defined */

/*! Check a condition and return 0 if condition is true
 *  HML_ZERO_GEN is the opposite of HML_ZERO_CHECK
 */

#ifndef HML_ZERO_GEN
#define HML_ZERO_GEN(cond)\
do {\
    if (cond)\
    {\
        return 0;\
    }\
} while (0)
#endif  /* HML_ZERO_GEN defined */

#endif /* HML_ERRMACROS_H_INCLUDED_ */
