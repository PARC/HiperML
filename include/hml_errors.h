/*****************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

/*
 * hpg_errors.h:      Contains error return codes for HiperML
 */

#ifndef HML_ERRORS_H_INCLUDED_
#define HML_ERRORS_H_INCLUDED_

/*! This has been a placeholder for undefined errors.  It shouldn't happen */
#define cHmlErrUndefined    -1

/*! The status code that's returned from any of our routines when
 *  there's no error.
 */
#define cHmlErrSuccess      0

/*! The general error */
#define cHmlErrGeneral                  1 /* something bad happened */

#define cHmlErrUnknown                  2 /* error unknown */

/*! this error is raised when a client-supplied buffer is too small. */
#define cHmlErrClientBufTooSmall        3

#define cHmlErrOutofMemory              4 /* out of memory */
#define cHmlErrInvalidArg               5 /* invalid query parameter */
#define cHmlErrInvalidArgName           6 /* invalid query parameter name */
#define cHmlErrInvalidArgType           7 /* invalid query parameter type */
#define cHmlErrInvalidArgValue          8 /* invalid query parameter value */
#define cHmlErrMissingArg               9 /* missing query parameter */

#endif /* HML_ERRORS_H_INCLUDED_ */
