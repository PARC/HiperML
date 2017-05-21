/*************************************************************
 *  Copyright (c) 2017, Palo Alto Research Center.           *
 *  All rights reserved.                                     *
 *************************************************************/

#ifndef HML_THREAD_GROUP_H_INCLUDED_
#define HML_THREAD_GROUP_H_INCLUDED_
#include <pthread.h>
#include "hml_common.h"

typedef HmlErrCode(*HmlThreadFunc)(uint32_t    thread,
                                   void     *args);

typedef struct HmlThreadGroupStruct HmlThreadGroup;

typedef struct {
  uint32_t        thread;
  HmlErrCode      errCode;
  HmlThreadGroup *group;
  void           *args;
} HmlThreadArgs;

struct HmlThreadGroupStruct {
  uint32_t       size;   /* # of threads in the group */
  HmlThreadArgs *args;
  HmlThreadFunc  func;
  pthread_t     *pthreads; /* pointer to 'size' # of POSIX threads */
  /* By default, all threads are enabled, i.e., disabled[t] == FALSE */
  bool          *disabled; /* pointer to 'size' # of boolean flags */
  /* all threads in the same group share the same attribute */
  pthread_attr_t attr;
};

/* make prototypes usable from C++ */
#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus */

HmlErrCode
hmlThreadGroupInit(HmlThreadGroup *group,
                   uint32_t        size);

HmlErrCode
hmlThreadGroupResize(HmlThreadGroup *group,
                     uint32_t        newSize);

HmlErrCode
hmlThreadGroupEnableThread(HmlThreadGroup *group,
                           uint32_t        thread);

HmlErrCode
hmlThreadGroupDisableThread(HmlThreadGroup *group,
                            uint32_t        thread);

HmlErrCode
hmlThreadGroupSetErrorCode(HmlThreadGroup *group,
                           uint32_t        thread,
                           HmlErrCode      errCode);

HmlErrCode
hmlThreadGroupError(HmlThreadGroup *group);

/* 'VarArgs' means all threads get their own private version of 'args'
 * and thus, one needs to specify the size of each 'args' using the
 * 'argSize' parameter
 */
HmlErrCode
hmlThreadGroupRunVarArgs(HmlThreadGroup *group,
                         HmlThreadFunc   func,
                         void           *args,
                         size_t          argSize);

/* All threads get the same 'args', and thus no need to specify
 * the size of 'args', unlike the ...VarArgs() function above
 */
HmlErrCode
hmlThreadGroupRun(HmlThreadGroup *group,
                  HmlThreadFunc   func,
                  void           *args);

HmlErrCode
hmlThreadGroupDelete(HmlThreadGroup *group);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* HML_THREAD_GROUP_H_INCLUDED_ */
