/*****************************************************************
*  Copyright (c) 2017. Palo Alto Research Center                *
*  All rights reserved.                                         *
*****************************************************************/

/* hml_thread_group.c
 */

#include "hml_thread_group.h"

HmlErrCode
hmlThreadGroupInit(HmlThreadGroup *group,
                   uint32_t        size) {
  HML_ERR_PROLOGUE;
  uint32_t  thread;
  int32_t   pthreadErrCode;

  group->size = size;
  CALLOC(group->args, HmlThreadArgs, size);
  for(thread = 0; thread < size; thread++) {
    group->args[thread].thread = thread;
    group->args[thread].group = group;
  }
  /* allocate pthread objects */
  CALLOC(group->pthreads, pthread_t, size);
  /* initially all threads are enabled, i.e., disabled[t] == false */
  CALLOC(group->disabled, bool, size);
  /* declare that the pthreads to be created are joinable */
  pthreadErrCode = pthread_attr_init(&group->attr);
  HML_ERR_GEN(pthreadErrCode, cHmlErrGeneral);
  pthreadErrCode =
    pthread_attr_setdetachstate(&group->attr, PTHREAD_CREATE_JOINABLE);
  HML_ERR_GEN(pthreadErrCode, cHmlErrGeneral);

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlThreadGroupResize(HmlThreadGroup *group,
                     uint32_t        newSize) {
  HML_ERR_PROLOGUE;
  uint32_t  thread;

  REALLOC(group->args, HmlThreadArgs, newSize);
  for(thread = 0; thread < newSize; thread++) {
    group->args[thread].thread = thread;
    group->args[thread].group = group;
  }
  /* re-allocate pthread objects */
  REALLOC(group->pthreads, pthread_t, newSize);
  /* re-allocate disable flags */
  REALLOC(group->disabled, bool, newSize);
  /* initially all threads are enabled, i.e., disabled[t] == false */
  memset(group->disabled, 0, sizeof(bool) * newSize);
  /* update the thread group size */
  group->size = newSize;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlThreadGroupEnableThread(HmlThreadGroup *group,
                           uint32_t        thread) {
  HML_ERR_PROLOGUE;

  HML_ERR_GEN(thread >= group->size, cHmlErrGeneral);
  group->disabled[thread] = false;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlThreadGroupDisableThread(HmlThreadGroup *group,
                            uint32_t        thread) {
  HML_ERR_PROLOGUE;

  HML_ERR_GEN(thread >= group->size, cHmlErrGeneral);
  group->disabled[thread] = true;

  HML_NORMAL_RETURN;
}

static void *
hmlThread(void *threadArgs) {
  HmlThreadArgs  *args = (HmlThreadArgs *)threadArgs;
  HmlThreadGroup *group = args->group;

  if(group->func) {
    args->errCode = (*group->func)(args->thread, args->args);
  }

  return NULL;
}

HmlErrCode
hmlThreadGroupSetErrorCode(HmlThreadGroup *group,
                           uint32_t          thread,
                           HmlErrCode      errCode) {
  HML_ERR_PROLOGUE;

  HML_ERR_GEN(thread >= group->size, cHmlErrGeneral);
  group->args[thread].errCode = errCode;

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlThreadGroupError(HmlThreadGroup *group) {
  HML_ERR_PROLOGUE;
  uint32_t   thread;

  /* check for error code returned by each thread */
  for(thread = 0; thread < group->size; ++thread) {
    if(group->disabled[thread] == false) {
      if(group->args[thread].errCode) {
        return group->args[thread].errCode;
      }
    }
  }

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlThreadGroupRunVarArgs(HmlThreadGroup *group,
                         HmlThreadFunc   func,
                         void           *args,
                         size_t          argSize) {
  HML_ERR_PROLOGUE;
  uint32_t   thread;
  int32_t    pthreadErrCode;

  HML_ERR_GEN(!func, cHmlErrGeneral);
  group->func = func;
  for(thread = 0; thread < group->size; ++thread) {
    if(group->disabled[thread] == false) {
      group->args[thread].args = (char *)args + argSize * thread;
      pthreadErrCode = pthread_create(&group->pthreads[thread],
                                      &group->attr,
                                      hmlThread,
                                      &group->args[thread]);
      HML_ERR_GEN(pthreadErrCode, cHmlErrGeneral);
    }
  }
  /* join all pthreads created */
  for(thread = 0; thread < group->size; ++thread) {
    if(group->disabled[thread] == false) {
      pthreadErrCode = pthread_join(group->pthreads[thread], NULL);
      HML_ERR_GEN(pthreadErrCode, cHmlErrGeneral);
    }
  }
  /* check for error code returned by each thread */
  HML_ERR_PASS(hmlThreadGroupError(group));

  HML_NORMAL_RETURN;
}

HmlErrCode
hmlThreadGroupRun(HmlThreadGroup *group,
                  HmlThreadFunc   func,
                  void           *args) {
  return hmlThreadGroupRunVarArgs(group, func, args, 0);
}

HmlErrCode
hmlThreadGroupDelete(HmlThreadGroup *group) {
  HML_ERR_PROLOGUE;
  int32_t  pthreadErrCode;

  /* delete the attribute object */
  pthreadErrCode = pthread_attr_destroy(&group->attr);
  HML_ERR_GEN(pthreadErrCode, cHmlErrGeneral);
  FREE(group->args);
  FREE(group->pthreads);
  FREE(group->disabled);
  memset(group, 0, sizeof(HmlThreadGroup));

  HML_NORMAL_RETURN;
}
