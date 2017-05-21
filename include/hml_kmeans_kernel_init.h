/*****************************************************************
 *  Copyright (c) 2017. Palo Alto Research Center                *
 *  All rights reserved.                                         *
 *****************************************************************/

#ifndef HML_KMEANS_KERNEL_INIT_H_INCLUDED_
#define HML_KMEANS_KERNEL_INIT_H_INCLUDED_

#include "hml_kmeans_kernel.h"

/* init kernels that do not use texture memory */
void hmlKmeansInitKernelAssignGmem(HmlKmeansAssignKernel *kernel);

void hmlKmeansInitKernelAssignSmem(HmlKmeansAssignKernel *kernel);

void hmlKmeansInitKernelAssignSmemUnroll(HmlKmeansAssignKernel *kernel);

void hmlKmeansInitKernelUpdateGmem(HmlKmeansUpdateKernel *kernel);

void hmlKmeansInitKernelUpdateSmem(HmlKmeansUpdateKernel *kernel);

/* init kernels that use texture memory */
void hmlKmeansInitKernelAssignGmemTex(HmlKmeansAssignKernel *kernel);

void hmlKmeansInitKernelAssignSmemTex(HmlKmeansAssignKernel *kernel);

void hmlKmeansInitKernelAssignSmemUnrollTex(HmlKmeansAssignKernel *kernel);

void hmlKmeansInitKernelUpdateGmemTex(HmlKmeansUpdateKernel *kernel);

void hmlKmeansInitKernelUpdateSmemTex(HmlKmeansUpdateKernel *kernel);

#endif /* HML_KMEANS_KERNEL_INIT_H_INCLUDED_ */
