
cc_library(jit_kernel_mkl SRCS mkl.cc DEPS jit_kernel_base dynload_mklml)
set(JIT_KERNEL_DEPS ${JIT_KERNEL_DEPS} dynload_mklml jit_kernel_mkl PARENT_SCOPE)

# use mkl kernels by name and type
USE_JITKERNEL_MORE(kMatMul, mkl)
USE_JITKERNEL_MORE(kVMul, mkl)
USE_JITKERNEL_MORE(kVAdd, mkl)
USE_JITKERNEL_MORE(kVScal, mkl)
USE_JITKERNEL_MORE(kVExp, mkl)
USE_JITKERNEL_MORE(kVSquare, mkl)
USE_JITKERNEL_MORE(kVSigmoid, mkl)
USE_JITKERNEL_MORE(kVTanh, mkl)
USE_JITKERNEL_MORE(kSeqPool, mkl)
USE_JITKERNEL_MORE(kSoftmax, mkl)
USE_JITKERNEL_MORE(kEmbSeqPool, mkl)
