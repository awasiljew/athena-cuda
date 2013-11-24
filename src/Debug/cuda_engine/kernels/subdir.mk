################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../cuda_engine/kernels/integrate_2d_cuda_kernel.cu 

CU_DEPS += \
./cuda_engine/kernels/integrate_2d_cuda_kernel.d 

OBJS += \
./cuda_engine/kernels/integrate_2d_cuda_kernel.o 


# Each subdirectory must supply rules for building sources it contributes
cuda_engine/kernels/%.o: ../cuda_engine/kernels/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I/usr/local/cuda/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


