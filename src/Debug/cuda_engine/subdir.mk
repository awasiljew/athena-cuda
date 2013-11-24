################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
OBJS += \
./cuda_engine/integrate_2d_cuda.o \
./cuda_engine/new_dt_cuda.o \
./cuda_engine/set_bvals_cuda.o 


# Each subdirectory must supply rules for building sources it contributes
cuda_engine/integrate_2d_cuda.o: ../cuda_engine/integrate_2d_cuda.cu
	@echo 'Building file: $<'
	@echo 'Invoking: Resource Custom Build Step'
	$(CUDA_NVCC) -c -arch sm_13 "../cuda_engine/integrate_2d_cuda.cu" -o "cuda_engine/integrate_2d_cuda.o"
	@echo 'Finished building: $<'
	@echo ' '

cuda_engine/new_dt_cuda.o: ../cuda_engine/new_dt_cuda.cu
	@echo 'Building file: $<'
	@echo 'Invoking: Resource Custom Build Step'
	$(CUDA_NVCC) -c -arch sm_13 "../cuda_engine/new_dt_cuda.cu" -o "cuda_engine/new_dt_cuda.o"
	@echo 'Finished building: $<'
	@echo ' '

cuda_engine/set_bvals_cuda.o: ../cuda_engine/set_bvals_cuda.cu
	@echo 'Building file: $<'
	@echo 'Invoking: Resource Custom Build Step'
	$(CUDA_NVCC) -c -arch sm_13 "../cuda_engine/set_bvals_cuda.cu" -o "cuda_engine/set_bvals_cuda.o"
	@echo 'Finished building: $<'
	@echo ' '


