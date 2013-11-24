################################################################################
# Automatically-generated file. Do not edit!
################################################################################

CUDA_LIB = "/home/adam/cuda-5.0/lib64"
CUDA_INCLUDE = "/home/adam/cuda-5.0/include" 
CUDA_NVCC = "/home/adam/cuda-5.0/bin/nvcc"

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include cuda_engine/subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: athena_cuda

# Tool invocations
athena_cuda: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C Linker'
	gcc -L$(CUDA_LIB) -o "athena_cuda" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C_DEPS)$(EXECUTABLES) athena_cuda
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
