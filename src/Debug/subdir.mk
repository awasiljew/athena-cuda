################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../__integrate_2d__.o \
../ath_array.o \
../ath_files.o \
../ath_log.o \
../cc_pos.o \
../convert_var.o \
../debug_tools_cuda.o \
../dump_binary.o \
../esystem_prim.o \
../esystem_roe.o \
../field_loop.o \
../flux_hlle.o \
../flux_roe.o \
../init_grid.o \
../integrate_2d_cuda.o \
../lr_states_plm.o \
../main.o \
../new_dt.o \
../new_dt_cuda.o \
../output.o \
../output_ppm.o \
../par.o \
../set_bvals.o \
../set_bvals_cuda.o \
../show_config.o \
../utils.o \
../utils_cu.o 

C_SRCS += \
../__integrate_2d__.c \
../ath_array.c \
../ath_files.c \
../ath_log.c \
../blast.c \
../cc_pos.c \
../convert_var.c \
../cpaw2d.c \
../debug_tools_cuda.c \
../dump_binary.c \
../dump_tab.c \
../esystem_prim.c \
../esystem_roe.c \
../field_loop.c \
../flux_hlle.c \
../flux_roe.c \
../init_grid.c \
../lr_states_plm.c \
../main.c \
../new_dt.c \
../output.c \
../output_ppm.c \
../output_tab.c \
../par.c \
../set_bvals.c \
../show_config.c \
../utils.c \
../utils_cu.c 

OBJS += \
./__integrate_2d__.o \
./ath_array.o \
./ath_files.o \
./ath_log.o \
./blast.o \
./cc_pos.o \
./convert_var.o \
./cpaw2d.o \
./debug_tools_cuda.o \
./dump_binary.o \
./dump_tab.o \
./esystem_prim.o \
./esystem_roe.o \
./field_loop.o \
./flux_hlle.o \
./flux_roe.o \
./init_grid.o \
./lr_states_plm.o \
./main.o \
./new_dt.o \
./output.o \
./output_ppm.o \
./output_tab.o \
./par.o \
./set_bvals.o \
./show_config.o \
./utils.o \
./utils_cu.o 

C_DEPS += \
./__integrate_2d__.d \
./ath_array.d \
./ath_files.d \
./ath_log.d \
./blast.d \
./cc_pos.d \
./convert_var.d \
./cpaw2d.d \
./debug_tools_cuda.d \
./dump_binary.d \
./dump_tab.d \
./esystem_prim.d \
./esystem_roe.d \
./field_loop.d \
./flux_hlle.d \
./flux_roe.d \
./init_grid.d \
./lr_states_plm.d \
./main.d \
./new_dt.d \
./output.d \
./output_ppm.d \
./output_tab.d \
./par.d \
./set_bvals.d \
./show_config.d \
./utils.d \
./utils_cu.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -mpc64 -I$(CUDA_INCLUDE) -O0 -c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


