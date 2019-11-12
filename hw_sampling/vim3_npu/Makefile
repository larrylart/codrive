
# change to your AML SDK path and also remove comments/comment sample includes bellow, see INCFLAGS
AML_SDK_PATH=/root/kits/NPU/aml_npu_sdk
MACHINE=$(shell uname -m)

CC=arm-linux-gnueabihf-gcc
CXX=arm-linux-gnueabihf-g++
CCO=arm-linux-gnueabihf-gcc -o $@
CXXO=arm-linux-gnueabihf-g++ -o $@

CPPFLAGS= -O3 --std=c++11 -march=armv8-a -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize -fPIC 
AMLCFLAGS= -DLINUX -Wall -D_REENTRANT -fno-strict-aliasing -mtune=cortex-a53 -march=armv8-a -O2 -DgcdDEBUG=0 \
-DgcdENABLE_3D=1 -DgcdENABLE_2D=0 -DgcdENABLE_VG=0 -DgcdUSE_VX=1 -DUSE_VDK=1 -DgcdMOVG=0 -DEGL_API_FB \
-DgcdSTATIC_LINK=0 -DgcdFPGA_BUILD=0 -DGC_ENABLE_LOADTIME_OPT=1 -DgcdUSE_VXC_BINARY=0 -DgcdGC355_MEM_PRINT=0 \
-DgcdGC355_PROFILER=0 -DVIVANTE_PROFILER=1 -DVIVANTE_PROFILER_CONTEXT=1 -DDEBUG=0

LDFLAGS=-mtune=cortex-a53 -march=armv8-a -Wl,-rpath-link $(AML_SDK_PATH)/linux_sdk/linux_sdk/build/sdk/drivers  
LDLIBS=-lpthread -lm -ldl -lrt -lOpenVX -lOpenVXU -lCLC -lVSC -lGAL -lovxlib 
#LDLIBS += \
#-L$(AML_SDK_PATH)/linux_sdk/linux_sdk/build/sdk/drivers 
#-L$(AML_SDK_PATH)/linux_sdk/linux_sdk/acuity-ovxlib-dev/lib 
#-L/root/code/fenix/build/toolchains/gcc-linaro-aarch64-linux-gnu/aarch64-linux-gnu/libc/lib 

LIBSTATIC=$(AML_SDK_PATH)/linux_sdk/linux_sdk/acuity-ovxlib-dev/lib/libjpeg.a 


# handle cortex-a57 - coral dev board
ifeq ($(MACHINE),aarch64)
	# add -D_CV_OLD_ for opencv 2
	CPPFLAGS = -O3 --std=c++11 -mcpu=cortex-a73 -funsafe-math-optimizations -ftree-vectorize -fPIC 
	CC=aarch64-linux-gnu-gcc
	CXX=aarch64-linux-gnu-g++
	CCO=aarch64-linux-gnu-gcc -o $@
	CXXO=aarch64-linux-gnu-g++ -o $@
endif

INCFLAGS = \
-I. \
-I./include/ \
-I./include/aml_sdk/ \
-I./include/aml_sdk/HAL \
-I./include/ovxlib/ 

# use this with the SDK instead of the above 3
#-I$(AML_SDK_PATH)/linux_sdk/linux_sdk/build/sdk/include \
#-I$(AML_SDK_PATH)/linux_sdk/linux_sdk/build/sdk/include/HAL \
#-I$(AML_SDK_PATH)/linux_sdk/linux_sdk/acuity-ovxlib-dev/include

# opencv - modify for older opencv versions
#OPENCV_LIBS = $(shell pkg-config opencv --libs)
#OPENCV_CXXFLAGS = $(shell pkg-config opencv --cflags)
# opencv4
OPENCV_LIBS = $(shell pkg-config opencv4 --libs)
OPENCV_CXXFLAGS = $(shell pkg-config opencv4 --cflags)

aml_obj_detect: obj/aml_obj_detect.o obj/aml_worker.o obj/vnn_mobilenetssdv2a.o 
	$(CXX) $(LDFLAGS) -o aml_obj_detect obj/aml_obj_detect.o obj/aml_worker.o obj/vnn_mobilenetssdv2a.o $(LDLIBS) $(OPENCV_LIBS) 

obj/aml_obj_detect.o: aml_obj_detect.cpp 
	$(CXXO) $(CPPFLAGS) $(INCFLAGS) $(OPENCV_CXXFLAGS) -c aml_obj_detect.cpp 

obj/aml_worker.o: aml_worker.cpp 
	$(CXXO) $(CPPFLAGS) $(AMLCFLAGS) $(INCFLAGS) $(OPENCV_CXXFLAGS) -c aml_worker.cpp 

obj/vnn_mobilenetssdv2a.o: vnn_mobilenetssdv2a.c 
	$(CCO) $(AMLCFLAGS) $(INCFLAGS) -c vnn_mobilenetssdv2a.c 

clean:
# -f so this will succeed even if the files don't exist
	rm -f aml_obj_detect *.o obj/*.o
