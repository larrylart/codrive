
TF_PATH=/root/kits/tensorflow/
TF_LIB_STATIC=./lib/libtensorflow-lite_armv7l.a
MACHINE=$(shell uname -m)

CC=arm-linux-gnueabihf-gcc
CXX=arm-linux-gnueabihf-g++
CXXO=arm-linux-gnueabihf-g++ -o $@

CPPFLAGS= -O3 --std=c++11 -march=armv7-a -mfpu=neon-vfpv4 -funsafe-math-optimizations -ftree-vectorize -fPIC
LDFLAGS=
LDLIBS=-lpthread -lm -ldl -ledgetpu

# handle cortex-a57 - coral dev board
ifeq ($(MACHINE),aarch64)
	TF_LIB_STATIC = ./lib/libtensorflow-lite_aarch64.a
	# add -D_CV_OLD_ for opencv 2 (apt version on coral's dev board)
	CPPFLAGS = -O3 --std=c++11 -mcpu=cortex-a57 -funsafe-math-optimizations -ftree-vectorize -fPIC 
	CC=aarch64-linux-gnu-gcc
	CXX=aarch64-linux-gnu-g++
	CXXO=aarch64-linux-gnu-g++ -o $@
endif

CPPFLAGS += \
-I. \
-I${TF_PATH} \
-I./lib/ \
-I$(TF_PATH)/tensorflow/lite/tools/make/downloads/flatbuffers/include/

# opencv
#OPENCV_LIBS = $(shell pkg-config opencv --libs)
#OPENCV_CXXFLAGS = $(shell pkg-config opencv --cflags)
# opencv4
OPENCV_LIBS = $(shell pkg-config opencv4 --libs)
OPENCV_CXXFLAGS = $(shell pkg-config opencv4 --cflags)

tpu_obj_detect: obj/tpu_obj_detect.o obj/tpu_worker.o 
	$(CXX) $(LDFLAGS) -o tpu_obj_detect obj/tpu_obj_detect.o obj/tpu_worker.o $(TF_LIB_STATIC) $(LDLIBS) $(OPENCV_LIBS) 

obj/tpu_obj_detect.o: tpu_obj_detect.cpp 
	$(CXXO) $(CPPFLAGS) $(OPENCV_CXXFLAGS) -c tpu_obj_detect.cpp 

obj/tpu_worker.o: tpu_worker.cpp 
	$(CXXO) $(CPPFLAGS) $(OPENCV_CXXFLAGS) -c tpu_worker.cpp 

clean:
# -f so this will succeed even if the files don't exist
	rm -f tpu_obj_detect *.o obj/*.o
