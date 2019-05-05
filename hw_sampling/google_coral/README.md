# Coral TPU / c++ sample app / Mobilenet SSD v2

Quick sample to test google's tpu performance with:

* **Odroid XU4 + USB Edge TPU + Logitech c920 / Ubuntu 18.04 / 4.14**

* **Coral SBC + Google's MIPI-CSI 5MP camera / Mendel (Debian) / 4.9**

### Results Odroid XU4 with USB TPU

* **odroid xu4 at idle**: 3.1 W
* **odroid xu4 + TPU + Logitech 920 WEBCAM**: 6.1 W - 24.89 Real FPS / 42.90 Inference FPS (23ms) -  load average: 0.52
* **odroid xu4 + Logitech 920 WEBCAM + app dry run (no tpu)**: 4.9 W - load average: 0.64, 0.61, 0.62
* **USB TPU power usage**: 1W - 1.2 W 

### Results Coral Dev Board

* **TPU DEV BOARD idle**: 2.7 W
* **TPU DEV BOARD + Logitech 920 WEBCAM**: 5.4 W - 24.10 Real FPS / 78.15 Inference FPS (13ms) - load average: 0.70
* **TPU DEV BOARD + MIPI 5MP CAM**: 4.5 W - 29.4 Real FPS / 78.70 Inference FPS (13ms) - load average: 0.87, 0.58, 0.26
* **TPU DEV BOARD - ETH** = - 0.6W

## How to get the sample to work 

```sh
make
# specify the id of your camera 0=default
./tpu_obj_detect --camera_device=0 
# to save last image awith nnotation to a file add --annotate, it will save obj_detect_note.jpg in current path
./tpu_obj_detect --camera_device=0 --annotate
# to use a video instead of camera try
./tpu_obj_detect --video=test.mp4 --annotate
```

## Required

* Install **opencv 4.1** - sample should work with older verions, v2 with small a modification, see makefile

* Install **tensor flow**. Note, tf static libraries for armv7 and aarch64 (coral sbc) are included in this release, see lib/libtensorflow-lite_armv7l.a, libtensorflow-lite_aarch64.a

```sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/lite
# for armv7 you can use this script to build
tools/make/build_rpi_lib.sh
# for coral dev board - quick hack
cp tools/make/build_rpi_lib.sh tools/make/build_coral_lib.sh
# modify as follow and run
CC_PREFIX=aarch64-linux-gnu- 
make -j 3 -f tensorflow/lite/tools/make/Makefile TARGET=rpi TARGET_ARCH=cortex-a57
```

