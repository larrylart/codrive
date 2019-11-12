# VIM3 AML NPU / c++ example app / Mobilenet SSD v2

### Quick sample to test NPU performance with:

* **Khadas VIM3 / NPU + Logitech c920 / Ubuntu 18.04 / 4.9.190

### Results 

* **VIM3 at idle**: 2.1 W (used to be 1.4 W with a previous kernel version)
* **VIM3 + NPU + Logitech 920 WEBCAM**: 4.1 W - 29.86 Real FPS / 72.20 Inference FPS (14ms) -  load average: 0.49

## How to get the sample to work 

```sh
# Edit Makefile and change AML_SDK_PATH= to your AML/SDK instalation path
make
# display video window
./tpu_obj_detect --gui=true
# specify the id of your camera 0=default
./tpu_obj_detect --camera_device=0 
# to save last image awith nnotation to a file add --annotate, it will save obj_detect_note.jpg in current path
./tpu_obj_detect --camera_device=0 --annotate
# to use a video instead of camera try
./tpu_obj_detect --video=test.mp4 --annotate
```

## Required

* Install **opencv 4.1** - sample should work with older verions, v2 with small a modification, see makefile. I will also include the opencv world lib(libopencv_world.so.4.1) in the lib folder you can try to copy that, say in /url/lib instead.

```sh
shell> apt-get install v4l-utils v4l2ucp libv4l-0 libv4l-dev libomxil-bellagio0-components-videosrc uvcdynctrl uvcdynctrl-data v4l-utils libv4l-dev libv4l-0
shell> apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
shell> apt-get install libavcodec-dev libavcodec-extra57 libavcodec-extra libavdevice-dev ffmpeg libavformat-dev libavresample-dev libavutil-dev libxine2-ffmpeg libavfilter-dev
shell> apt-get install libxine2-dev
shell> wget https://github.com/opencv/opencv/archive/4.1.0.zip -O opencv_4.1.0.zip
shell> unzip opencv_4.1.0.zip
shell> wget https://github.com/opencv/opencv_contrib/archive/4.1.0.tar.gz -O  opencv_contrib_4.1.0.tar.gz
shell> tar -xzvf opencv_contrib_4.1.0.tar.gz
shell> git clone https://github.com/opencv/opencv_extra.git
shell> cd opencv-4.1.0/
shell> mkdir build
shell> cd build
shell> cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local  -D OPENCV_GENERATE_PKGCONFIG=YES -D WITH_V4L=ON -DBUILD_opencv_world=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=OFF -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.1.0/modules -D WITH_OPENCL=ON -D WITH_GTK=ON -D WITH_TBB=ON  -D WITH_OPENGL=ON -D WITH_OPENMP=ON -D BUILD_EXAMPLES=ON -D ENABLE_NEON=ON -D OPENCV_ENABLE_NONFREE=ON -D WITH_FFMPEG=ON -D WITH_GTK_2_X=ON -D WITH_LIBV4L=ON -D WITH_XINE=ON -D WITH_GSTREAMER=ON -D WITH_GDAL=ON -D WITH_HALIDE=ON  -D ENABLE_PRECOMPILED_HEADERS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D ENABLE_CXX11=ON -DEXTRA_C_FLAGS="-mcpu=cortex-a73 -ftree-vectorize -mfloat-abi=hard" -DEXTRA_CXX_FLAGS="-mcpu=cortex-a73 -ftree-vectorize -mfloat-abi=hard" -DWITH_JPEG=ON ..
shell> make -j5
shell> make install
```

* Install AML/SDK (ask Khadas team) or you can try using minimal build headers and libraries (see folders include, lib) 

* To run in GUI mode you might get an error/warning. To go around that you might want to do the following:

```sh
apt-get install libatk-adaptor libgail-common
# add to .bashrc then run: source .bashrc
export NO_AT_BRIDGE=1
 ```

# How to convert custom/other models with tensorflow/acuity

## Exporting tensorflow models for AML NPU. You will need Ubuntu 18.04(16.04) x86 with a newer CPU (after 2011)

* Download model(model.ckpt.data-xxxxx,model.ckpt.index,model.ckpt.meta,pipeline.config) and extract in a folder 

* Export to tensforflow lite:

```sh
python3 /usr/local/lib/python3.6/dist-packages/object_detection-0.1-py3.6.egg/object_detection/export_tflite_ssd_graph.py --pipeline_config_path pipeline.config --trained_checkpoint_prefix model.ckpt --output_directory out --add_postprocessing_op=false
``` 

* Visualize exported graph:
```sh
python3 /usr/local/lib/python3.6/dist-packages/tensorflow/python/tools/import_pb_to_tensorboard.py --model_dir out/tflite_graph.pb --log_dir /tmp/tensorboard
tensorboard --logdir=/tmp/tensorboard
``` 

* If you want to summarize_graph to see input/output nodes, but for that you need to install tensorflow source.

```sh
apt-get install git
wget https://github.com/bazelbuild/bazel/releases/download/0.29.1/bazel-0.29.1-installer-linux-x86_64.sh
chmod +x bazel-0.29.1-installer-linux-x86_64.sh
./bazel-0.29.1-installer-linux-x86_64.sh
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/
git checkout r1.10
./configure
bazel build //tensorflow/tools/pip_package:build_pip_package
bazel build tensorflow/tools/graph_transforms:summarize_graph
# now use the tool
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=frozen_inference_graph.pb
``` 

* Export to inference frozen graph, however I didn't manage to import this in Acuity.

```sh
python3 /usr/local/lib/python3.6/dist-packages/object_detection-0.1-py3.6.egg/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path pipeline.config --trained_checkpoint_prefix model.ckpt --output_directory out
``` 

## Importing/graph with Acuity. You will need the same x86 machine as above

* Go in the Acuity tool folder: aml_npu_sdk/acuity-toolkit/conversion_scripts and copy the tflite_graph_v2.pb(in this case) from previous exporting.
Note that for mobilenet SSD I had to use concat_1 instead raw_outputs/class_predictions since I got an error for op RealDiv not supported and I implemented instead the last part in the code.
This is the command from 0_import_model.sh which I show in clear so you can see the parameters.

```sh
../bin/convertensorflow --tf-pb ./model/tflite_graph_v2.pb --inputs normalized_input_image_tensor --input-size-list '300,300,3' --outputs 'raw_outputs/box_encodings concat_1' --net-output mobilenet_ssd_v2.json --data-output mobilenet_ssd_v2.data
``` 

* Edit and modify the NAME at the top in 1_quantize_model.sh and 2_export_case_code.sh, as in this case replace it with mobilenet_ssd_v2;

* If all good you should have the converted graph mobilenet_ssd_v2.nb along with a template source code folder.

* Note: For mobilenet SSD I needed to export the anchors, to helpposition the detection boxes. For that I used a script (see in scripts folder in this repository) and save in a .h file to include in the source code.

```sh
python3 ./export_anchors.py model/tflite_graph_v2.pb > nn_anchors.h
``` 



 