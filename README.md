# Yolo_tracking

This software tracks objects in videos by using an image neural detection. To train the network I use AlexeyAB YOLO repository. The software includes a GUI that allow the user to prepare the dataset to train, train the network and track the objects.

**HOW TO INSTALL IT IN UBUNTU**

Open a terminal and intall all packages necessary to compile OPENCV:
<pre><code>
sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python3-dev python-numpy python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
</pre></code>

Download OPENCV source code:
<pre><code>
cd ~/
mkdir OPENCV
cd OPENCV
wget https://github.com/opencv/opencv/archive/master.zip
unzip master.zip
rm master.zip
wget https://github.com/opencv/opencv_contrib/archive/master.zip
unzip master.zip
rm master.zip
mv opencv-master opencv
mv opencv_contrib-master opencv_contrib
cd opencv
mkdir build
cd build
</pre></code>

Compile OPENCV. We will compile it with GPU activated. However, you need a NVIDIA GPU to do this step. If you don't have an NVDIA, skip lines WITH_CUDA, WITH_CUDNN, OPENCV_DNN_CUDA, CUDA_ARCH_BIN. If you have an NVIDIA GPU, check the arch at https://developer.nvidia.com/cuda-gpus#compute and change the parameter at CUDA_ARCH_BIN with this value. You can also change the -j8 parameter in the last line to compile it with the number or processors your CPU has.

<pre><code>
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D CUDA_ARCH_BIN=7.0 \
-D WITH_CUBLAS=1 \
-D OPENCV_EXTRA_MODULES_PATH=~/OPENCV/opencv_contrib/modules \
-D HAVE_opencv_python3=ON \
-D PYTHON_EXECUTABLE=~/.virtualenvs/opencv_cuda/bin/python \
-D BUILD_EXAMPLES=ON -D
OPENCV_GENERATE_PKGCONFIG=YES ..
make -j8
sudo make install
</pre></code>

Download YOLO source code:
<pre><code>
cd ~/
mkdir YOLO
cd YOLO
wget https://github.com/AlexeyAB/darknet/archive/master.zip
unzip master.zip
rm master.zip
cd darknet-master
</pre></code>

Change parameters for compilation. You need to edit the first lines of Makefile that are inside darknet-master. These parameters depend on the hardware you have. You can use GPU, CUDNN and CUDNN_HALF only if you have an NVIDA GPU. To know if you can also use CUDNN and CUDNN_HALF check https://developer.nvidia.com/cuda-gpus#compute. If the capability is 3.0 or above you can use CUDNN. If it is 5.3 or above you can use HALF. However, OPENCV requires minimum version 5.3 to be compiled with the GPU. If you couldn't install OPENCV with the GPU activated, switch off these parameters (put 0).
<pre><code>
GPU=1
CUDNN=1
CUDNN_HALF=1
OPENCV=1
AVX=1
OPENMP=1
</pre></code>
After these lines, depending on the YOLO version there might appear some commented line (they have a # at the beggining) that start by "ARCH". If you can find your GPU architecture, remove the # before the ARCH for your architecture. Then compile YOLO:
<pre><code>
make
</pre></code>

If you recieve an error about NVCC, open the file at /home/yourusername/.bashrc and add the following lines at the end of the file. Do change XYZ by the version of your CUDA, eg.: cuda-XYZ -> cuda-11.3.
<pre><code>
export CUDA_HOME=/usr/local/cuda-XYZ
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
PATH=${CUDA_HOME}/bin:${PATH}
export PATH
</pre></code>

Now you can run the software you can download here. Just unzip and do:
<pre><code>
python3 ./run.py
</pre></code>

If you recieve an error because you miss a python library, you can install it using:

<pre><code>
sudo apt-get install python3-pip
pip3 install package-names
</pre></code>

where package-names are the libraries you need to install.

**HOW TO USE IT**

1) You need to obtain a dataset with the objects you want to detect. Click on the first button from the left to prepare this dataset. 
2) The next step is to prepare the YOLO network. You can do it by pressing at the second button in the menu. 
3) Once you have the files to train the network, click on the third button to start training the network.
4) Finally, once the network is trained you can click the fourth button to track the objects. Objects cannot overlap.
