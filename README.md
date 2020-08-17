# TensorRT Inference Engine

credits: CIFAR10 reader copied from https://github.com/prabhuomkar/pytorch-cpp/blob/master/tutorials/intermediate/deep_residual_network/src/main.cpp
#### NOTE: Please download and unzip the source-code of the libtorch library in lib folder before compiling
The following are the steps to build this project  
All the commands below are to be executed in directory`inference/build-dir`

## Build
open `inference/build-dir` in terminal and type the following commands

```bash
 cmake ..
```
```bash
 make
```
You will get the following output
```bash
[ 33%] Building CXX object CMakeFiles/archit_inf.dir/src/main.cpp.o
[ 66%] Building CXX object CMakeFiles/archit_inf.dir/src/cifar10.cpp.o
[100%] Linking CXX executable archit_inf
[100%] Built target archit_inf
```
The above steps will build the executable `archit_inf`. 
(archit is my nickname :- ) )
## Usage

To do the latency test type

```bash
./archit_inf batch_size # where batch_size can be between 1 and 512

```
for example
```bash
./archit_inf 32 #for batch size of 32

```
#### NOTE:
If no argument is given batch size is taken as 1  
Accuracy is computed and printed only when batch size is 1

## Contributing
Madhav Wagle  
gmail: architwagle@gmail.com  
TSEC,
Mumbai University
