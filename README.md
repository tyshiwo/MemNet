
# MemNet 
### [[Paper]](http://cvlab.cse.msu.edu/pdfs/Image_Restoration%20using_Persistent_Memory_Network.pdf)

### Citation
If you find MemNet useful in your research, please consider citing:

	@inproceedings{Tai-MemNet-2017,
	  title={MemNet: A Persistent Memory Network for Image Restoration},
	  author={Tai, Ying and Yang, Jian and Liu, Xiaoming and Xu, Chunyan},
	  booktitle={Proceedings of International Conference on Computer Vision},
	  year={2017}
	}
	
## Implement adjustable gradient clipping 
modify sgd_solver.cpp in your_caffe_root/src/caffe/solvers/, where we add the following codes in funciton ClipGradients():

Dtype rate = GetLearningRate();

const Dtype clip_gradients = this->param_.clip_gradients()/rate;

## Training (Taking Super-resolution task as the example)
1. Preparing training/validation data using the files: generate_trainingset_x234/generate_testingset_x234 in "data/SuperResolution" folder. "Train_291" folder contains 291 training images and "Set5" folder is a popular benchmark dataset.
2. We release two MemNet architectures: MemNet_M6R6_80C64 and MemNet_M10R10_212C64 in "caffe_files" folder. Choose either one to do training. 

    	$ cd ./caffe_files/MemNet_M6R6_80C64
    	$ ./train_MemNet_M6R6_80C64.sh

## Test (Taking Super-resolution task as the example)
1. Remember to compile the matlab wrapper: make matcaffe, since we use matlab to do testing.
2. We release two pretrained models: MemNet_M6R6_80C64 and MemNet_M10R10_212C64 in "model" folder. Choose either one to do testing on benchmark Set5. 

    	$ cd ./results/MemNet_M6R6_80C64
    	$ matlab
    	>> test_MemNet_M6R6_SR
	The results are stored in "results" folder, with both reconstructed images and PSNR/SSIMs.

## More Qualitative results
### Image denoising
![](figures/final_GD.png) 

![](figures/supp_GD.png) 

### Super-resolution
![](figures/final_SR.png) 

![](figures/supp_SR.png) 

### JPEG deblocking
![](figures/final_JD.png) 

![](figures/supp_JD.png) 


