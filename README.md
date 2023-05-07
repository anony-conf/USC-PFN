# Unified Self-cycle Consistency for Parser-free Virtual Try-on, 2023
**Official code for paper "[Unified Self-cycle Consistency for Parser-free Virtual Try-on](https://arxiv.org/abs/)"**


<!-- **The training code has been released.** -->

![image](https://github.com/geyuying/PF-AFN/blob/main/show/compare_both.jpg?raw=true)

The pursuit of an efficient lifestyle has been stimulating the development of virtual try-on technology. However, generating high-quality virtual try-on images remains challenging due to the inherent difficulties such as modeling non-rigid garment deformation and unpaired garment-person images. Recent groundbreaking formulations, including in-painting, cycle consistency, and in-painting-based knowledge distillation, have enabled self-supervised generation of try-on images. Nonetheless, these methods decouple different garment domains in the try-on result distribution through dual generators or “teacher knowledge”, and their multi-model cross-domain pipeline may act as a significant bottleneck of major generator, leading to reduced try-on quality. To tackle these limitations, we propose a new Self-Cycle Consistency virtual try-on Network (SCCN), which enables the learning of virtual try-on for different garment domains using only a single model. A self-cycle consistency architecture in a round mode is first proposed for virtual try-on, which effectively avoids third-party interference noise ($e.g.$, erroneous human segmentation and irresponsible teacher knowledge). Particularly, Markov Random Field based non-rigid flow estimation formulation is leveraged for more natural garment deformation. Moreover, SCCN can employ general generators for self-supervised cycle training. Experiments demonstrate that our method achieves SOTA performance on a popular virtual try-on benchmark.

[[Paper]](https://arxiv.org/abs/)       [[Supplementary Material]](https://github.com/geyuying/PF-AFN/blob/main/PFAFN_supp.pdf)

[[Checkpoints for Test]](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing)

[[Training_Data]](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing)
[[Test_Data]](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view?usp=sharing)

[[VGG_Model]](https://drive.google.com/file/d/1Mw24L52FfOT9xXm3I1GL8btn7vttsHd9/view?usp=sharing)

## Our Environment
- anaconda3

- pytorch 1.6.0

- torchvision 0.7.0

- cuda 11.7

- cupy 8.3.0

- opencv-python 4.5.1
 
- python 3.6

1 tesla V100 GPU for training; 1 tesla V100 GPU for test


## Installation
`conda create -n sccn python=3.6`

`source activate sccn     or     conda activate sccn`

`conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=11.7 -c pytorch`

`conda install cupy     or     pip install cupy==8.3.0`

`pip install opencv-python`

git clone https://github.com/anony-conf/SCCN.git

cd SCCN

<!-- ## Training on VITON dataset 
1. cd PF-AFN_train
2. Download the VITON training set from [VITON_train](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing) and put the folder "VITON_traindata" under the folder "dataset".
3. Dowload the VGG_19 model from [VGG_Model](https://drive.google.com/file/d/1Mw24L52FfOT9xXm3I1GL8btn7vttsHd9/view?usp=sharing) and put "vgg19-dcbb9e9d.pth" under the folder "models".
4. First train the parser-based network PBAFN. Run **scripts/train_PBAFN_stage1.sh**. After the parser-based warping module is trained, run **scripts/train_PBAFN_e2e.sh**.
5. After training the parser-based network PBAFN, train the parser-free network PFAFN. Run **scripts/train_PFAFN_stage1.sh**. After the parser-free warping module is trained, run **scripts/train_PFAFN_e2e.sh**.
6. Following the above insructions with the provided training code, the [[trained PF-AFN]](https://drive.google.com/file/d/1Pz2kA65N4Ih9w6NFYBDmdtVdB-nrrdc3/view?usp=sharing) achieves FID 9.92 on VITON test set with the test_pairs.txt (You can find it in https://github.com/minar09/cp-vton-plus/blob/master/data/test_pairs.txt). -->

## Run the demo
1. cd PF-AFN_test
2. First, you need to download the checkpoints from [checkpoints](https://drive.google.com/file/d/1_a0AiN8Y_d_9TNDhHIcRlERz3zptyYWV/view?usp=sharing) and put the folder "PFAFN" under the folder "checkpoints". The folder "checkpoints/PFAFN" shold contain "warp_model_final.pth" and "gen_model_final.pth". 
3. The "dataset" folder contains the demo images for test, where the "test_img" folder contains the person images, the "test_clothes" folder contains the clothes images, and the "test_edge" folder contains edges extracted from the clothes images with the built-in function in python (We saved the extracted edges from the clothes images for convenience). 'demo.txt' records the test pairs. 
4. During test, a person image, a clothes image and its extracted edge are fed into the network to generate the try-on image. **No human parsing results or human pose estimation results are needed for test.**
5. To test with the saved model, run **test.sh** and the results will be saved in the folder "results".
6. **To reproduce our results from the saved model, your test environment should be the same as our test environment, especifically for the version of cupy.** 

![image](https://github.com/geyuying/PF-AFN/blob/main/show/compare.jpg?raw=true)
## Dataset
1. [VITON](https://github.com/xthan/VITON) contains a training set of 14,221 image pairs and a test set of 2,032 image pairs, each of which has a front-view woman photo and a top clothing image with the resolution 256 x 192. Our saved model is trained on the VITON training set and tested on the VITON test set.
2. To train from scratch on VITON training set, you can download [VITON_train](https://drive.google.com/file/d/1Uc0DTTkSfCPXDhd4CMx2TQlzlC6bDolK/view?usp=sharing).
3. To test our saved model on the complete VITON test set, you can download [VITON_test](https://drive.google.com/file/d/1Y7uV0gomwWyxCvvH8TIbY7D9cTAUy6om/view?usp=sharing).

## License
The use of this code is RESTRICTED to non-commercial research and educational purposes.

## Acknowledgement
Our code is based on the implementation of "Clothflow: A flow-based model for clothed person generation" (See the citation below), including the implementation of the feature pyramid networks (FPN) and the ResUnetGenerator, and the adaptation of the cascaded structure to predict the appearance flows. If you use our code, please also cite their work as below.

https://github.com/levindabhi/SieveNet
## Citation
If our code is helpful to your work, please cite:
```
@article{du2023unified,
  title={Unified Self-cycle Consistency for Parser-free Virtual Try-on},
  author={Du, Chenghu and Wang, Junyin and Liu, Shuqin and Xiong, Shengwu},
  journal={arXiv preprint arXiv:2303.00000},
  year={2023}
}
```

