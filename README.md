# Pytorch Implementation - SyRaGAN

>**Update completion**<br>

**SyRa : Synthesized rain image for deraining algorithms**<br>

[Jaewoong Choi](https://github.com/jaewoong1),  [Daeha Kim](https://github.com/kdhht2334),  [Sanghyuk Lee](https://github.com/shlee625),  [Byung Cheol Song](https://scholar.google.com/citations?user=yo-cOtMAAAAJ&hl=ko&oi=sra)

On this repository, SyRaGAN's code and instructions for synthesizing rain images are explained.

![figure1](https://user-images.githubusercontent.com/54341727/130918704-2e9eeb97-442b-404e-8393-c439795b2597.png)

## Requirements
Install the dependencies:
```
bash
conda create -n SyRaGAN python=3.6.7
conda activate SyRaGAN
conda install -y pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0
pip install tqdm
```


## Pretrained model
>**Click to download [pretrained SyRa-GAN](https://drive.google.com/file/d/1TGqwSroSOsS77J2rQVGXfCpT6jcI5fuQ/view?usp=sharing)**<br>


## SyRa dataset
>**Click to download [SyRa](https://drive.google.com/drive/folders/1SSLpAKuW6U2gPk6601agOMNeA5Kx5_zf?usp=sharing)**<br> : Trainset - 10K clear image and 50K synthesized rain image , Testset - 1K clear image and 5K synthesized rain image

>**Click to download [SyRa-HQ](https://drive.google.com/drive/folders/1PUXDTdf0vGeZaH7sbc9xXCCTWM7ouwZ_?usp=sharing)**<br> : Trainset - 1K clear image and 5K synthesized rain image , Testset - 100 clear image and 500 synthesized rain image

## Training dataset
As training data, Rain100L [1], Rain100H [1], Rain800 [2], Rain1200 [3], Rain1400 [4], and SPA-data [5] were used.
The training image is used by concating each clear image and rain image.

## Training SyRaGAN
Divide your training images into the following locations : `./data/rains/train/A` `./data/rains/train/B`

**Example of training image :**<br>

![raind4697](https://user-images.githubusercontent.com/54341727/130947400-d35d3fe4-7903-4786-b232-22d6d270946d.jpg)

**Run**<br>

```
python main.py --img_size 256 --mode train --checkpoint_dir expr/checkpopint/SyRa --resume_iter 0 --gpu 0
```

## Synthesizing rain image
Put clear images in the following location. `./asset/folder_of_your_data/folder_of_your_data`

Put checkpoint file in the following location. `./expr/checkpoint/SyRa`

**Run**<br>

```
python main.py --img_size 256 --mode syn --checkpoint_dir expr/checkpoint/SyRa --out_dir expr/result --data folder_of_your_data --resume_iter 100000
```

5 syntheiszed rain images will be created for each clear image in `./expr/result`

## References
[1] Yang, Wenhan, et al. "Deep joint rain detection and removal from a single image." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[2] Zhang, He, Vishwanath Sindagi, and Vishal M. Patel. "Image de-raining using a conditional generative adversarial network." IEEE transactions on circuits and systems for video technology 30.11 (2019): 3943-3956.

[3] Zhang, He, and Vishal M. Patel. "Density-aware single image de-raining using a multi-stream dense network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[4] Fu, Xueyang, et al. "Removing rain from single images via a deep detail network." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.

[5] Wang, Tianyu, et al. "Spatial attentive single-image deraining with a high quality real rain dataset." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
