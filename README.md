# SyRa : Synthesized rain image for deraining algorithms
On this repository, SyRaGAN's code and instructions for synthesizing rain images are explained.

## Requirements
Install the dependencies:
```bash
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
>**Click to download [SyRa](https://drive.google.com/drive/folders/1SSLpAKuW6U2gPk6601agOMNeA5Kx5_zf?usp=sharing)**<br>
>**Click to download [SyRa-HQ](https://drive.google.com/drive/folders/1PUXDTdf0vGeZaH7sbc9xXCCTWM7ouwZ_?usp=sharing)**<br>

## Training dataset
As training data, Rain100L  [1](https://drive.google.com/drive/folders/1SSLpAKuW6U2gPk6601agOMNeA5Kx5_zf?usp=sharing), Rain100H [1](https://drive.google.com/drive/folders/1SSLpAKuW6U2gPk6601agOMNeA5Kx5_zf?usp=sharing), Rain800 [1](https://drive.google.com/drive/folders/1SSLpAKuW6U2gPk6601agOMNeA5Kx5_zf?usp=sharing), Rain1200 [1](https://drive.google.com/drive/folders/1SSLpAKuW6U2gPk6601agOMNeA5Kx5_zf?usp=sharing), Rain1400 [1](https://drive.google.com/drive/folders/1SSLpAKuW6U2gPk6601agOMNeA5Kx5_zf?usp=sharing), and SPA-data [1](https://drive.google.com/drive/folders/1SSLpAKuW6U2gPk6601agOMNeA5Kx5_zf?usp=sharing) were used.
The training image is used by concating each clear image and rain image.

