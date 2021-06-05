# Multiscale image contrast amplification (MUSICA)

**MUSICA**[[1]](#1) is a contrast enhancement approach based on multiresolution representation of the original image, commonly applied in computed radiography.
This repo is a Python implementation of MUSICA algorithm using Laplacian Pyramid[[2]](#2) as the multiresolution representation.

**Note:** Implementation works only for grayscale images.
## Instructions:
- Required functions are in *musica.py*
```console
$ git clone https://github.com/lafith/MUSICA.git
$ cd MUSICA
$ pip install -r requirements.txt
$ python3 test.py
```
- In *test.py* arbitrary values are taken for parameters, tuning may give better results.
## Result:
- Sample image is taken from [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/images/a7e0a141.jpg)
![test_result](https://user-images.githubusercontent.com/39316548/120895618-70a28000-c63b-11eb-87fb-04c8b21aac5b.png)

## References
<a id="1">[1]</a> 
[Vuylsteke, Pieter, and Emile P. Schoeters. "Multiscale image contrast amplification (MUSICA)." Medical Imaging 1994: Image Processing. Vol. 2167. International Society for Optics and Photonics, 1994.](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/2167/0000/Multiscale-image-contrast-amplification-MUSICA/10.1117/12.175090.short)
<br>
<a id="2">[2]</a> 
[Burt, Peter J., and Edward H. Adelson. "The Laplacian pyramid as a compact image code." Readings in computer vision. Morgan Kaufmann, 1987. 671-679.](https://www.sciencedirect.com/science/article/pii/B9780080515816500659)
