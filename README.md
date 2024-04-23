# MALCC
This is a demonstration of MALCC, which extends the interpretation of prediction results called LIME to RGB channels for image classification.

broccoli.jpg(元画像)

![ori](images/broccoli.jpg)

You can visualize the important areas and colors when estimating ↑ as broccoli.

![important area](results/sample.png)

It seems that the G of the broccoli in the back is used as the basis for classification.

# How to use

clone
```
git clone https://github.com/tsutsu-22/MALCC
```

```
cd MALCC
```

The package can be installed with:
*If you do not have a GPU, please change tensorflow-gpu in requirements.txt to tensorflow.
```
pip install -r requirements.txt
```

Parameter and image settings → see params below
*The initial setting is MALCC using broccoli image classification and superpixel.

Run the following:
```
python main.py
```

# Where Results are
```
MALCC/results/bloccoli/merge/*.png
```

## params
You can change the following contents.
```python
  ####param####
  num_pattern=500  #Number of patterns in multiple regression analysis
  name='broccoli'　#input image name
  img=f'images/{name}.jpg'
  svdir=f'results/{name}'
  savename='test_output'
  ths=[0.05,0.03,0.01,0.005,0.001] #What percentage of the top should be saved?
  #############
```
num_pattern...The more patterns of mask images to create, the more accurate it will be, but the calculation time will increase.

img...Place the path of the input image and what you want to classify with ImageNet in the same directory and write it here.

svdir...Which directory to save the results in

savename...Resulting image name

th...What percentage of important areas and colors should be displayed?


### reference
Please refer to below for the general flow

https://ascii.jp/elem/000/004/007/4007762/

Images from ImageNet are obtained from ↓

https://starpentagon.net/analytics/ilsvrc2012_class_image/

LIME's git↓

https://github.com/marcotcr/lime/tree/73f03130b1fa8dbb3378457e78c82d4889942f83


