# HER - EMNLP2019
## Reading Like {HER}: Human Reading Inspired Extractive Summarization
This is a Pytorch implementation of [Reading Like {HER}: Human Reading Inspired Extractive Summarization](https://www.aclweb.org/anthology/D19-1300.pdf)". 

## 0. Enviroment
python 2.7\
pytorch 1.3\
pyrouge 0.1.3
 
## 1. Prepare Dataset 
We evaluate our models on three datasets: the CNN, the DailyMail and the com- bined CNN/DailyMail ```(Hermann et al., 2015; Nallapati et al., 2016).``` You can download dataset from [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail).

We use the bandit settings and pretrained vocab embeddings provided by ```Dong et al. (2018).``` and you can download [here](https://drive.google.com/file/d/1W0QQkz5VNCk-YAnpSRc0ONFgR5SPGDA8/view?usp=sharing). 

Put all the downloaded files under folder ``data``.

## 2. Preprocessing
```
python dataLoader.py
```

## 3. Train / Test
```
python main.py / python evaluate.py --std_rouge
```


