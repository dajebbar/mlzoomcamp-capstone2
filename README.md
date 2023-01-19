# ML Zoomcamp Capstone 2: IMDb movies sentiment analysis

Using sentiment analysis to classify documents based on their polarity. In particular, this project works with a dataset of 50,000 movie reviews from the Internet Movie Database (IMDb) and build a predictor that can distinguish between positive and negative review.

 # Dataset

This [project](https://ai.stanford.edu/~amaas/data/sentiment/) uses a dataset with more than **`50,000`** reviews provided by Maas and others.

# Overview
This dataset contains movie reviews along with their associated binary sentiment polarity labels. It is intended to serve as a benchmark for sentiment classification. This document outlines how the dataset was gathered, and how to use the files provided.

### Files
* `imdbEr.txt`
* `imdb.vocab`
* `README`
* `train`
* `test`

# Contents of the folder 
![tree](./img/tree.png)  
# Exploratory Data Analysis
See the [Kitchenware_EDA.ipynb](./notebooks/Kitchenware_EDA.ipynb) for this task.

### Image sizes statistics
![sizes](./figures/sizes.png)  
The images have various and varied sizes, ranging from 39 Kilopixels to 976 Kilopixels, also more than half of the images are concentrated at 750 Kpixels. This is bad news since the tensors must imperatively have the same size.

### The width to height ratio (lx/ly)
![lxy_ratio](./figures/ratio_lx_ly.png)  
Most images are vertical.

### By width (lx)
![width](./figures/width.png)  
The width varies from 233 to 1000, and it is concentrated on 1000.
### By height (ly)
![heigth](./figures/ly.png)  
The height varies from 174 to 1000, and it is concentrated on 750.
### Labels statistics
![labels](./figures/labels.png)  
**Not** all labels are representend equaly.

# Models
See the [kaggle_zoomcamp_competition_1.ipynb](./notebooks/kaggle_zoomcamp_competition_1.ipynb)  and [kaggle_zoomcamp_competition_2.ipynb](./notebooks/kaggle_zoomcamp_competition_2.ipynb) for this task.
## Summary of the models ✖️: without_ | ✔️: with_
| Model | Data Augmentation | Transfer Learning | Epochs | Losss | Accuracy % |
|:---|:---|:---|:---|:---|:---|
|model 1|✖️|✖️|25|0.8718|67.0266|
|model 2|✖️|✖️|10|1.0553|66.7387|
|model 3|✖️|✖️|10|0.7926|71.7063|
|model 4|✔️|✖️|25|0.7995|68.8265|
|model 5|✔️|vgg16 ✔️|25|0.7966|92.1526|
|model 6|✔️|vgg 16 ✔️|10|✖️|✖️|
|model 7|✔️|vgg 16 ✔️|10|0.4065|91.3607|
|model 8|✔️|efficientnet0 ✔️|10|0.4896|91.2887|
|model 9|✔️|efficientnet0 ✔️|10|0.4793|90.9287|
|model 10|✔️|efficientnetB7 ✔️|10|0.2978|92.0806|
|model 10 bis|✔️|efficientnetB7 ✔️|10|0.2990|93.5925|
|model 11|✔️|efficientnetB7 ✔️|20|0.2830|93.9525|
|model 12|✔️|efficientnetB7 ✔️|25|0.2534|93.6645|
|model 13|✔️|resnet50 ✔️|10|1.5776|39.7408|

I tried a multitude of models first, without data augmentation or transfer learning, it had a bad impact and I couldn't exceed an accuracy of 72%. Then, thanks to data augmentation and transfer learning, I was able to increase accuracy. the best accuracy was obtained with model 12 with a lost function which dropped to **0.2534** thanks to transfer learning via the base model `EfficientNetB7` as shown in the graph below.
![best-model](./figures/model12.png)  
We see that the test loss continues to drop, while the test accuracy continues to increase. I stopped at 25 epochs but I think the model can give better results by increasing the number of epochs, especially since the overfitting is minimal.  
**Note**:
The `ResNet50` was the worst model.
# Deployment of model
 I am using Streamlit on linux ubuntu, in order to deploy the model. To deploy this model with Sreamlit, please use:
  ```console
  pipenv run streamlit run predict.py
  ```

# Virtual Environment/venv 

I used pipenv for the virtual environment. In order to use the same venv as me, do use: 
```console 
pip install pipenv
```
To replicate the environment, on your command line, use 
```console
pipenv install tensorflow streamlit efficientnet
```  
**Note**: I don't have a GPU installed on my laptop for this I only installed tensorflow without GPU configuration. If you have a GPU, try to configure tensorflow with NVIDIA, CUDA and cuDNN.

# Docker

**Note**:  
To perform the following steps you should logon to your DockerHub Account ( `Login & Password`)

I have built the model and pushed it to [dajebbar/kitchenware-model:v.1.0](https://hub.docker.com/r/dajebbar/kitchenware-model). 
To use it just 
```console
docker pull dajebbar/kitchenware-model:v.1.0
```

Or in order to take the model from the docker container I built, just replace 
```Dockerfile
FROM python:3.9-slim 

#with 

FROM dajebbar/kitchenware-model:v.1.0 
```  
in the dockerfile.


If you choose to build a docker file locally instead, here are the steps to do so:
1. Create a Dockerfile as such:
```Dockerfile
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=TRUE

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system && rm -rf /root/.cache

COPY ["predict.py", "kitchenwareModel.h5", "banner.png", "./"]

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD ["predict.py"]
```

This allows us to `install python, run pipenv and its dependencies, run our predict script and our model itself and deploys our model using streamlit`.

Similarly, you can just use the dockerfile in this repository.

2. Build the Docker Container with :
```console
 docker build -t kitchenware-model .
 ```

3. Run the Docker container with:

```console
Docker run --rm --name kitchenware -p 8501:8501 kitchenware-model
```

4. tag the docker container with:

```console

docker tag kitchenware-model dajebbar/kitchenware-model:v.1.0

```
5. Push it to Docker registry with :

```console
docker push dajebbar/kitchenware-model:v.1.0

```
# Test the project
Here are some screen shots showing the interaction with the application, which represents the final product.
![cmd](./figures/cmd.png)
![cup](./figures/cup.png)
![fork](./figures/fork.png)
![plate](./figures/plate.png)  

A quick demo of app can be found [here](https://youtu.be/yvnH_PTsQFM).
# Want to Contribute?
* Fork 🍴 the repository and send PRs.
* Do ⭐ this repository if you like the content.


**Connect with me:**

<p align="center">
  <a href="https://ma.linkedin.com/in/abdeljebbar-boubekri-656b30192" target="blank"><img align="center" src="https://www.vectorlogo.zone/logos/linkedin/linkedin-tile.svg" alt="https://ma.linkedin.com/in/abdeljebbar-boubekri-656b30192" height="30" width="30" /></a>
  <a href="https://www.twitter.com/marokart/" target="blank"><img align="center"  src="https://img.icons8.com/color/48/000000/twitter--v2.png" alt="https://www.twitter.com/marokart/" height="30" width="30" /></a>
  <a href="https://www.kaggle.com/dajebbar" target="blank">
    <img align="center" src="https://img.icons8.com/external-tal-revivo-shadow-tal-revivo/38/external-kaggle-an-online-community-of-data-scientists-and-machine-learners-owned-by-google-logo-shadow-tal-revivo.png" alt="https://www.kaggle.com/dajebbar" height="30" width="30" /></a>
  
</p>
