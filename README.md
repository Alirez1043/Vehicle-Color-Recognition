# Vehicle-Color-Recognition

## How to Run App

### Tools 

+ ### [ Docker](https://docs.docker.com/desktop/)
+ ### [Tensorflow_Serving](https://www.tensorflow.org/tfx/guide/serving)
+ ### [Git_lfs](https://git-lfs.github.com/)

---------------------------------------------------------------------------------------------
**To clone large repo you need to install git lfs ,so install it and follow below instructions** .

### 1. Clone the project
```
git lfs clone https://github.com/Alirez1043/Vehicle-Color-Recognition.git
```
Change current directory to project directory
```
cd Vehicle-Color-Recognition
```
### 2. Build and run the Docker Stack (TensorFlow serving and Web App)
```
docker-compose build 
```
```
docker-compose up
```
### 3. Open the Web App and try it out !
The App should be running on    ```http://localhost:8080```

