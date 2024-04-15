# Music Genre Classification

## Description
This repository presents a toolkit for working with MPEG7 descriptors in image processing tasks. It provides functionalities for training, evaluating, and querying image descriptors using Python and OpenCV.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Dataset](#dataset)
- [Results](#results)

## Overview
This project offers a set of scripts for working with MPEG7 descriptors, including training descriptor models on a dataset, evaluating their performance, and querying image with this dataset.

## Installation
To use this toolkit, follow these steps:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.

## Usage
The project provides three main functionalities: training, evaluation, and querying the model with audio file to predict its genre.

### Training
Trains the AI model using the provided dataset, saving the trained model with the specified name.
To train a model, use the following command:
```bash
python script.py train [dataset_path] [model_name]
```

* `dataset_path`: Path to the dataset directory containing separate directories for training, validation, and testing sets.
* `model_name`: Name of the model file to save (.ckpt). The model will be save in the `trained_models` folder.

### Evaluation
Evaluates the performance of the AI model using the provided dataset and the specified trained model. It will compute the accuracy and displays the confusion matrix on the testing set.
To evaluate the performance of the model, execute the following command.
```bash
python main.py eval [dataset_path] [model_name]
```

* `dataset_path`: Path to the dataset directory containing separate directories for training, validation, and testing sets.
* `model_name`: Name of the model file to use (.ckpt). The model must be in the trained_models folder.

### Query
Predicts the genre of an audio file using the trained model.
To query an audio file, use the following command:
```bash
python main.py query [audio_file_path] [model_name
```

* `audio_file_path`: Path to the audio file for querying.
* `model_name`: Name of the model file to use (.ckpt). The model must be in the trained_models folder.

## Model

The model is a convolutional neural network (CNN) designed for audio classification tasks. It includes convolutional layers followed by max-pooling for feature extraction, followed by dense layers for classification. Dropout regularization is applied to prevent overfitting. 

To evaluate a new song, the model divide the audio file into a certain number of chunks (modifiable in the [config file](config.py) ) and classify each of these chunks. At the end, it computes the average score of each genre for all the chunks and the greatest is the predicted genre.

## Dataset

The script automatically checks if the dataset directory contains separate directories named `train`, `val`, and `test` for training, validation, and testing sets, respectively. If the dataset is not already split into these directories, it automatically splits the dataset using the split_dataset function before proceeding with training or evaluation. The percentages of the split are defined in the [config file](config.py).

```
dataset/
│
├── train/
│   ├── genre1/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   │
│   ├── genre2/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   │
│   └── ...
│
├── val/
│   ├── genre1/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   │
│   ├── genre2/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   │
│   └── ...
│
└── test/
    ├── genre1/
    │   ├── audio1.wav
    │   ├── audio2.wav
    │   └── ...
    │
    ├── genre2/
    │   ├── audio1.wav
    │   ├── audio2.wav
    │   └── ...
    │
    └── ...
```

The audio file name name are not relevant. The image extension allower are `.mp3` `.wav` `.ogg`, `.flac`, `.aac` and `.au`

## Results

To assess the model's performance, I used the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification). The dataset consists of 1000 audio tracks, each 30 seconds long, evenly divided into 10 genres (so 100 tracks per genre). The 10 genres are : 
* Rock
* Metal
* Blues
* Disco
* Jazz
* Classical
* Country
* Hip-hop
* Pop
* Reggae

By training a model on this dataset and performing the evaluation on the `test` set, we obtain an accuracy of 79% and the following confusion matrix :

<img width="509" alt="Capture d’écran 2024-04-15 à 15 37 49" src="https://github.com/lucas-gtr/MusicGenreClassification/assets/12534925/83bd57cf-64f5-43c7-b199-4bf6ca65aed9">

<img width="1141" alt="Capture d’écran 2024-04-15 à 15 38 41" src="https://github.com/lucas-gtr/MusicGenreClassification/assets/12534925/467327e0-0365-456b-b359-5ee5246ef64d">

The results of the model evaluation are pretty good with an high accuracy among most music genres. However, the model tends to struggle to distinguish rock and disco genres, occasionally misclassifying rock tracks as blues or disco, and disco tracks as reggae. Overall, the model is pretty accurate and is ready to be used on other music.
