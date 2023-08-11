# Toxic Comment Classification and Discord Bot

![Python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MLflow](https://img.shields.io/badge/MLflow-1.x-green.svg)

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Dataset](#dataset)
- [Word Embedding](#word-embedding)
- [Experimental Tracking](#experimental-tracking)
- [Best Model](#best-model)
- [Discord Bot Deployment](#discord-bot-deployment)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Overview

This repository presents a comprehensive toxic comment classification project that encompasses preprocessing, model training, experimental tracking, and real-time deployment as a Discord bot. The primary objectives are:

- Train and evaluate three distinct models (LSTM, BiLSTM, BiLSTM+CNN) for toxic comment classification.
- Employ the Kaggle dataset, comprising comments categorized into six levels of toxicity.
- Leverage Sentence Transformer for word embedding, enhancing model performance.
- Utilize MLflow for efficient experimental tracking and model comparison.
- Deploy the best-performing model (BiLSTM) as a Discord bot capable of identifying and deleting toxic messages in real-time.

## Models

The project explores three advanced deep learning architectures:
1. **LSTM Model:** Implements a Long Short-Term Memory (LSTM) network for comment classification.
2. **BiLSTM Model:** Utilizes a Bidirectional LSTM (BiLSTM) network to capture bidirectional context.
3. **BiLSTM + CNN Model:** Combines a Bidirectional LSTM with a Convolutional Neural Network (CNN) for enhanced feature extraction.

## Dataset

The dataset is sourced from Kaggle and consists of comments labeled across six toxicity levels. The dataset contains a total of 160,000 samples, each with associated toxicity labels.

## Word Embedding

The project harnesses Sentence Transformer for word embedding. This technique enhances model performance by capturing intricate semantic relationships between words.

## Experimental Tracking

Experimental tracking and comparison of different model iterations are facilitated through MLflow. This ensures comprehensive monitoring and evaluation of various metrics.

## Best Model

After extensive evaluation, the BiLSTM model has emerged as the top-performing architecture, showcasing remarkable accuracy, F1-score, precision, and recall.

## Discord Bot Deployment

The most successful model, BiLSTM, has been deployed as a Discord bot. This bot effectively identifies and promptly deletes toxic messages, contributing to a healthier online conversation environment.

## Results

Performance evaluation of the models yielded the following key metrics:

| Model        | Accuracy | F1-Score | Precision | Recall |
|--------------|----------|----------|-----------|--------|
| LSTM         | 0.97     | 0.97     | 0.971     | 0.97   |
| BiLSTM       | 0.973    | 0.972    | 0.971     | 0.973  |
| BiLSTM + CNN | 0.969    | 0.97     | 0.971     | 0.969  |

## Acknowledgements

We extend our gratitude to Kaggle for providing the dataset and the open-source community for the invaluable tools and libraries used throughout this project.

## Contact

For inquiries or feedback, please contact Samman Shrestha(mailto:shresthasamman125@gmail.com).


