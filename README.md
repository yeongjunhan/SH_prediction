# SH_prediction
SH project (July 2018 to January 2019) sudo code

## Purpose
Predict stock's directional prices (2 classes. Compared to prediction period, stock prices rise or fall) with Convolutional Neural Network(CNN) and lagged logistic regression. Especially, market is some developed countries.

## Data set
Embedded Reuter news headline and Stock values over the past 20 years.

## Algorithm
Based on the reference (Deep Learning for Event-Driven Stock Prediction. Ding et al.(2015), the algorithm is as following:

0. Make haedlines into word vectors. (NLP)
1. Summarize daily word vectors to represent the feature of that day using Principal Component(PC), Inter Quartile Range(IQR), and Median
2. Build CNN graph and Extract the local information of headline news.
3. Get the optimal hyper-parameters on tuning procedures. Performance measure is the validation accuracy (Random search).
4. Measure the performance of the selected model.

## Results

Back testing results are as follows.

| Market | Performance(3M) |
|:------:|:---------------:|
|   JP   |      64.3%      |
|   DE   |      58.4%      |
|   GB   |      54.5%      |
|   S&P  |      63.1%      |
|   KR   |      61.9%      |

## License
- YeongJun, Han.(SNU graduate university, master degree at Aug, 2019, major in statistics.
- Data and paths are not available.
- training.py and daily_prediction.py script are not available.
