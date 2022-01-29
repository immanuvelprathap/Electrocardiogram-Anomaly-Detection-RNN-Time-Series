# Electrocardiogram-Anomaly-Detection-RNN-Time-Series

RNN based Time-series Anomaly detector model implemented in Pytorch.

This is an implementation of RNN based time-series anomaly detector, which consists of two-stage strategy of time-series prediction and anomaly score calculation.


## Requirements
* Python 3.5+
* Pytorch 0.4.0+
* Numpy
* Matplotlib
* Scikit-learn

## Dataset you can implement on!
__1. NYC taxi passenger count__
 * The New York City taxi passenger data stream, provided by the [New
York City Transportation Authority](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml )
 * preprocessed (aggregated at 30 min intervals) by Cui, Yuwei, et al. in ["A comparative study of HTM and other neural network models for online sequence learning with streaming data." Neural Networks (IJCNN), 2016 International Joint Conference on. IEEE, 2016.](http://ieeexplore.ieee.org/abstract/document/7727380/)
  , [code](https://github.com/numenta/htmresearch/tree/master/projects/sequence_prediction)

__2. Electrocardiograms (ECGs)__
 * The ECG dataset containing a single anomaly corresponding to a pre-ventricular contraction

__3. 2D gesture (video surveilance)__
 * X Y coordinate of hand gesture in a video

__4. Respiration__
 * A patients respiration (measured by thorax extension, sampling rate 10Hz)

__5. Space shuttle__
 * Space Shuttle Marotta Valve time-series

__6. Power demand__
 * One years power demand at a Dutch research facility

The Time-series 2~6 are provided by E. Keogh et al. in
["HOT SAX: Efficiently Finding the Most Unusual Time Series Subsequence." In The Fifth IEEE International Conference on Data Mining. (2005)
](http://ieeexplore.ieee.org/abstract/document/1565683/)
  , [dataset](http://www.cs.ucr.edu/~eamonn/discords/)


## Result for ECG Dataset
__1. Time-series prediction:__
Predictions from the stacked RNN model


![prediction2](https://github.com/immanuvelprathap/Electrocardiogram-Anomaly-Detection-RNN-Time-Series/blob/master/result/ecg/fig.gif)

__2. Anomaly detection:__

Anomaly scores from the Multivariate Gaussian Distribution model


* Electrocardiograms (ECGs) (filename: chfdb_chf14_45590)


![scores3](https://github.com/immanuvelprathap/Electrocardiogram-Anomaly-Detection-RNN-Time-Series/blob/master/result/ecg/chfdb_chf13_45590/fig_detection/fig_scores_channel0.png)


![scores4](https://github.com/immanuvelprathap/Electrocardiogram-Anomaly-Detection-RNN-Time-Series/blob/master/result/ecg/chfdb_chf13_45590/fig_detection/fig_scores_channel1.png)

## Evaluation

Model performance was evaluated by comparing the model output with the pre-labeled ground-truth. Note that the labels are only used for model evaluation. The anomaly score threshold was increased from 0 to some maximum value to plot the change of precision, recall, and f1 score. Here we show only the results for the ECG dataset. Execute the code yourself and see more results.

__1. Precision, recall, and F1 score:__

* Electrocardiograms (ECGs) (filename: chfdb_chf14_45590)

a. channel 0

![f1ecg1](https://github.com/immanuvelprathap/Electrocardiogram-Anomaly-Detection-RNN-Time-Series/blob/master/result/ecg/chfdb_chf13_45590/fig_detection/fig_f_beta_channel0.png)

b. channel 1

![f1ecg2](https://github.com/immanuvelprathap/Electrocardiogram-Anomaly-Detection-RNN-Time-Series/blob/master/result/ecg/chfdb_chf13_45590/fig_detection/fig_f_beta_channel1.png)
