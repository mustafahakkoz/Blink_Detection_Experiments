## Blink Detection Experiments:

2019-2020 Spring CSE4097 - Graduation Project I

3 different blink detection methods are examined in the scope of graduation thesis as a side project. We were planning to uses these methods to extract blink-based features, but we gave up on these later.

1) **simple thresholding:** Eye-aspect-ratio based model with constant thresholds which defines a proper blink behaviour.

2) **adaptive model:** Same approach above with adaptive thresholds.

3) **machine learning model:** Completely different experiment with a machine learning approach.

---

#### Repo Content and Implementation Steps:

[**eye-blink-detection-1-simple-model.ipynb**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/eye-blink-detection-1-simple-model.ipynb)

- Basic model is an implementation of [paper by Kazemi and Sullivan (2014).](https://www.semanticscholar.org/paper/One-millisecond-face-alignment-with-an-ensemble-of-Kazemi-Sullivan/d78b6a5b0dcaa81b1faea5fb0000045a62513567) It uses Dlib's "get_frontal_face_detector" (HoG and SVM) to detect faces and "shape_predictor" (ensemble of regression trees) to detect facial landmarks. Then it uses EAR formula to get eye openness.

- After calculating EAR values, blinks are detected by using 3 constant thresholds:
  
  ```
  # eye aspect ratio to indicate blink
  EAR_THRESHOLD = 0.21
  
  # number of consecutive frames the eye must be below the threshold
  EAR_CONSEC_FRAMES = 3 
  
  # how many frames we should skip at the beggining
  SKIP_FIRST_FRAMES = 0 
  ```

- Building a pipeline to read videos and annotations of *TalkingFace* dataset, to evaluate the simple model on it and to write / to read output files.

[**eye-blink-detection-2-adaptive-model.ipynb**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/eye-blink-detection-2-adaptive-model.ipynb)

- Analyzing blink detection problem with timeseries approach. The goal is to make 3 thresholds adaptive to any situation.

- To make the threshold SKIP_FIRST_FRAMES adaptive, converging slope of linear fitting line is used.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada1.png" alt="" height="200">

- Instead of using a constant value for EAR_THRESHOLD, outlier detection is used. 3 different methods IQR, confidence intervals and limiting z values are experimented.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada2.png" alt="" height="200">

- To overcome the issue with false blinks while yawning and smiling, *Error of EAR-EWMA (exponentially weighted moving average)* is used insted of direct usage of EAR values in outlier detection.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada3.png" alt="" height="100">
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada4.png" alt="" height="200">

- For the third threshold EAR_CONSEC_FRAMES, the relation between significant values of PACF plot and blink durations, is examined.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada7.png" alt="" height="300">

- Stationarity analysis, Dickey-Fuller test, rolling mean, exponentially decaying and time-shifting and confidence intervals are used to build a handcrafted function to estimate significant values of PACF.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada8.png" alt="" height="300">

- Another method, grid search with ARIMA, is tested. Best parameters of ARIMA is found then p value is chosen as significant value of PACF.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada9.png" alt="" height="300">

- Other experiments on exploring RSI (Relative Strength Indicator), CasualImpact.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada5.png" alt="" height="100">
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada6.png" alt="" height="200">
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada10.png" alt="" height="300">

- After all of these experiments, the most efficient adaptive model is decided as a pipeline of ``EWMA + outlier detection with z values + guessing significant points of PACF``. 


