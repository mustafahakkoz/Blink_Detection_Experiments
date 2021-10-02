## Blink Detection Experiments:

2019-2020 Spring CSE4097 - Graduation Project I

3 different blink detection methods are examined in the scope of graduation thesis as a side project. We were planning to uses these methods to extract blink-based features, but we gave up on these later.

1) **simple thresholding:** Eye-aspect-ratio based model with constant thresholds which defines a proper blink behaviour.

2) **adaptive model:** Same approach above with adaptive thresholds.

3) **machine learning model:** Completely different experiment with a machine learning approach.

---

#### Online Notebooks:

1. [Simple model](https://www.kaggle.com/hakkoz/eye-blink-detection-1-simple-model)

2. [Adaptive model](https://www.kaggle.com/hakkoz/eye-blink-detection-2-adaptive-model-v2)

3. a. [ML model part-1](https://www.kaggle.com/hakkoz/eye-blink-detection-3-ml-model-part1)

   b. [ML model part-2](https://www.kaggle.com/hakkoz/eye-blink-detection-3-ml-model-part2)

4. [Comparison of models](https://www.kaggle.com/hakkoz/eye-blink-detection-4-comparison)

5. a. [Utility script-1](https://www.kaggle.com/hakkoz/utils)

   b. [Utility script-2](https://www.kaggle.com/hakkoz/utils2)

   c. [Utility script-3](https://www.kaggle.com/hakkoz/utils3)

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

- Building a pipeline to read videos and annotations of [TalkingFace](https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/data/talking_face/talking_face.html) dataset, to evaluate the simple model on it and to write / to read output files.

[**eye-blink-detection-2-adaptive-model.ipynb**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/eye-blink-detection-2-adaptive-model.ipynb)

- Analyzing blink detection problem with timeseries approach. The goal is to make 3 thresholds adaptive to any situation.

- To make the threshold SKIP_FIRST_FRAMES adaptive, converging slope of linear fitting line is used.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada1.png" alt="" height="200">

- Instead of using a constant value for EAR_THRESHOLD, outlier detection is used. 3 different methods IQR, confidence intervals and limiting z values are experimented.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada2.png" alt="" height="200">

- To overcome the issue with false blinks while yawning and smiling, *Error of EAR-EWMA (exponentially weighted moving average)* is used insted of direct usage of EAR values in outlier detection.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada3.png" alt="" height="30">
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada4.png" alt="" height="200">

- For the third threshold EAR_CONSEC_FRAMES, the relation between significant values of PACF plot and blink durations, is examined.  
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada7.png" alt="" height="300">

- Stationarity analysis, Dickey-Fuller test, rolling mean, exponentially decaying and time-shifting and confidence intervals are used to build a handcrafted function to estimate significant values of PACF.
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada8.png" alt="" height="300">

- Another method, grid search with ARIMA, is tested. Best parameters of ARIMA is found then p value is chosen as significant value of PACF.  
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada9.png" alt="" height="200">

- Other experiments on exploring RSI (Relative Strength Indicator), CasualImpact.  
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada5.png" alt="" height="40">
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada6.png" alt="" height="200">
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ada10.png" alt="" height="300">

- After all of these experiments, the most efficient adaptive model is decided as a pipeline of ``EWMA + outlier detection with z values + guessing significant points of PACF``. 

[**eye-blink-detection-3-ml-model-part1.ipynb**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/eye-blink-detection-3-ml-model-part1.ipynb)

- For this part [Eyeblink8](https://www.blinkingmatters.com/research) dataset is processed to get EAR values and annotations to train ML models in classification phase.

[**eye-blink-detection-3-ml-model-part2.ipynb**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/eye-blink-detection-3-ml-model-part2.ipynb)

- For this part, the ML model described in the paper by Soukupova and Cech (2016) is implemented. Output dataframes of part-1 is used. (n x 13) matrix of training set is constructed. Columns are defining ``13 frame window (6 previous frames + current frame + 6 next frames)``. Then the model is evaluated on same test set (TalkingFace).  
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/ml1.png" alt="" height="300">

- Another SVM implementation by https://github.com/rezaghoddoosian/Early-Drowsiness-Detection is tested and compared with our results. But it was an old version of SVC (scikit<0.18) and its **predict()** method gives error so manual implementation of rbf prediction method is developed. (see https://stackoverflow.com/questions/28593684/replication-of-scikit-svm-srv-predictx)

[**eye-blink-detection-4-comparison.ipynb**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/eye-blink-detection-4-comparison.ipynb)

- Comparison of 4 models: simple model, adaptive model, ml model, ml model from RLDD.  
  <img title="" src="https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/images/newplot.png" alt="" height="400">

[**utils.py**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/utils/utils.py)

- Utility functions based on "eye_blink_detection_1_simple_model.ipynb".

[**utils2.py**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/utils/utils2.py)

- Utility functions based on "eye_blink_detection_2_adaptive_model.ipynb".

[**utils3.py**](https://github.com/mustafahakkoz/Blink_Detection_Experiments/blob/main/utils/utils3.py)

- Utility functions based on "eye_blink_detection_3_ml_model_part1" and "eye_blink_detection_3_ml_model_part2".



