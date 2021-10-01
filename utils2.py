# utility functions 2: adaptive-model
# based on "eye_blink_detection_2_adaptive_model.ipynb"

# import libraries
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

# import utility functions
from utils import *

############################################################################################################

# estimate SKIP_FIRST_FRAMES threshold
# returns n -> estimated value of the threshold
# m0 -> slope at this point
# array of slopes until this point
def estimate_first_n(data, start_n=50, limit_n=300, step=1, epsilon=10e-8):
    n = start_n 
    M=[]
    while True:
        # for first n values fit a linear regression line 
        data0 = data[:n]
        m0,b0 = np.polyfit(np.arange(n),data0, 1)
        M.append(m0)
        
        # check if n + step reaches limit
        if n + step > limit_n-1:
            print("error - reached the limit")
            break

        # for first n + step values fit a linear regression line 
        data1 = data[:n+step]
        m1,b1 = np.polyfit(np.arange(n + step),data1, 1)

        # if m1-m0 converges to epsilon
        if abs(m1 - m0) < epsilon and m0 > 0:
            break
        n += step
        
    return n, m0, M

############################################################################################################

# utility function to get fps of the video and to calculate frame count from given duration
def secs_to_frame_count(input_full_path, seconds):
    # read fps of the video
    fps = cv2.VideoCapture(input_full_path).get(cv2.CAP_PROP_FPS)
    # calculate 
    frame_count = seconds * fps
    return frame_count

############################################################################################################

# remove outliers from a list by using IQR method
def detect_outliers_iqr(input_list, alpha=1.5):
    # calculate interquartile range
    q25, q75 = np.percentile(input_list, 25), np.percentile(input_list, 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    
    # calculate the outlier cutoff
    cut_off = iqr * alpha
    lower, upper = q25 - cut_off, q75 + cut_off
    
    # identify outliers
    outliers = [(i, x) for (i, x) in list(enumerate(input_list, 1)) if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    
    # remove outliers
    clean_input_list = [x for x in input_list if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(clean_input_list))
    print("")
    
    return clean_input_list, outliers, upper, lower

############################################################################################################

# calculate upper and lower limits for given confidence interval
def detect_outliers_conf(input_list, confidence=0.95):
    # identify boudaries
    lower, upper  = st.t.interval(confidence, len(input_list)-1, loc=np.mean(input_list), \
                                        scale=st.sem(input_list))
    
    # identify outliers
    outliers = [(i, x) for (i, x) in list(enumerate(input_list, 1)) if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    
    # remove outliers
    clean_input_list = [x for x in input_list if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(clean_input_list))
    print("")
    
    return clean_input_list, outliers, upper, lower

############################################################################################################

# calculate upper and lower limits for given confidence interval
def detect_outliers_z(input_list, z_limit=2):
    # identify boudaries
    mu = input_list.mean()
    sigma = input_list.std()
    val = z_limit * sigma
    lower  =  mu - val
    upper =  mu + val
    
    # identify outliers
    outliers = [(i, x) for (i, x) in list(enumerate(input_list, 1)) if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    
    # remove outliers
    clean_input_list = [x for x in input_list if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(clean_input_list))
    print("")
    
    return clean_input_list, outliers, upper, lower

############################################################################################################

# calculate EWMA, error and then outliers to detect blinks
def detect_closeness_ewma(data_df):
    # calculate EWMA with low weight on current values (alpha) so span is 10
    data_ewm = data_df.ewm(alpha=0.1).mean()

    # calculate mean_squared_error
    data_err = data_df - data_ewm
    
    # run detect_outliers_iqr() on errors
    _, outliers, _, _ =  detect_outliers_z(data_err,3)

    # only get negative outliers
    outliers_neg = [(a,b) for (a,b) in outliers if b<0]
    outlier_indexes = list(zip(*outliers_neg))[0]
    outlier_values = list(zip(*outliers_neg))[1]
    
    return outlier_indexes, outlier_values

############################################################################################################

# get first blink
# returns last frame of the blink detected
def construct_blinks_ewma(data_df):
    # get outliers
    closed_indexes, _ = detect_closeness_ewma(data_df)
    
    # detect consecutive closed frames
    gaps = [(s, e) for s, e in zip(closed_indexes, closed_indexes[1:]) if s+1 < e] # find gaps between groups
    borders = iter([closed_indexes[0]] + list(sum(gaps, () )) + [closed_indexes[-1]]) # use sum() as a trick to flatten gaps, then append first and last frames 
    groups = list(zip(borders, borders)) # zip again to build group again
    groups_verbose = [tuple(range(s, e + 1)) for (s,e) in groups] # write elements of groups verbosely
    
    # correct gaps and construct verbosely
    gaps_verbose = [tuple(range(s+1,e)) for (s, e) in gaps]
    
    return groups_verbose, gaps_verbose, closed_indexes

############################################################################################################

# updated version of estimate_first_n by EWMA
# estimate SKIP_FIRST_FRAMES threshold
# returns n -> estimated value of the threshold
# m0 -> slope at this point
# array of slopes until this point
def estimate_first_n_v2(data_df, start_n=40, limit_n=300, step=1, epsilon=10e-8):
    # run get_first_blink(data) to get blink_guarenteed
    blinks, _, _ = construct_blinks_ewma(data_df)               
    first_blink = blinks[0]
    ending_frame = first_blink[-1]
    
    n = start_n 
    M=[]
    while True:
        # for first n values fit a linear regression line 
        data0 = data_df[:n]
        m0,b0 = np.polyfit(np.arange(n),data0, 1)
        M.append(m0)
        
        # check if n + step reaches limit
        if n + step > limit_n-1:
            print("error - reached the limit")
            break

        # for first n + step values fit a linear regression line 
        data1 = data[:n+step]
        m1,b1 = np.polyfit(np.arange(n + step),data1, 1)

        # if m1-m0 converges to epsilon
        if abs(m1 - m0) < epsilon and m0 > 0 and n > ending_frame:
            break
        n += step
    
    return n, m0, M

############################################################################################################

# Relative strength indicator
def rsi_rs(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs), rs

############################################################################################################

# print statistics of rolling mean/std and dickey-fuller test
def get_stationarity(timeseries_df):
    
    # rolling statistics
    rolling_mean = timeseries_df.rolling(window=12).mean()
    rolling_std = timeseries_df.rolling(window=12).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries_df, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries_df)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    return result[0], result[1], result[4]['1%'], result[4]['5%'], result[4]['10%']

############################################################################################################

# function runs acf and pacf then finds significant values
def find_significant_vals(timeseries, nlags=40, is_first=True):
    # calculate acf and pacf values
    _, conf_acf = acf(timeseries,  alpha=.05, nlags=nlags)
    _, conf_pacf = pacf(timeseries,  alpha=.05, nlags=nlags)
    
    # initilize output values
    sig_acf = 0
    sig_pacf = 0
    
    # for acf
    for i in range(nlags):
        if conf_acf[i][0]>0:
            sig_acf = i
        elif conf_acf[i][1]<0:
            sig_acf = i
        else:
            break
    
    # for pacf
    for i in range(nlags):
        if conf_pacf[i][0]>0:
            sig_pacf = i
        elif conf_pacf[i][1]<0:
            sig_pacf = i
        else:
            break
    
    # if significant numbers are less than 3 (1th is already 0 lag so it doesn't count)
    # try to detect one more sequence of significant values
    privelege = 0
    end = 0
    if sig_acf<3:
        # for acf
        for i in range(nlags):
            if conf_acf[i][0]>0:
                sig_acf = i
                if privelege > 0:
                    end=1
            elif conf_acf[i][1]<0:
                sig_acf = i
                if privelege > 0:
                    end=1
            else:
                if end == 1:
                    break
                privelege += 1
                continue
                
    privelege = 0
    end = 0            
    if sig_pacf<3:
        # for pacf
        for i in range(nlags):
            if conf_pacf[i][0]>0:
                sig_pacf = i
                if privelege > 0:
                    end=1
            elif conf_pacf[i][1]<0:
                sig_pacf = i
                if privelege > 0:
                    end=1
            else:
                if end == 1:
                    break
                privelege += 1
                continue
    
    # if significant numbers are more than 9 use error of moving average to approximate
    if sig_pacf>9 or sig_acf>9:
        rolling_mean = timeseries.rolling(window=10).mean()
        cal_df_minus_mean = timeseries - rolling_mean
        cal_df_minus_mean.dropna(inplace=True)
        if is_first == True:
            sig_pacf, sig_acf = find_significant_vals(cal_df_minus_mean, nlags, False)
    return  sig_pacf, sig_acf

############################################################################################################

# evaluate combinations of p, d and q values for an ARIMA model
def gridsearch_ARIMA_params(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    model = ARIMA(dataset, order=order)
                    results = model.fit(disp=0)
                    mse = mean_squared_error(dataset[d:], results.fittedvalues)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.8f' % (order,mse))
                except:
                    print('ARIMA{} exception'.format(order))
                    continue
    print('Best ARIMA%s MSE=%.8f' % (best_cfg, best_score))
    return best_cfg, best_score

############################################################################################################

# recalculate outputs of process_video()
def recalculate_data_ewma(frame_info_df, closeness_list, blink_list, consec_th):
    # reconstruct blinks using EWMA method
    data_df = frame_info_df['avg_ear']
    blinks, gaps, closed_indexes = construct_blinks_ewma(data_df) 
    
    # rebuild closeness_list
    length = len(frame_info_df)
    closeness_list = [0] * length
    for i in closed_indexes:
        closeness_list[i] = 1

    # update frame_info_df['closeness']
    new_column_closeness = pd.Series(closeness_list, name='closeness')
    frame_info_df.update(new_column_closeness)
    
    # update frame_info_df['blink_no']
    for blink in blinks:
        if len(blink) < consec_th:
            blinks.remove(blink)
            
    blink_start_frame_list = [0] * len(data)
    blink_end_frame_list = [0] * len(data)
    blink_no_list = [0] * len(data)
    for i, blink in enumerate(blinks):
        for frame in blink:
            blink_no_list[frame-1] = i
        blink_no_list[frame:] = [i+1]*len(blink_no_list[frame:])
        blink_start_frame_list[frame:] = [blink[0]]*len(blink_start_frame_list[frame:])
        blink_end_frame_list[frame:] = [blink[-1]]*len(blink_end_frame_list[frame:])

        
    new_column_blink_no = pd.Series(blink_no_list, name='blink_no')
    frame_info_df.update(new_column_blink_no)
    
    # update frame_info_df['blink_start_frame']
    new_column_blink_start_frame = pd.Series(blink_start_frame_list, name='blink_start_frame')
    frame_info_df.update(new_column_blink_start_frame)
    
    # update frame_info_df['blink_end_frame']
    new_column_blink_end_frame = pd.Series(blink_end_frame_list, name='blink_end_frame')
    frame_info_df.update(new_column_blink_end_frame)
    
    return frame_info_df, closeness_list, blink_list

############################################################################################################

# build adaptive_model pipeline
# if you want to display blinks set display_blinks=True (it requires long time and memory so default is False)
# if you want to read annotation file and run comparison metrics set test_extention="tag" or any file extension
# REMARK: your annotation file an video file must have the same name to use this function.
# if you want to write outputs set write_results=True
# You change maximum duration of calibration phase in secs by calibration_max argument. Default is 10 secs
def adaptive_model(input_full_path, display_blinks=False, test_extention=False, write_results=False, \
                   calibration_max=10):
    
    # calculate a limit_n for "calibration_max" secs with using fps of the video
    limit_frame = secs_to_frame_count(input_full_path, calibration_max)
    
    # process the video and get the ear values for the first "limit_frame" values
    df_init, _, _, _, _, _ = process_video(input_full_path, up_to=limit_frame)
    
    # build series
    data_init = df_init['avg_ear']
        
    # estimate skip_n and build calibration dataframe
    skip_n, _, _ = estimate_first_n_v2(data_init, limit_n=limit_frame)
    calibration = data_init[:skip_n]
    cal_df = pd.DataFrame(calibration)
    
    # estimate consec_th using calibration dataframe
    consec_th, _ = find_significant_vals(cal_df, nlags=40)
    
    # after the estimation of 3 thresholds, process the video wholly and get the full results
    frame_info_df, closeness_predictions, blink_predictions, frames, video_info, scores_string \
        = process_video(input_full_path)
    
    # recalculate outputs by using three thresholds
    frame_info_df, closeness_predictions, blink_predictions = recalculate_data_ewma(frame_info_df, \
                                                            closeness_predictions,blink_predictions, consec_th)
              
    # recalculate data again by skipping "skip_n" frames
    frame_info_df, closeness_predictions_skipped, blink_predictions_skipped, frames_skipped \
        = skip_first_n_frames(frame_info_df, closeness_predictions, blink_predictions, frames, \
                              skip_n = skip_n)

    # first display statistics by using original outputs
    scores_string += display_stats(closeness_predictions, blink_predictions, video_info)

    # then display statistics by using outputs of skip_first_n_frames() function which are 
    #"closeness_predictions_skipped" and "blinks_predictions_skipped"
    if(skip_n > 0):
        scores_string += display_stats(closeness_predictions_skipped, blink_predictions_skipped, video_info, \
                                 skip_n = skip_n)
    
    # if you want to display blinks
    if display_blinks == True:
        # display starting, middle and ending frames of all blinks by using "blinks" and "frames"
        display_blinks(blink_predictions_skipped, frames_skipped)
        
    # if you want to read tag file
    if test_extention != False:
        extention = ""
        # default file extension is ".tag"
        if test_extention == True:
            extention = "tag"
        else:
            extention = test_extention
        # remove video extention i.e. ".avi"
        clean_path = os.path.splitext(input_full_path)[0]
        # read tag file
        closeness_test, blinks_test = read_annotations("{}.{}".format(clean_path, extention), skip_n = skip_n)
        # display results by using outputs of read_annotations() function 
        # which are "closeness_test", "blinks_test"
        scores_string += display_stats(closeness_test, blinks_test, skip_n = skip_n, test = True)
        # display results by using "closeness_test" and "closeness_predictions"
        scores_string += display_test_scores(closeness_test, closeness_predictions_skipped)
        
    # if you want to write results
    if write_results == True:
        # write prediction output files by using outputs of skip_first_n_frames() function
        write_outputs(input_full_path, closeness_predictions_skipped, blink_predictions_skipped, \
                      frame_info_df, scores_string)
        if test_extention != False:
            # write test output files by using outputs of skip_first_n_frames() function
            # no need to write frame_info_df and scores_string since they already have written above
            write_outputs(input_full_path, closeness_test, blinks_test, \
                          test = True)
            
    return frame_info_df, closeness_predictions_skipped, blink_predictions_skipped, frames_skipped, \
            video_info, scores_string

############################################################################################################
