import datetime as dt
import numbers
import os
from collections import defaultdict
from copy import deepcopy
from itertools import product

import click
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy.stats as st
import yfinance  # import fix_yahoo_finance formerly
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, Add, Multiply
from keras.layers import Input, Dense, Dropout, Concatenate, Conv3D, Lambda, Conv2D, LeakyReLU
from keras.models import load_model
from pandarallel import pandarallel
from pandas_datareader import data as pdr
from sklearn import metrics as skmetrics
from tqdm import tqdm

from utils import SinCosPositionalEmbedding, np_pad_to_size, vectorize, get_date_infos_discrete, get_date_infos, \
    tf_lookback, conditioned_continuous

pandarallel.initialize(progress_bar=True, nb_workers=4)
yfinance.pdr_override()


def load_glove(glove_file='./glovepretrained/glove.6B.50d.txt'):
    """
    Load embeddings glove as gensim model.
    """
    tmp_file = get_tmpfile("test_word2vec.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    return KeyedVectors.load_word2vec_format(tmp_file)


def group_by_timeseries(timeseries_align, dataframe, datecolumn):
    """
    Creates a list, that has on a position i a list of all entries corresponding to all events from dataframe with
    dates from timeseries_align[i-1] to timeseries_align[i].
    datecolumn specifies the column from dataframe to use for dates (or index if being None)
    """
    if datecolumn is None:
        date_series = dataframe.index
        date_series_len = len(date_series)
    else:
        date_series = dataframe[datecolumn].iloc
        date_series_len = len(dataframe[datecolumn])
    # combine dates based on the info we need: #aggregate by workdays in "data"
    newsi = 0
    newsdate = date_series[newsi]
    news_up_to_workday = []
    for worki, workdate in tqdm(enumerate(timeseries_align)):
        all_covered_news = []
        if newsi < date_series_len:
            while workdate >= newsdate:
                all_covered_news.append(dataframe.iloc[newsi])
                newsi += 1
                if newsi >= date_series_len:
                    break
                newsdate = date_series[newsi]
        # now we have moved to a newsday, that is more recent, than the selected workday
        # so lets save all and then move one workday
        news_up_to_workday.append(all_covered_news)
    return news_up_to_workday


def load_reddit_news(embeddings_model,
                     timeseries_align,
                     rednews_csv="./rednews/worldnewsen.csv",
                     ntopbyday=25,
                     max_words_len=20,
                     ):
    """
    From a dataset 'rednews_csv' of all reddit entries, creates a numpy array corresponding to top scored
    'ntopbyday' entries for the days in between timeseries_align[i-1] and timeseries_align[i],
     embedds them by using gensim model embeddings_model.
    Each title is truncated (or padded if shorter) to 'max_words_len' words.

    Returns numpy array of a shape (len(timeseries_align), ntopbyday, max_words_len, embedding size)
    """
    data_news = pd.read_csv(rednews_csv, delimiter=';')

    data_news['published_date'] = data_news['published'].parallel_apply(
        lambda text: dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").date())
    
    top_by_day = data_news.sort_values(by=['published_date', 'score'],
                                       ascending=[True, False]).groupby(['published_date']).head(n=ntopbyday)
    
    def block_trunc_pad_zeroes(item, xlen=max_words_len):
        return np_pad_to_size([vectorize(word, embeddings_model) for word in item[:xlen]], minsizes=(None, xlen, None))
    
    top_by_day['embedded'] = top_by_day['title'].str.split().apply(block_trunc_pad_zeroes)
    top_by_day['parseddate'] = top_by_day['published_date']
    
    # combine dates based on the info we need: #aggregate by workdays in "data"
    news_up_to_workday = group_by_timeseries(timeseries_align, top_by_day, 'parseddate')
    
    for i, newslist in enumerate(news_up_to_workday):
        newslist.sort(key=lambda item: item['score'], reverse=True)
        news_up_to_workday[i] = [news['embedded'] for news in newslist[:ntopbyday]]
    
    news_up_to_workday_np = [np_pad_to_size(news,
                                            minsizes=(ntopbyday, None, None))
                             if len(news) > 0 else None for news in news_up_to_workday]
    np_news_all = np_pad_to_size(news_up_to_workday_np)
    assert np_news_all.ndim == 4
    assert np_news_all.shape[0:3] == (len(timeseries_align), ntopbyday, max_words_len)
    return np_news_all


def proc_marker(trainseries_align, csvname):
    """
    Reads, normalizes and processes a marker from policyuncertainity.
    Groups data based on trainseries_align with min and max functions.
    """
    markers_source = pd.read_csv(csvname,
                                 parse_dates=[['year', 'month', 'day']],
                                 date_parser=lambda x: pd.datetime.strptime(x, "%Y %m %d"))
    
    index_min = markers_source['daily_policy_index'].min()
    index_max = markers_source['daily_policy_index'].max()
    markers_source['daily_policy_index'] = (markers_source['daily_policy_index'] - index_min) / (index_max - index_min)
    
    markers_group_workday = group_by_timeseries(trainseries_align, markers_source, datecolumn='year_month_day')
    markers_group_workday = [[marker['daily_policy_index'] for marker in markers] for markers in markers_group_workday]
    markers_group_workday = [(min(days), max(days)) if len(days) > 0 else (0.5, 0.5)
                             for days in markers_group_workday]
    # there are in fact no NaNs, so the choice of 0.5 is not important.
    return np.array(markers_group_workday)


def proc_fred_marker(trainseries_align, csvname):
    """
    Load a csv dataset exported from FRED, normalizes, groups all datacolumns based on dates in trainseries_align
    with min and max functions.
    When missing value is encountered, fills it with the last known value in that column (and zero otherwise).

    Returns a numpy array of size (len(trainseries_align), 2* number of data columns in csvname)
    """
    markers_source = pd.read_csv(csvname, delimiter='\t',
                                 parse_dates=['DATE'], na_values=['.'], index_col='DATE',
                                 date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))
    
    index_min = markers_source.min()
    index_max = markers_source.max()
    markers_source = (markers_source - index_min) / (index_max - index_min)
    
    # forward fill (that means that values are propagated forward = we can see into the past, not into the future)
    markers_source.fillna(method='ffill', inplace=True)
    markers_source.fillna(value=0, inplace=True)
    
    markers_group_workday = group_by_timeseries(trainseries_align, markers_source, datecolumn=None)
    minmaxed_cols = []
    for column in markers_source.columns:
        col_proc = [[marker[column] for marker in markers] for markers in markers_group_workday]
        col_proc = [(min(days), max(days)) if len(days) > 0 else (None, None)
                    for days in col_proc]
        col_proc = pd.DataFrame(col_proc, columns=[column + "min", column + "max", ])
        minmaxed_cols.append(col_proc)
    
    markers_all = pd.concat(minmaxed_cols, axis=1)
    
    markers_all.fillna(method='ffill', inplace=True)
    markers_all.fillna(value=0.0, inplace=True)
    print("FRED markers columns: {}".format(markers_all.columns))
    return markers_all.to_numpy()


def download_to_file(url, file):
    r = requests.get(url, allow_redirects=True)
    open(file, 'wb').write(r.content)


def load_markers(trainseries_align, down=True):
    """
    Loads external markers previously downloaded and saved from
    https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv
    https://www.policyuncertainty.com/media/UK_Daily_Policy_Data.csv
    and FRED & concatenates.
    """
    if down:
        download_to_file("https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv",
                         "./economic_markers/All_Daily_Policy_Data.csv")
        download_to_file("https://www.policyuncertainty.com/media/UK_Daily_Policy_Data.csv",
                         "./economic_markers/UK_Daily_Policy_Data.csv")
        
    fredpopular = proc_fred_marker(trainseries_align, "./economic_markers/biggerlist_Daily.txt")
    usa = proc_marker(trainseries_align, "./economic_markers/All_Daily_Policy_Data.csv")
    uk = proc_marker(trainseries_align, "./economic_markers/UK_Daily_Policy_Data.csv")
    return np.concatenate([usa, uk, fredpopular], axis=-1)


def get_train_valid(data, np_news_all, np_markers_all,
                    predict_quantity,
                    data_total_start,
                    data_total_end,
                    train_startdate,
                    valid_startdate,
                    valid_enddate=None,
                    predict_change=True,
                    x_columns=None,
                    discrete_targets=False):
    """
    Normalizes Yahoo stock tickers data (optionally selects columns), turns prediction column into differences, adds
    info about dates, creates a batch dimension (being 1 because we are able to put everything into the memory)
     and returns a keras-like training and validation data (x, y, weights).
    The source data are named based on their origin.

    Includes experimental options for returning only the sign of the data for classification.

    """
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    dates_index = data_normalized.loc[data_total_start:data_total_end].index
    train_start = dates_index.get_loc(train_startdate)
    valid_start = dates_index.get_loc(valid_startdate)
    # this is the last available date in the data and it will not be included as source, only as a target
    # in other words - forget last datapoint because it has no prediction of next value
    if valid_enddate is not None:
        valid_end = dates_index.get_loc(valid_enddate)
    else:
        valid_end = len(dates_index) - 1
        
    test_start = valid_end + 1
    test_end = len(dates_index) - 1
    
    if x_columns:
        npdata_x = data_normalized.loc[data_total_start:data_total_end][x_columns].to_numpy()
    else:
        npdata_x = data_normalized.loc[data_total_start:data_total_end].to_numpy()
    
    pred_col = data_normalized.loc[data_total_start:data_total_end][predict_quantity].to_numpy()
    if predict_change:
        npdata_y = np.roll(pred_col[:, -1:], -1) - pred_col[:, -1:]  # change to the next day
    else:
        npdata_y = np.roll(pred_col[:, -1:], -1)  # quantity the next day
    
    if discrete_targets == 'sign':
        y_orig = npdata_y
        npdata_y = np.zeros(list(npdata_y.shape[:-1]) + [2 * npdata_y.shape[-1]])
        npdata_y[y_orig[:, 0] > 0.0, 1] = 1.0
        npdata_y[y_orig[:, 0] <= 0.0, 0] = 1.0
        assert np.sum(npdata_y) == npdata_y.shape[0]
        
        eval_info = y_orig[train_start:test_end]
    else:
        eval_info = npdata_y[train_start:test_end]
    
    npdata_dates_disc = get_date_infos_discrete(dates_index)
    npdata_dates = get_date_infos(dates_index)
    
    def make_train(arr):
        return np.expand_dims(arr[:valid_start], axis=0)  # makes a 1-batch and selects data for validation
    
    train = [
        {'inp_stock': make_train(npdata_x),
         'inp_markers': make_train(np_markers_all),
         'inp_dates': make_train(npdata_dates),
         'inp_news': make_train(np_news_all),
         'inp_dates_disc': make_train(npdata_dates_disc),
         },
        make_train(npdata_y), None
    ]
    # do not propagate gradients for data, that are present only for historic lookups
    train_y_weights = np.zeros((1, train[1].shape[1],))
    train_y_weights[0, train_start:valid_start] = 1.0
    train[2] = train_y_weights
    
    def make_val(arr):
        return np.expand_dims(arr[train_start:valid_end], axis=0)  # makes a 1-batch and selects data for validation
    
    val = [
        {'inp_stock': make_val(npdata_x),
         'inp_markers': make_val(np_markers_all),
         'inp_dates': make_val(npdata_dates),
         'inp_news': make_val(np_news_all),
         'inp_dates_disc': make_val(npdata_dates_disc),
         },
        make_val(npdata_y), None
    ]
    # do not evaluate based on data, that are present only for historic lookups
    val_y_weights = np.zeros((1, val[1].shape[1],))
    val_y_weights[0, (valid_start - train_start):] = 1.0
    val[2] = val_y_weights

    if test_start < test_end:
        def make_test(arr):
            return np.expand_dims(arr[train_start:test_end], axis=0)  # makes a 1-batch and selects data for validation
    
        test = [
            {'inp_stock': make_test(npdata_x),
             'inp_markers': make_test(np_markers_all),
             'inp_dates': make_test(npdata_dates),
             'inp_news': make_test(np_news_all),
             'inp_dates_disc': make_test(npdata_dates_disc),
             },
            make_test(npdata_y), None
        ]
        # do not evaluate based on data, that are present only for historic lookups
        test_y_weights = np.zeros((1, test[1].shape[1],))
        test_y_weights[0, (test_start - train_start):] = 1.0
        test[2] = test_y_weights
    else:
        print("using valid as test set.")
        test = val
    
    
    assert all([val[0][arr].shape[0:2] == val[0]['inp_stock'].shape[0:2] for arr in val[0]]), \
        "All training data should have the same batch and time dimensions"
    assert val[1].shape[0:2] == val[0]['inp_stock'].shape[0:2], "target data should have the same dimensionality"
    
    assert all([train[0][arr].shape[0:2] == train[0]['inp_stock'].shape[0:2] for arr in train[0]]), \
        "All training data should have the same batch and time dimensions"
    assert train[1].shape[0:2] == train[0]['inp_stock'].shape[0:2], "target data should have the same dimensionality"
    
    return train, val, test, eval_info


def characterize_bin_classification(pred_y, real_y):
    """
    Prints detailed info for binary classification.
    """
    report = skmetrics.classification_report(real_y, pred_y)
    df_confusion = skmetrics.confusion_matrix(real_y, pred_y)
    report += "\n" + str(df_confusion)
    report += "\n" + str("Accuracy: {}".format(float(sum([df_confusion[i, i] for i in range(len(df_confusion))], 0.0))
                                               / float(sum(df_confusion.flatten(), 0.0))))
    return report


def print_binclassification(pred_y, real_y):
    """
    Prints detailed info for binary classification.
    """
    print(characterize_bin_classification(pred_y, real_y))


def predict_apply_weights(model, x, truth, weights):
    """
    Produces prediction and goldstandard based on input, model and then selects only relevant data based on weights.
    """
    predicted_y = model.predict(x, batch_size=1)
    pred_y = predicted_y[0][weights > 0]
    real_y = truth[weights > 0]
    return pred_y, real_y


def sse(pred_y, real_y):
    err = pred_y - real_y
    err = err * err
    return np.sum(err)


def eval_predictions(model, test, eval_orig_y, discrete_preds,
                     model_name
                     ):
    """
    Predicts given model on given sets and analyzes performance.
    """
    pred_y, real_y = predict_apply_weights(model, test[0], test[1][0], test[2][0])
    
    print(real_y.shape, pred_y.shape)
    
    if discrete_preds:
        val_y_weights = test[2][0]
        r = eval_orig_y[val_y_weights > 0, -1]
        preds_positive = (pred_y[:, -1] > 0.5).astype(int)
        print_binclassification(preds_positive, real_y[:, -1].astype(int))
        # potential - then calibrate based on different times, this tells us just the maximum where we can get
        print("the potential where we can get (if calibrated based on validation set) is: ")
        p = np.mean(r[preds_positive])
        n = np.mean(r[~preds_positive])
        print(p, n)
        
        preds_calibrated = preds_positive * p + (1 - preds_positive) * n
        
        plot_preds(preds_calibrated, r, model_name)
    else:
        plot_preds(pred_y, real_y, model_name)


def cached_op(filename, function):
    """
    Reads numpy array fro ma cache if exists.
    Saves a function's results to a cache if does not exist.
    """
    if filename and os.path.exists(filename):
        results = np.load(filename)
    else:
        results = function()
        if filename:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            #pathlib.Path(filename).mkdir(parents=True, exist_ok=True)
            np.save(filename, results)
    return results


def run_inputs_corrupt_analysis_percent(model, valid, datasources_to_corrupt=None,
                                        corrupt_value=0.0,
                                        percents_steps=100, tries_per_experiment=3,
                                        percent_divisor=None):
    """
    Randmly corrupts valid data specified in datasources_to_corrupt (if none, uses all the keys in valid[0]) by
    corrupt_value (can be a sklearn's random distribution).
    Proceeds with number of steps in percents_steps, each meaning a (step / percent_divisor) fraction of data to corrupt.
    Each case is executed tries_per_experiment times and recorded in a returned array of size
    [percent_steps, tries_per_experiment].
    """
    if percent_divisor is None:
        percent_divisor = percents_steps
    results = defaultdict(list)
    for percent, itry in tqdm(product(range(percents_steps), range(tries_per_experiment)),
                              total=tries_per_experiment * percents_steps):
        _x = deepcopy(valid[0])
        for key in (_x if datasources_to_corrupt is None else datasources_to_corrupt):
            size = int(_x[key].size * (percent / float(percent_divisor)))
            indices = np.random.choice(np.arange(_x[key].size), replace=False,
                                       size=size)
            if isinstance(corrupt_value, numbers.Number):
                _x[key][np.unravel_index(indices, _x[key].shape)] = corrupt_value
            else:
                _x[key][np.unravel_index(indices, _x[key].shape)] = corrupt_value.rvs(size=size)
        pred_y, real_y = predict_apply_weights(model, _x, valid[1][0], valid[2][0])
        results[percent].append(sse(pred_y, real_y))
    
    return np.array([results[i] for i in range(percents_steps)])


def run_inputs_corrupt_analysis(model, valid, datasource_to_corrupt=None,
                                        corrupt_value=0.0,
                                        tries_per_experiment=3,  # important only when corrupt value is random varaible
                                        group_conseq_inputs=2
                                        ):
    """
    Corrupts only a specified columns datasource_to_corrupt in the sources.
    As our data are groupped by 2 columns per marker (min and max), we need to corrupt both of them
    (coded in the parameter group_conseq_inputs).
    """
    assert valid[0][datasource_to_corrupt].shape[-1] % group_conseq_inputs == 0
    ntrials = int(valid[0][datasource_to_corrupt].shape[-1] / group_conseq_inputs)
    results = defaultdict(list)
    for i, itry in tqdm(product(range(ntrials), range(tries_per_experiment)),
                              total=tries_per_experiment * ntrials):
        _x = deepcopy(valid[0])
        if isinstance(corrupt_value, numbers.Number):
            _x[datasource_to_corrupt][(group_conseq_inputs*i):(group_conseq_inputs*(i+1))] = corrupt_value
        else:
            size_to_generate = list(valid[0][datasource_to_corrupt].shape[0:-1]) + [group_conseq_inputs]
            _x[datasource_to_corrupt][(group_conseq_inputs*i):
                                      (group_conseq_inputs*(i+1))] = corrupt_value.rvs(size=size_to_generate)
        pred_y, real_y = predict_apply_weights(model, _x, valid[1][0], valid[2][0])
        results[i].append(sse(pred_y, real_y))
    
    return np.array([results[i] for i in range(ntrials)])


def analyze_specific_input_corruption(model, valid, corrupt_value,
                       cache_name,
                       model_name,
                       plot_name,
                       model_complet, baseline,
                       datasource_to_corrupt=None,
                        group_conseq_inputs=2,
                       tries_per_experiment=1
                       ):
    results = cached_op(cache_name + plot_name +'.npy' if cache_name is not None else None,
                        lambda: run_inputs_corrupt_analysis(model, valid,
                                                            datasource_to_corrupt=datasource_to_corrupt,
                                                            corrupt_value=corrupt_value,
                                                            group_conseq_inputs=group_conseq_inputs,
                                                            tries_per_experiment=tries_per_experiment))

    relative_change = (results - model_complet) / (baseline-model_complet)
    np.savetxt(model_name + plot_name + "markers_dep.csv", relative_change, delimiter=";")

    """
    from here on the analysis is hardcoded and not automatic - for example since it has discovered the
    zeroth input of markers to be the most important, we will plot it.
    """

    _x = deepcopy(valid[0])
    _x['inp_markers'][0:2] = 0.0
    pred_y, real_y = predict_apply_weights(model, _x, valid[1][0], valid[2][0])
    plot_preds(pred_y, real_y, model_name + plot_name + "0-inpmarkers_corruption")
    
    return results


def analyze_corruption(model, valid, corrupt_value,
                       cache_name,
                       model_name,
                       plot_name,
                       model_complet, baseline,
                       datasources_to_corrupt=None,
                       percents_steps=100,
                       percent_divisor=100,
                       tries_per_experiment=3
                       ):
    results = cached_op(cache_name + plot_name +'.npy' if cache_name is not None else None,
                        lambda: run_inputs_corrupt_analysis_percent(model, valid,
                                                                    corrupt_value=corrupt_value,
                                                                    datasources_to_corrupt=datasources_to_corrupt,
                                                                    percents_steps=percents_steps,
                                                                    percent_divisor=percent_divisor,
                                                                    tries_per_experiment=tries_per_experiment))
    
    results_x = [[i] * results.shape[-1] for i in range(results.shape[-2])]
    
    #find first evidence when all the sse's are higher than model_complet
    corrupt_result = 0
    significant_change = (baseline - model_complet) * 0.05
    for sses, x in zip(results, results_x):
        if np.all(sses > model_complet + significant_change):
            corrupt_result = x[0] / percent_divisor
            break
    
    results = np.array(results).flatten()
    results_x = np.array(results_x).flatten()
    
    # linear correlation between predictions and outputs
    fig, ax = plt.subplots(figsize=(16, 16))
    plt.title('SSE by {} corruption, significant at {}'.format(
        datasources_to_corrupt if datasources_to_corrupt else "data", corrupt_result), fontsize=30)
    # plt.axhline(y=0.0, color='black', linestyle='-')
    plt.axhline(y=model_complet, color='orange', linestyle='-')
    plt.axhline(y=baseline, color='blue', linestyle='-')
    plt.plot([], color='orange', label='original model sse')
    plt.plot([], color='blue', label='baseline sse')
    plt.scatter(results_x, results)
    fitl = [float(x) for x in np.polyfit(results_x, results, 1)]
    plt.plot(results_x, fitl[0] * results_x + fitl[1], color='red', linewidth=2,
             label='correlation: y={:.6f}+{:.6e}*x'.format(fitl[1], fitl[0]))
    plt.legend(loc='upper left', fontsize=30)
    plt.ylabel('SSE model score', fontsize=30)
    plt.xlabel('Data corruption percentage', fontsize=30)
    plt.savefig(model_name + plot_name + '.png')
    plt.show()
    return corrupt_result


def analyze_model(model, valid, cache_name=None,
                  train=None, model_name=None):
    pred_y, real_y = predict_apply_weights(model, valid[0], valid[1][0], valid[2][0])
    train_pred_y, train_real_y = predict_apply_weights(model, train[0], train[1][0], train[2][0])
    model_complet = sse(pred_y, real_y)
    baseline = sse(0, real_y)
    
    print("lets look at the training data")
    plot_preds(train_pred_y, train_real_y, model_name + '_train')
    
    print("is the model robust to corrupt data ? (set some percentage to zero)")
    analyze_corruption(model, valid, cache_name=cache_name, model_name=model_name, plot_name="_corrupt_0",
                       model_complet=model_complet, baseline=baseline,
                       corrupt_value=0.0)
    
    print("is the model robust to more oscilating data ? (set some percentage to +-0.5)")
    analyze_corruption(model, valid, cache_name=cache_name, model_name=model_name, plot_name="_corrupt_high",
                       model_complet=model_complet, baseline=baseline,
                       corrupt_value=conditioned_continuous(
                                            st.rv_discrete(values=([0, 1], [0.5, 0.5])),
                                            [st.norm(loc=-0.5, scale=0.1), st.norm(loc=0.5, scale=0.1)]))
    
    print("is the model robust to extremal data ? (set some percentage to +-1.0)")
    analyze_corruption(model, valid, cache_name=cache_name, model_name=model_name, plot_name="_corrupt_max",
                       model_complet=model_complet, baseline=baseline,
                       corrupt_value=st.rv_discrete(values=([-1.0, 1.0], [0.5, 0.5])))

    # since the maximal data corruption is the most effective, lets use it and see how each input is significant:
    # and the 5% error was achieved at ~18%, lets explore only till 30%:
    for datasource_to_corrupt in valid[0]:
        print("is the model input {} robust to extremal data ? (set some percentage to +-1.0)".format(datasource_to_corrupt))
        analyze_corruption(model, valid, cache_name=cache_name, model_name=model_name, plot_name="_corrupt_max"+datasource_to_corrupt,
                           model_complet=model_complet, baseline=baseline,
                           corrupt_value=st.rv_discrete(values=([-1.0, 1.0], [0.5, 0.5])),
                           datasources_to_corrupt=[datasource_to_corrupt],
                           percents_steps=30,
                           percent_divisor=100,
                           tries_per_experiment=3
                           )
        
    print("importance of inp_markers:")
    analyze_specific_input_corruption(model, valid, corrupt_value=0.0,
                                      cache_name=cache_name,
                                      model_name=model_name,
                                      plot_name="_input_markers_specific_",
                                      model_complet=model_complet, baseline=baseline,
                                      datasource_to_corrupt='inp_markers',
                                      group_conseq_inputs=2,
                                      tries_per_experiment=1
                                      )
    
    grads = evaluate_grad_dep(model, x=valid[0], y=valid[1],
                      batch_size=1,
                      verbose=1,
                      sample_weight=valid[2],
                      )
    # save gradients for manual inspections:
    for input_name, grad_vals in zip(valid[0], grads):
        if input_name == "inp_news":
            # for news input save just the summed grads over the top stories
            np.save(model_name + input_name + "_grads.npy", np.sum(grads[3], axis=(-1, -2)))
        else:
            np.save(model_name + input_name + "_grads.npy", grad_vals[0])


def plot_preds(pred_y, real_y, model_name):
    """
    Produces prediction and error plots (with regression) based on given predictions and goldstandards.
    """
    SMALLER_PLOT_DESCRIPTION = 20
    
    model_sse = sse(pred_y, real_y)
    baseline_sse = sse(0, real_y)
    print(model_sse)
    print("(pred=0 => {})".format(baseline_sse))
    print("only regarded as positive/negative predictions:")
    classification_eval = characterize_bin_classification((pred_y > 0).astype(int), (real_y > 0).astype(int))
    classification_eval = "Signum only as a binary classification problem: \n" + classification_eval
    print(classification_eval)
    
    fig, ax = plt.subplots(figsize=(18, 16))
    plt.title('Model prediction vs ground truth', fontsize=30)
    plt.axhline(y=0.0, color='black', linestyle='-')
    plt.plot(real_y, label='Truth', color='orange')
    plt.plot([], color='black', label='Baseline sse: {}'.format(baseline_sse))
    plt.plot(pred_y, label='Prediction sse: {}'.format(model_sse), linewidth=1, color='blue')
    plt.ylabel('prediction', fontsize=30)
    plt.xlabel('time', fontsize=30)
    plt.legend(loc='upper left', fontsize=30)
    ax.text(0.05, 0.30, classification_eval, transform=ax.transAxes, fontsize=SMALLER_PLOT_DESCRIPTION,
            verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})
    plt.savefig(model_name + '_pred_truth.png')
    plt.show()
    
    fig, ax = plt.subplots(figsize=(18, 16))
    plt.title('Model prediction', fontsize=30)
    plt.axhline(y=0.0, color='black', linestyle='-')
    plt.plot(pred_y, label='Prediction')
    plt.ylabel('prediction', fontsize=30)
    plt.xlabel('time', fontsize=30)
    plt.legend(loc='upper left', fontsize=30)
    plt.savefig(model_name + '_pred.png')
    plt.show()
    
    baseline_error = real_y * real_y
    model_error = (real_y - pred_y) * (real_y - pred_y)
    print(model_error.shape)
    
    fig, ax = plt.subplots(figsize=(18, 16))
    ex = np.arange(model_error.shape[0])
    e_fit = [float(x) for x in np.polyfit(ex, model_error, 1)]
    # print("linear regression of error: y={}+{:}*x".format(e_fit[1], e_fit[0]))
    ex2 = np.arange(real_y.shape[0])
    e_fit2 = [float(x) for x in np.polyfit(ex2, baseline_error, 1)]
    # print("linear regression of 0-pred error: y={}+{:}*x".format(e_fit2[1], e_fit2[0]))
    plt.title('Errors in time', fontsize=30)
    plt.axhline(y=0.0, color='black', linestyle='-')
    plt.plot(model_error, label='squared error')
    plt.plot(baseline_error - model_error, label='squared error diff baseline - model')
    plt.plot(ex, e_fit[0] * ex + e_fit[1], color='red', linewidth=2,
             label='error: y={:.6f}+{:.6e}*x'.format(e_fit[1], e_fit[0]))
    plt.plot(ex, e_fit2[0] * ex + e_fit2[1], color='black', linewidth=2,
             label='baseline error: y={:.6f}+{:.6e}*x'.format(e_fit2[1], e_fit2[0]))
    # plt.plot([], [], ' ', label="Extra label on the legend")
    plt.ylabel('error', fontsize=30)
    plt.xlabel('time', fontsize=30)
    plt.legend(loc='upper left', fontsize=30)
    plt.savefig(model_name + '_errors.png')
    plt.show()

    ex3 = np.arange(model_error.shape[0])
    e_fit3 = [float(x) for x in np.polyfit(ex3, baseline_error - model_error, 1)]
    fig, ax = plt.subplots(figsize=(18, 16))
    plt.title('Difference of errors', fontsize=30)
    plt.axhline(y=0.0, color='black', linestyle='-')
    plt.plot(baseline_error - model_error, label='squared error diff baseline - model')
    plt.plot(ex, e_fit3[0] * ex + e_fit3[1], color='red', linewidth=2,
             label='diff error: y={:.6f}+{:.6e}*x'.format(e_fit3[1], e_fit3[0]))
    plt.ylabel('error', fontsize=30)
    plt.xlabel('time', fontsize=30)
    plt.legend(loc='upper left', fontsize=30)
    plt.savefig(model_name + '_errors_diff.png')
    plt.show()
    
    # linear correlation between predictions and outputs
    fig, ax = plt.subplots(figsize=(18, 16))
    plt.title('Linear correlation between predictions and ground truth', fontsize=30)
    plt.axhline(y=0.0, color='black', linestyle='-')
    plt.scatter(real_y, pred_y)
    fitl = [float(x) for x in np.polyfit(real_y[:, -1], pred_y[:, -1], 1)]
    plt.plot(real_y[:, -1], fitl[0] * real_y[:, -1] + fitl[1], color='red', linewidth=2,
             label='correlation: y={:.6f}+{:.6e}*x'.format(fitl[1], fitl[0]))
    plt.legend(loc='upper left', fontsize=30)
    plt.ylabel('prediction', fontsize=30)
    plt.xlabel('ground truth', fontsize=30)
    plt.savefig(model_name + '_corr_pred_truth.png')
    plt.show()


def derivative_wrt_inputs(model):
    inputs = (model._feed_inputs +
              model._feed_targets +
              model._feed_sample_weights)
    if model._uses_dynamic_learning_phase():
        inputs += [K.learning_phase()]
    # Return loss and metrics, no gradient updates.
    # Does update the network states.
    grads = K.gradients(model.total_loss, model.input)
    dep_function = K.function(
        inputs,
        grads,
        name='dep_function',
        **model._function_kwargs)
    return dep_function


from keras.engine import training_arrays
def evaluate_grad_dep(model, x=None, y=None,
             batch_size=None,
             verbose=1,
             sample_weight=None,
             steps=None):
    if x is None and y is None and steps is None:
        raise ValueError('If evaluating from data tensors, '
                         'you should specify the `steps` '
                         'argument.')
    # Validate user data.
    x, y, sample_weights = model._standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        batch_size=batch_size)
    # Prepare inputs, delegate logic to `test_loop`.
    if model._uses_dynamic_learning_phase():
        ins = x + y + sample_weights + [0.]
    else:
        ins = x + y + sample_weights
    f = derivative_wrt_inputs(model)
    return training_arrays.test_loop(model, f, ins,
                                     batch_size=batch_size,
                                     verbose=verbose,
                                     steps=steps)


def mse_with_sign_categorical(y_true, y_pred):
    """
    Optionally forces the mse to pay bigger attention to the sign of predictions.
    """
    squared_error = K.square(y_pred - y_true)
    mse = K.mean(squared_error, axis=-1)
    
    truth_positive = K.cast(K.greater(y_true, 0.0), dtype=float)
    pred_positive = K.cast(K.greater(y_pred, 0.0), dtype=float)
    
    bin_crossentropy = K.mean(K.binary_crossentropy(truth_positive, pred_positive), axis=-1)
    
    return mse + bin_crossentropy


def wavenetBlock(neurons, filter_size, dilation_rate):
    def f(input_):
        residual = input_
        tanh_out = Conv1D(neurons, filter_size,
                                 dilation_rate=dilation_rate,
                                 padding='causal',
                                 activation='tanh')(input_)
        sigmoid_out = Conv1D(neurons, filter_size,
                                    dilation_rate=dilation_rate,
                                    padding='causal', #border_mode='same',
                                    activation='sigmoid')(input_)
        merged = Multiply()([tanh_out, sigmoid_out])
        skip_out = LeakyReLU()(Conv1D(1, 1, padding='causal')(merged))
        out = Add()([skip_out, residual])
        return out, skip_out
    
    return f


def load_our_model(model_name):
    return load_model(model_name, custom_objects={"SinCosPositionalEmbedding": SinCosPositionalEmbedding,
                                           "tf_lookback": tf_lookback})


def load_weights_as_possible(model, start_weights_from):
    """
    Load weights with same layer names if possible.
    """
    try:
        model.load_weights(start_weights_from, by_name=True,
                     skip_mismatch=True, reshape=False)
    except Exception as e:
        xmodel = load_our_model(start_weights_from)
        xdict = {}
        for layer in xmodel.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                xdict[layer.name] = weights
        for layer in model.layers:
            if layer.name in xdict:
                try:
                    layer.set_weights(xdict[layer.name])
                except Exception as e:
                    pass


def run_experiment(model_name, train, valid, discrete_preds=False,
                   start_weights_from=None):
    """
    
    """
    inp_stock = Input(name='inp_stock', shape=[None, train[0]['inp_stock'].shape[-1]])
    inp_markers = Input(name='inp_markers', shape=[None, train[0]['inp_markers'].shape[-1]])
    inp_dates = Input(name='inp_dates', shape=[None, train[0]['inp_dates'].shape[-1]])
    inp_dates_disc = Input(name='inp_dates_disc', shape=[None, train[0]['inp_dates_disc'].shape[-1]],
                           dtype='float32')
    dates_embedded = SinCosPositionalEmbedding(4, from_inputs_features=[0,1,2],
                                               embeddings=['sin', 'cos', 'lin'])(inp_dates)
    inp_news = Input(name='inp_news', shape=[None] + list(train[0]['inp_news'].shape[2:]))
    
    conv_news_r = LeakyReLU()(Conv3D(50, kernel_size=(1, 1, 5), padding="valid")(inp_news))
    conv_news_r2 = LeakyReLU()(Conv3D(50, kernel_size=(1, 1, 5), padding="valid", dilation_rate=2)(conv_news_r))
    pool_over_words = Lambda(lambda u: K.concatenate([K.max(u, axis=-1), K.mean(u, axis=-1)], axis=-1))(conv_news_r2)
    conv_news = LeakyReLU()(Conv2D(50, kernel_size=(1, 5), padding="same")(pool_over_words))
    pool_over_news = Lambda(lambda u: K.concatenate([K.max(u, axis=-1), K.mean(u, axis=-1)], axis=-1))(conv_news)
    news_drp = Dropout(0.15)(pool_over_news)
    
    inp_all = Concatenate(axis=-1)([inp_stock, inp_markers, dates_embedded, inp_dates_disc])
    yshift = 251
    wshift = 5
    years_shifts = [Lambda(lambda item: tf_lookback(item, i * yshift))(inp_all) for i in
                    range(1, 3)]  # a bit of hyperparameter tuning - 7 years back do the trick us!
    weeks_shifts = [Lambda(lambda item: tf_lookback(item, i * wshift))(inp_all) for i in range(1, 4)]
    all_and_before = Concatenate(axis=-1)([inp_all] + years_shifts + weeks_shifts)
    
    # wavenet architecture:
    all_info = Concatenate(axis=-1)([all_and_before, news_drp])
    A, B = wavenetBlock(64, 2, 2)(all_info)
    skip_connections = [B]
    for i in range(20):
        A, B = wavenetBlock(64, 2, 2 ** ((i + 2) % 9))(A)
        skip_connections.append(B)
    net = LeakyReLU()(Add()(skip_connections))
    net = LeakyReLU()(Conv1D(400, 1, padding='causal')(net))
    net = Dropout(0.45)(net)
    output = Dense(train[1].shape[-1])(net)
    
    model = Model(inputs=[inp_stock, inp_markers, inp_dates, inp_news, inp_dates_disc], outputs=[output])
    model.summary()
    if discrete_preds:
        model.compile(loss='categorical_crossentropy', optimizer='adam', sample_weight_mode="temporal",
                      metrics=['categorical_accuracy'])
        # care, the number printed by metrics is without applied the weights provided!
    else:
        model.compile(loss='mse', optimizer='adam', sample_weight_mode="temporal",
                      metrics=['mse'])
    
    callbacks = [ModelCheckpoint(model_name,
                                 monitor='val_loss', save_best_only=True, mode='min',
                                 verbose=True),
                 EarlyStopping(monitor='val_loss', patience=60,
                               mode='min', verbose=True)]
    
    if start_weights_from is not None:
        print("Loading weights from previous model {}".format(start_weights_from))
        load_weights_as_possible(model, start_weights_from)
    
    hist = model.fit(x=train[0], y=train[1], sample_weight=train[2],
                     batch_size=1, epochs=200,
                     verbose=1,
                     callbacks=callbacks,
                     validation_data=valid,
                     shuffle=False,
                     )
    
    # Plot training & validation loss values from second epoch, because loss for first one can be unnecessarily high
    plt.plot(hist.history['loss'][1:])
    plt.plot(hist.history['val_loss'][1:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(model_name + '_losses.png')
    plt.show()
    
    import os
    if os.path.exists(model_name):
        model.load_weights(model_name)
    else:
        model.load_weights('best-' + model_name)
    return model


def common_data(discrete_targets=False,
                    data_total_start='1999-01-01',
                    data_total_end='2020-03-20',
                    train_startdate='2000-01-03',
                    valid_startdate='2017-01-03',
                    valid_enddate='2019-12-31',
                    predict_quantity=[('Volume', '^GSPC')],
                ):
    # load stocks:
    data = pdr.get_data_yahoo(['^DJI', '^IXIC', '^RUT', '^GSPC'], start=data_total_start, end=data_total_end)
    data.columns = data.columns.to_flat_index()
    assert sum(data.isna().sum()) <= 0, "select different tickers or care for nans..."
    
    glove_embeddings = load_glove()
    
    np_markers_all = load_markers(data.index)
    
    np_news_all = load_reddit_news(glove_embeddings, data.index)
    
    train, valid, test, eval_orig_y = get_train_valid(data, np_news_all, np_markers_all,
                                                discrete_targets='sign' if discrete_targets else False,  # pep
                                                predict_quantity=predict_quantity,
                                                data_total_start=data_total_start,
                                                data_total_end=data_total_end,
                                                train_startdate=train_startdate,
                                                valid_startdate=valid_startdate,
                                                valid_enddate=valid_enddate,
                                                )
    return train, valid, test, eval_orig_y


@click.command()
@click.option('--model_name', default="wavenet.h5",
              help='The models file name to save',
              )
@click.option('--discrete_targets', default=False,
              help='Set to true for experimenting with discrete targets')
@click.option('--start_weights_from', default=None,
              help='Start the training from weights file')
@click.option('--force_rewrite', default=False,
              help='Start the training from weights file')
def cmd_train(model_name, discrete_targets, start_weights_from, force_rewrite):
    """
    Trains and saves a model. Plots basic graphs.
    """
    if os.path.exists(model_name) and not force_rewrite:
        raise ValueError("Model already exists!")  # no overwritting
    train, valid, test, eval_orig_y = common_data(discrete_targets=discrete_targets)

    if not os.path.exists(os.path.dirname(model_name)):
        os.makedirs(os.path.dirname(model_name))
    model = run_experiment(model_name, train, valid, discrete_preds=discrete_targets,
                           start_weights_from=start_weights_from)
    eval_predictions(model, test, eval_orig_y, discrete_targets, model_name)


@click.command()
@click.option('--model_name', default="wavenet1.4488.h5",
              help='The models file name to save',
              )
@click.option('--discrete_targets', default=False,
              help='Set to true for experimenting with discrete targets')
@click.option('--cache_name', default="/cached/a",
              help='Set to true for experimenting with discrete targets')
def cmd_eval(model_name, discrete_targets, cache_name):
    """
    Analyses a model, produces and shows all graphs and caches all results.
    """
    train, valid, test, eval_orig_y = common_data(discrete_targets=discrete_targets)
    
    model = load_our_model(model_name)
    
    eval_predictions(model, test, eval_orig_y, discrete_targets, model_name)

    analyze_model(model, test, train=train,
                  cache_name=cache_name, model_name=model_name)


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


@click.command()
@click.option('--model_name', default=None,
              help='The model name, where the gradients are saved from cmd_eval.',
              )
def cmd_analyse_grads(model_name):
    """
    Non automated analysis of saved gradients to allow additional exploration of model's dependencies.
    
    Currently plots two plots used in the report file.
    """
    grads_files_list = ["inp_news_grads", "inp_stock_grads", "inp_markers_grads",
                   "inp_dates_grads","inp_dates_disc_grads",]
    grads_list = [np.load(model_name + grads_file + ".npy").squeeze() for grads_file in grads_files_list]

    by_days = [np.stack((normalize(np.max(np.abs(grads), axis=-1)),
                normalize(np.sum(np.abs(grads), axis=-1))), axis=-1)
               for grads in grads_list]
    by_features = [np.stack((normalize(np.max(np.abs(grads), axis=-2)),
                normalize(np.sum(np.abs(grads), axis=-2))), axis=-1)
               for grads in grads_list]

    by_days_x = np.arange(0, by_days[0].shape[0])
    by_days_all = sum(by_days)
    
    #by_days_summed = sum()
    

    start = 0
    for i in range(by_days_all.shape[0]):
        if by_days_all[i, 0] > 0 or by_days_all[i, 1] > 0:
            start = i
            break
    
    fig, ax = plt.subplots(figsize=(18, 16))
    plt.title('Normalized absolute gradient contributions by days', fontsize=30)
    plt.plot(by_days_x[start:], by_days_all[start:, 0], label='max', color='red')
    plt.plot(by_days_x[start:], by_days_all[start:, 1], label='sum', color='blue')
    plt.ylabel('gradient size', fontsize=30)
    plt.xlabel('time', fontsize=30)
    plt.legend(loc='upper left', fontsize=30)
    plt.savefig(model_name + "inp_all_grads_days" + '.png')
    plt.show()

    for inp_name, feature_grads in zip(grads_files_list, by_features):
        # by_features_x = np.arange(0, len(by_features[0][0]))
        fig, ax = plt.subplots(figsize=(18, 16))
        plt.title('Gradients by features '+inp_name, fontsize=30)
        plt.plot(feature_grads[:, 0], label='max', color='red',  marker='o')
        plt.plot(feature_grads[:, 1], label='mean', color='blue',  marker='o')
        plt.ylabel('gradient size', fontsize=30)
        plt.xlabel('feature', fontsize=30)
        plt.legend(loc='upper left', fontsize=30)
        plt.savefig(model_name + inp_name + '_grads.png')
        plt.show()


@click.group()
def clisap():
    pass

clisap.add_command(cmd_eval)
clisap.add_command(cmd_train)
clisap.add_command(cmd_analyse_grads)

if __name__ == "__main__":
    clisap()
