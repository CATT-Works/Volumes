import sys
import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import r2_score
from sklearn.utils import shuffle

from time import time, strftime, gmtime

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


from prepare_data import get_XY

def normalize(hp, train, test = None, save_scaler = False): 
    """
    Normalization with using StandardScaller() from sklearn
    Arguments:
        hp          - object with hyperparameters
        train, test - features to be normalized (test is optional)
        save_scaler - if True than scaler is saved based on hp.FILECORE_PATH variable
    Returns:
        features_train, features_test - normalized features
    """
    scaler = StandardScaler().fit(train.astype(np.float64))
    features_train = scaler.transform(train.astype(np.float64))
    
    if save_scaler:
        pickle.dump(scaler, open('{}_scaler.p'.format(hp.FILECORE_PATH), 'wb'))

    
    if test is not None:
        features_test = scaler.transform(test.astype(np.float64))    
        return features_train, features_test
    else:
        return features_train    
    
def get_basic_model(hp, n_inputs):
    inputs = tf.keras.layers.Input(shape=(n_inputs,), name='Inputs')
    
    i = 0
    for cell, batchnorm, dropout in zip(hp.CELLS, hp.BATCHNORM, hp.DROPOUT):
        i += 1
        if i == 1:
            deep = Dense(cell, name='Dense_{}'.format(i)) (inputs)
        else:
            deep = Dense(cell, name='Dense_{}'.format(i)) (deep)
        
        if batchnorm:
            deep = BatchNormalization(name='BatchNorm_{}'.format(i)) (deep)        
        
        deep = Activation('elu', name='Elu_{}'.format(i)) (deep)

        if dropout > 0:
            deep = Dropout(dropout, name='Dropout_{}'.format(i)) (deep)
        
        
    if hp.DEEPANDWIDE:
        both = tf.keras.layers.concatenate([deep, inputs], name='WideAndDeep')    
        output = Dense(1, name='Output') (both)
    else:
        output = Dense(1, name='Output') (deep)
        
    model = tf.keras.Model(inputs, output)
    return model

def create_model(hp, n_inputs, gpus = None):
    """
    Creates a fully connected model.
    Arguments:
        hp          - object with hyperparameters
        n_inputs - number of inputs (shape[1] of features vector)
        gpus     - if not none (and >=2) multi_gpu_model is created
    Returns:
        model - craeted and compiled model
    """    
    
    
    
    if gpus is not None and gpus >= 2:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = get_basic_model(hp, n_inputs)
            model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
            opt = optimizers.Adam(lr = hp.LEARNING_RATE, decay = hp.DECAY)
            model.compile(loss='mae', optimizer=opt)
        return model, strategy
    else:
        model = get_basic_model(hp, n_inputs)
        opt = optimizers.Adam(lr = hp.LEARNING_RATE, decay = hp.DECAY)
        model.compile(loss=hp.LOSS, optimizer=opt)        
        return model, None
    

def plot_train_valid_test(hp, train, valid, beautiful_plot = False, save_path = None):
    """
    Plots train and valid losses.
    Arguments: 
        train, valid - losses (output from model.fit)
        beatuiful_plot - if True, the beautiful version of the plot is produced
        save_path      - path to save the plot (default = None)
    """
    epochs = np.arange(len(train)) + 1
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(epochs, train, c='b', label='Train loss')
    ax1.plot(epochs, valid, c='g', label='Valid loss')
    #ax1.set_ylim([0,1])
    
    if beautiful_plot:
        plt.gcf().subplots_adjust(bottom=0.20, left=0.20)
        plt.legend(loc='upper right', fontsize=FONTSIZE);
        plt.xlabel('Epochs', fontsize=FONTSIZE+2)
        plt.ylabel('Loss', fontsize=FONTSIZE+2)
        if hp.EPOCHS > 15:
            ticks = np.arange(5, hp.EPOCHS + 1, 5)
        elif hp.EPOCHS > 10:
            ticks = np.arange(2, hp.EPOCHS + 1, 2)
        else:
            ticks = np.arange(1, hp.EPOCHS + 1, 1)
            
        ax1.set_xticks(ticks)
        plt.xticks(fontsize=FONTSIZE)
        plt.yticks(fontsize=FONTSIZE)    

        plt.grid(True)
    else:
        plt.legend(loc='lower left');
        plt.grid(True)
        plt.show()
        
    if save_path is not None:
        plt.savefig(save_path)    

    plt.show()

def train_test_station_split(hp, df, station):
    if hp.USE_ATR:
        col = "count_location"
    else:
        col = "tmc"
    df_train = df.loc[df[col] != station].copy()
    df_test = df.loc[df[col] == station].copy()

    return df_train, df_test

def prepare_traintest_data(hp, df, station):
    """
    Splits data by given station.
    Arguments:
        hp      - object with hyperparameters
        df      - dataset with features and labels
        station - station that should be placed in the test dataset
    Returns:
        df_train, df_test - dataframes with train and test dataset
        train_X, train_y  - train features and labels
        test_X, test_y    - test features and labels
        
    NOTE: In this version this code is also responsible for reducing the size of train dataset
    The size is reduced if hp.REDUCED_TRAIN is True. The starting date is encoded as the last
    five characters of hp.FILENAME (for example ....09-15 means that everything before 
    09/15/2017 is deleted from train dataset (test_dataset remaies complete)
    """
    df_train, df_test = train_test_station_split(hp, df, station)
    
    if hp.REDUCE_TRAIN:
        mydate = '2017-{}'.format(hp.FILENAME[-5:]) 
        df_train = df_train[df_train.datetime < mydate]    
    
    train_X, train_y = get_XY(df_train)    
    train_X, train_y = shuffle(train_X, train_y)   
    test_X, test_y = get_XY(df_test)    
    return df_train, df_test, train_X, train_y, test_X, test_y


def generate_results(df_test, pred, verbose = True):
    """
    Generates results after the training (testing) process
    Arguments:
        df_test    - test dataframes
        pred       - output from model.predict
        verbose    - if True than information for each station is printed
        plot_chart - if True than loss chart is plotted
    Returns:
        resdf  - dataframe with summary results for each TMC
        preddf - dataframe with predictions, ground_truth values and results
    """
    resdf = pd.DataFrame(columns=["count_location", "nr_points", "r2", "mape", "smape", "emfr"])
    
    testdf = df_test.copy()
    testdf['pred'] = pred
    testdf.loc[testdf.pred < 0, 'pred'] = 0
    testdf['MeanAbsErr'] = np.abs(testdf.pred - testdf.count_total)
    testdf['mape'] = 100 * testdf.MeanAbsErr / testdf.count_total    
    testdf['smape'] = 200 * np.abs(testdf.pred - testdf.count_total) / ( testdf.pred.abs() + testdf.count_total.abs() ) 
    
    for tmc in list(testdf.tmc.unique()):
        tmp = testdf.loc[testdf.tmc == tmc]
        r2 = r2_score(tmp.count_total, tmp.pred)
        mape = np.mean(tmp.mape)
        smape = np.mean(tmp.smape)
        max_volume = np.max(tmp.count_total)
        emfr = 100 * np.mean(tmp.MeanAbsErr / max_volume) 
        
        resdf.loc[tmc, "count_location"] = tmp.count_location.iloc[0]
        
        
        if 'count_type' in tmp.columns:
            resdf.loc[tmc, 'count_type'] = tmp.count_type.iloc[0]
        if 'greater_harrisburg' in tmp.columns:
            resdf.loc[tmc, 'greater_harrisburg'] = tmp.greater_harrisburg.iloc[0]
    
        resdf.loc[tmc, "nr_points"] = len(tmp)
        resdf.loc[tmc, "r2"] = r2
        resdf.loc[tmc, "mape"] = mape
        resdf.loc[tmc, "smape"] = smape
        resdf.loc[tmc, "emfr"] = emfr

    if verbose:
        if len(resdf) == 1:
            print('R2 {:.2f}, MAPE: {:.1f}%, SMAPE: {:.1f}%, EMFR: {:.3f}%'
                  .format(resdf.r2.iloc[0], resdf.mape.iloc[0], resdf.smape.iloc[0], resdf.emfr.iloc[0]))
        elif len(resdf) == 2:
            print('Mean R2 {:.2f} ({:.2f}, {:.2f}), mean MAPE: {:.1f}% ({:.1f}%, {:.1f}%), mean SMAPE: {:.1f}% ({:.1f}%, {:.1f}%), mean EMFR: {:.3f}% ({:.3f}%, {:.3f}%)'
                  .format(
                      np.mean(resdf.r2), resdf.r2.iloc[0], resdf.r2.iloc[1],
                      np.mean(resdf.mape), resdf.mape.iloc[0], resdf.mape.iloc[1],
                      np.mean(resdf.smape), resdf.smape.iloc[0], resdf.smape.iloc[1],
                      np.mean(resdf.emfr), resdf.emfr.iloc[0], resdf.emfr.iloc[1],
                  ))                     
        else:
            print('Mean R2 {:.2f}, mean MAPE: {:.1f}%, mean SMAPE: {:.1f}%, mean EMFR: {:.3f}%'
                  .format(
                      np.mean(resdf.r2), np.mean(resdf.mape), np.mean(resdf.smape), np.mean(resdf.emfr)
                  ))
                  
            print ('Median R2 {:.2f}, median MAPE: {:.1f}%, median SMAPE: {:.1f}%, median EMFR: {:.3f}%'
                  .format(
                      np.median(resdf.r2), np.median(resdf.mape), np.median(resdf.smape), np.median(resdf.emfr)
                  ))

    resdf.index = resdf.index.rename('tmc')            
    return resdf, testdf 


def make_predictions(hp, model, test_X):
    """
    Temporary function, solves the problems with Keras model.predict in case of using multiple GPUs.
    It works by adding the data in the end of the inputs, to make the len(input)%BATCH_SIZE == 0.
    Then it makes predictions and removes superfluous results
    Arguments:
        hp     - object with hyperparameters
        model  - keras model used for predictions
        test_X - input data
    Returns:
        pred - vector with predictions. Note: len(pred) == len(test_X)
    """
    xlen = test_X.shape[0]
    rem = xlen % hp.BATCH_SIZE
    sup = hp.BATCH_SIZE - rem

    if rem != 0:
        inp = np.vstack([test_X, test_X[-sup:, :]])
        pred = model.predict(inp)
        pred = pred[:-sup]    
    else:
        pred = model.predict(test_X)
    return pred


def train_model_cv(hp, df, stations = None, gpus = None, overwrite_allowed = True):
    """
    Trains the model with full cross_validation procedure
    Arguments:
        hp       - object with hyperparameters
        df       - dataframe with inputs and labels
        stations - list of stations (or TMCs) used for cross validation If not given all stations are used
    Returns:
        resdf  - dataframe with summary results
        preddf - dataframe with predictions, ground_truth values and results
    """
    summary_displayed = False
    hp.VERBOSE = 0
    resdf = pd.DataFrame(columns=["r2", "mape", 'emfr'])


    
    if hp.USE_ATR:
        if stations is None:
            all_stations = list(df.count_location.unique())
        else:
            all_stations = stations 
    else:
        print ('WARNING: hp.USER_ATR is false. NOT IMPLEMENTED!')
        return None, None
    all_stations.sort()
                
    nrmodels = len(all_stations)

    t = time()

    st_nr = 0
    for station in all_stations:
        st_nr += 1
        
        filepath = '{}_{}.hdf5'.format(hp.FILECORE_PATH, station)
        
        if (not overwrite_allowed) and (os.path.isfile(filepath)):
            print ("WARNING: file {} exists. Skipping station {} ({}/{})."
                  .format(filepath, station, st_nr, len(all_stations)))            
            continue


        (df_train, df_test, train_X, train_y, test_X, test_y) = prepare_traintest_data(hp, df, station)

        if hp.NORMALIZE:
            train_X, test_X = normalize(hp, train_X, test_X)

        sys.stdout.write('Station {} ({}/{}) - Preparing model...                 \r'
                        .format(station, st_nr, len(all_stations)))
        
        model, strategy = create_model(hp, train_X.shape[1], gpus)
            
        if not summary_displayed:
            print ('EPOCHS: {}                                    '.format(hp.EPOCHS))
            print (model.summary())
            summary_displayed = True
            print ("-----")

        sys.stdout.write('Station {} ({}/{}) - Fitting model ...                \r'
                        .format(station, st_nr, len(all_stations)))
        
        if gpus is None:
            history = model.fit(
                train_X, train_y, 
                validation_data = (test_X, test_y),
                batch_size = hp.BATCH_SIZE, 
                epochs=hp.EPOCHS, 
                verbose=hp.VERBOSE
            )
        else:
            history = model.fit(
                train_X, train_y, 
                validation_data = (test_X, test_y),
                batch_size = hp.BATCH_SIZE, 
                epochs=hp.EPOCHS, 
                steps_per_epoch = int(len(train_X) / hp.BATCH_SIZE),
                validation_steps = int(len(test_y) / hp.BATCH_SIZE),
                verbose=hp.VERBOSE
            )
            
        v_loss = history.history['val_loss']
        t_loss = history.history['loss']


        sys.stdout.write('Station {} ({}/{}) - Saving model...                \r'
                        .format(station, st_nr, len(all_stations)))

        model.save(filepath)

        sys.stdout.write('Station {} ({}/{}) - Testing model...                \r'
                        .format(station, st_nr, len(all_stations)))

        if gpus is None:
            pred = model.predict(test_X)
        else:
            pred = make_predictions(hp, model, test_X)
            
        print('Station {} ({}/{}) - {}                             '
              .format(station, st_nr, len(all_stations),
                      strftime('%H:%M:%S', gmtime(time() - t))))
        
        tmp_resdf, tmp_preddf = generate_results(df_test, pred, verbose=True)
        plot_train_valid_test(hp, t_loss, v_loss)

        if len(resdf) == 0:
            resdf = tmp_resdf
            preddf = tmp_preddf
        else:
            resdf = resdf.append(tmp_resdf, ignore_index=True)
            preddf = preddf.append(tmp_preddf, ignore_index=True)


    print ('Done in {}'.format(strftime('%H:%M:%S', gmtime(time() - t))))    
    return resdf, preddf


def test_model_cv(hp, df, base_model = None, stations = None, **kwargs):
    """
    Test the models created with full cross_validation procedure
    Arguments:
        hp           - object with hyperparameters
        df           - dataframe with inputs and labels
        base_model   - model to be ysed for predictions. If None, the models are loaded
                       based on the hp.FILENAME argument
        stations     - list of stations (or TMCs) used for cross validation If not given all stations are used
    Other arguments (in **kwargs):
        normalize_fl - if True then normalization for pretrained florida model is used.
    Returns:
        resdf  - dataframe with summary results
        preddf - dataframe with predictions, ground_truth values and results
    """
    summary_displayed = False
    model_created = False
    hp.VERBOSE = 0
    
    resdf = pd.DataFrame(columns=["r2", "mape", 'emfr'])

    if hp.USE_ATR:
        all_stations = list(df.count_location.unique())
    else:
        all_stations = list(df.tmc.unique())
    all_stations.sort()

    nrmodels = len(all_stations)

    t = time()


    st_nr = 0
    for station in all_stations:
        st_nr += 1


        (df_train, df_test, train_X, train_y, test_X, test_y) = prepare_traintest_data(hp, df, station)


        if hp.NORMALIZE:
            if ('normalize_fl' in kwargs.keys()) and kwargs['normalize_fl']:
                print ("Florida Normalization")
                test_X = normalizeFL(hp, test_X)
            else:
                train_X, test_X = normalize(hp, train_X, test_X)

        sys.stdout.write('Station {} ({}/{}) - Preparing model...                 \r'
                        .format(station, st_nr, len(all_stations)))


        filepath = '{}_{}.hdf5'.format(hp.FILECORE_PATH, station)        
        filepath_best = '{}_{}_best.hdf5'.format(hp.FILECORE_PATH, station)

        if base_model is None:
            if not model_created:
                model, _ = create_model(hp, train_X.shape[1])
                print (model.summary())
                model_created = True
                print ("-----")
            model.load_weights(filepath)
        else:
            if not model_created:
                model = base_model
                print (model.summary())
                model_created = True
                print ("-----")
                
        sys.stdout.write('Station {} ({}/{}) - Testing model...                \r'
                        .format(station, st_nr, len(all_stations)))

        pred = model.predict(test_X)

        print('Station {} ({}/{}) - {}                             '
              .format(station, st_nr, len(all_stations),
                      strftime('%H:%M:%S', gmtime(time() - t))))
        
        tmp_resdf, tmp_preddf = generate_results(df_test, pred, verbose=True)

        if len(resdf) == 0:
            resdf = tmp_resdf
            preddf = tmp_preddf
        else:
            resdf = resdf.append(tmp_resdf, ignore_index=True)
            preddf = preddf.append(tmp_preddf, ignore_index=True)

    print ('Done in {}'.format(strftime('%H:%M:%S', gmtime(time() - t))))    
    return resdf, preddf


def print_sumation(resdf):
    print ('    Measure       | Value')
    print ('------------------+-------')
    print ('Mean R2           |  {:.2f}'
           .format(resdf.r2.mean()))    
    print ('Median R2         |  {:.2f}'
           .format(resdf.r2.median()))    
    print ('Min R2            |  {:.2f}'
           .format(resdf.r2.min()))    
    print ('Max R2            |  {:.2f}'
           .format(resdf.r2.max()))    
    print ('------------------+-------')
    print ('Mean MAPE         | {:.2f}%'
           .format(resdf.mape.mean()))    
    print ('Median MAPE       | {:.2f}%'
           .format(resdf.mape.median()))    
    print ('Min MAPE          | {:.2f}%'
           .format(resdf.mape.min()))    
    print ('Max MAPE          | {:.2f}%'
           .format(resdf.mape.max()))    
    print ('------------------+-------')
    print ('Mean SMAPE        | {:.2f}%'
           .format(resdf.smape.mean()))    
    print ('Median SMAPE      | {:.2f}%'
           .format(resdf.smape.median()))    
    print ('Min SMAPE         | {:.2f}%'
           .format(resdf.smape.min()))    
    print ('Max SMAPE         | {:.2f}%'
           .format(resdf.smape.max()))    
    print ('------------------+-------')
    print ('Mean EMFR         | {:.4f}%'
           .format(resdf.emfr.mean()))    
    print ('Median EMFR       | {:.4f}%'
           .format(resdf.emfr.median()))    
    print ('Min EMFR          | {:.4f}%'
           .format(resdf.emfr.min()))    
    print ('Max EMFR          | {:.4f}%'
           .format(resdf.emfr.max()))    

    
def save_results(hp, preddf, resdf, data_filename, data_skiprows):
    tmp = pd.read_csv(data_filename, skiprows=data_skiprows)
    if 'datetime' in tmp.columns:
        tmp.datetime = pd.to_datetime(tmp.datetime)
    else:
        tmp['datetime'] = pd.to_datetime(tmp.date + ' ' + tmp.time)  
    tmp = tmp.merge(preddf[['count_location', 'tmc', 'datetime', 'pred', 'mape', 'smape']], on=["count_location", "tmc", "datetime"], how="inner")

    tmp.set_index("tmc").to_csv("{}_results_all.csv".format(hp.RESULTS_FILE))
    print ("{}_results_all.csv saved.".format(hp.RESULTS_FILE))

    resdf.sort_values("r2").to_csv("{}_results.csv".format(hp.RESULTS_FILE))
    print ("{}_results.csv saved.".format(hp.RESULTS_FILE))
