# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:19:27 2020

utilities

@author: Merten
"""
import pandas as pd
import numpy as np
import os
import scipy.interpolate as scpinter
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import tensorflow as tf


"""
INPUT DATA:
    - absorption curves
    - 3079 measurements, 1960 frequency points
    - smoothened (even though the 'outliers' are no true outliers)

"""


def load_data(n_sampling):
    """ load the raw measurement data and preprocess.
    
    n_sampling: number of frequency points requested by the user. 1960 for the original 1Hz sampling
    """

    """
    INPUT and TARGET DATA

    - measurements of alpha, sampled at 1Hz within 270-2229 Hz range
    - data are not noisy, and there are no true outliers. All obvious outliers
    are in fact physical effects that are not yet understood. However, we will
    smoothen the data for the purpose of this work.

    Furthermore, we'll downsample the data to reduce the required complexity.
    The number of sampling points is given by <n_sampling>.

    """

    # --- load the absorption measurements (input features)
    alpha = np.array(pd.read_csv(os.path.realpath('.') + '\\' + 'alphas.csv', header=None))

    # corresponding frequency vector 
    f_min = 270   # minimum frequency
    f_max = 2229  # maximum frequency
    f = np.linspace(start=f_min, stop=f_max, num=alpha.shape[1])  # 1Hz sampling
    
    # --- load the factors (target values)
    targets = np.array(pd.read_csv(os.path.realpath('.') + '\\' + 'targets.csv', header=None))

    # create a pandas data frame
    factor_names = ['30mm', '40mm', '50mm', '80mm',                   # specimen thickness
                    '89mm', '90mm', '91mm',                           # specimen diameter
                    'HZD', 'MES', 'SLF', 'SLH', 'SSF', 'SSH', 'WSS',  # cutting technology
                    'plunger_fixed', 'plunger_moved',                 # mounting
                    'computer', 'person1', 'person2', 'person3',      # operator
                    '1', '2', '3',                                    # sample index (obsolete)
                    'x', 'y', 'z']                                    # measurement index (obsolete)
    factors = pd.DataFrame(targets, columns=factor_names)

    # we do not need the number of the probe, neither the measurement number:drop them
    factors = factors.drop(['1', '2', '3', 'x', 'y', 'z'], axis=1)
    print('number of factors: '+str(np.array(factors).shape[1]))

    # -- filtering
    # there are some absorption curves that are commpletely negative. We'll remove them
    mins = np.mean(alpha, axis=1)  # find indices of the all-negatives
    
    # remove the all-negative samples
    alpha = alpha[mins>0,:]
    factors = factors[:][mins>0]
    print('removed ' + str(np.sum(mins<0)) + ' all-negative samples')
    
    # replace all negative entries with small value (0.01)
    alpha_pos = result = np.where(alpha<0, 0.01, alpha)

    # now fill in the current drops by rolling median and rolling max
    alpha_smooth_med = np.array(pd.DataFrame(alpha_pos).rolling(window=25, center=True, min_periods=1, axis=1).median())
    alpha_smooth_max = np.array(pd.DataFrame(alpha_smooth_med).rolling(window=50, center=True, min_periods=1, axis=1).max())

    # --- downsampling
    m = int(len(f)/n_sampling)
    f_coarse = f[::m]  # pick each mth element
    alpha_coarse = alpha_smooth_max[:, ::m]  # pick each mth element

    n_observations = alpha_coarse.shape[0]
    n_inputs  = len(f_coarse)
    print('number of measurements: ' + str(n_observations))
    print('number of frequency points: ' + str(n_inputs))
    
    return alpha_coarse, factors, factor_names, f_coarse


def prepare_data_univariate(X, factors, quant, cv_flag):
    """ Do the data splitting.
    
    inputs:
        - X: absorption values
        - factors: targets
        - quant: current factor dimension
        - cv_flag: wether to do cross-validation splitting or not
    """

    """
    1. Select one of the target values
    quant:
        - 'thickness'
        - 'diameter'
        - 'cutting_tech'
        - 'operator'
        - 'mounting'
    """

    # extract the one-hot encoded target values from the factors data frame
    if quant == 'thickness':
        class_names = ['30mm', '40mm', '50mm', '80mm']
    elif quant == 'diameter':
        class_names = ['89mm', '90mm', '91mm']
    elif quant == 'cutting_tech':
        class_names = ['HZD', 'MES', 'SLF', 'SLH', 'SSF', 'SSH', 'WSS']
    elif quant == 'operator':
        class_names = ['computer', 'person1', 'person2', 'person3']
    elif quant == 'mounting':
        class_names = ['plunger_fixed', 'plunger_moved']
    else:
        print('ERROR, wrong quantity chosen!')
        
    # get the correct factors
    y = np.array(factors[class_names])


    """
    2. Shuffle the data

    the data comes in very structured, so we better shuffle it
    """
    X, y = shuffle(X, y, random_state=1)


    """
    3. cross-validation splitting (if requested)
    """

    if cv_flag:

        # perform a stratified k=5-fold cross validation split
        skf = KFold(n_splits=5)
        skf.get_n_splits(X, y)
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for train_index, test_index in skf.split(X, y):
            # print(test_index)
            X_train_temp, X_test_temp = X[train_index], X[test_index]
            y_train_temp, y_test_temp = y[train_index], y[test_index]

            X_train.append(X_train_temp)
            X_test.append(X_test_temp)
            y_train.append(y_train_temp)
            y_test.append(y_test_temp)

    else:

        # do a stratified split (75-25)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.25)

    return X_train, X_test, y_train, y_test, class_names



def get_univariate_targets(y, quant):
    """ get the correct data labels.
    
    y: output values (one-hot encoded)
    quant: factor (as string), see below
    """
    
    # extract the one-hot encoded target values from the factors data frame
    if quant == 'thickness':
        class_names = ['30mm', '40mm', '50mm', '80mm']
    elif quant == 'diameter':
        class_names = ['89mm', '90mm', '91mm']
    elif quant == 'cutting_tech':
        class_names = ['HZD', 'MES', 'SLF', 'SLH', 'SSF', 'SSH', 'WSS']
    elif quant == 'operator':
        class_names = ['computer', 'person1', 'person2', 'person3']
    elif quant == 'mounting':
        class_names = ['plunger_fixed', 'plunger_moved']
    else:
        print('ERROR, wrong quantity chosen!')
        
    # get the correct one-hot encoded target values
    if type(y) is list:
        
        for idx,y_temp  in enumerate(y):
            y[idx] = y_temp[class_names]
            
    else:  
        y = y[class_names]
    
    return y, class_names


def build_model(model_name, n_features, n_outputs, multilabel_flag):
    """
    Build a classification model.

    - activation: ReLU except for last layer
    - activation (output layer): 
            - sigmoid (multi-label), 
            - softmax (single label, multiclass). The output will add up to 1

    for the MULTICLASS + MULTILABEL setting:
        choose sigmoid activation + binary_crossentropy
        do NOT use softmax (sum of outputs will equal to one)

    Parameters
    ----------
    model_name : TYPE
        DESCRIPTION.
    n_features : TYPE
        DESCRIPTION.
    n_outputs : TYPE
        DESCRIPTION.

    Returns
    -------
    model : TYPE
        DESCRIPTION.

    """  
    if multilabel_flag:
        last_act = 'sigmoid'
    else:
        last_act = 'softmax'

    if model_name == 'baseline':
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(n_features, input_shape=(n_features,), activation='relu', kernel_initializer='uniform'))
        model.add(tf.keras.layers.Dense(int(n_features/2), activation='relu'))
        model.add(tf.keras.layers.Dense(50, activation='relu'))
        model.add(tf.keras.layers.Dense(25, activation='relu'))
        model.add(tf.keras.layers.Dense(n_outputs, activation=last_act))
        model.summary()

    elif model_name == 'deepmlp':
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(n_features, input_shape=(n_features,), activation='relu', kernel_initializer='uniform'))
        model.add(tf.keras.layers.Dropout(0.2))
        
        if n_features == 1960:
            model.add(tf.keras.layers.Dense(int(n_features/2), activation='relu')) # 980
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/4), activation='relu')) # 490
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/8), activation='relu')) # 245
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/16), activation='relu')) # 123
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/32), activation='relu')) # 62
            model.add(tf.keras.layers.Dropout(0.2))
            
        elif n_features == 980:
            model.add(tf.keras.layers.Dense(int(n_features/2), activation='relu')) # 490
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/4), activation='relu')) # 245
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/8), activation='relu')) # 123
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/16), activation='relu')) # 62
            model.add(tf.keras.layers.Dropout(0.2))
        
        elif n_features == 392:
            model.add(tf.keras.layers.Dense(int(n_features/2), activation='relu')) # 196
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/4), activation='relu')) # 98
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/16), activation='relu')) # 49
            model.add(tf.keras.layers.Dropout(0.2))
        
        elif n_features == 196:
            model.add(tf.keras.layers.Dense(int(n_features/2), activation='relu')) # 98
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(int(n_features/16), activation='relu')) # 49
            model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(25, activation='relu'))
        model.add(tf.keras.layers.Dense(n_outputs, activation=last_act))
        model.summary()


    return model


def train_evaluate_model(model, epochs, batch_size, X_train, X_test, y_train, y_test):
    """
    Train a given model.

    As we have only classification tasks (binary, or multiclass), we stick to
    the following settings:
        - loss function: ?
        - optimizer: adam
        - metric: accuracy (the data set is balanced, so acc is ok)

    Parameters
    ----------
    model : Keras (TF) model
        keras model compiled using the <build_model> function
    epochs : int
        number of epochs
    batch_size : int
        batch size. Incease for smoothing the training behavior
    X_train : np array
        input data, training set
    X_test : np array
        input data, test set
    y_train : np array
        output data, training set
    y_test : np array
        output data, test set

    Returns
    -------
    model : keras (TF) model
        trained model instance
    history : keras
        model training history
    test_acc : float
        test set accuracy

    """

    # we will have to switch the loss function for the binary / multiclass setting
    n_output = y_train.shape[1]
    if n_output == 2:
        loss_fun = 'binary_crossentropy'
    elif n_output > 2:
        loss_fun = 'categorical_crossentropy'


    # compile model
    model.compile(loss=loss_fun, optimizer='Adam', metrics=['accuracy'])

    # fit model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), \
                        epochs=epochs, batch_size=batch_size, verbose=0)

    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#     # plot loss during training
    #plt.figure(num=None, figsize=(16, 6), dpi=100, facecolor='w')
    #plt.subplot(121)
    #plt.ylabel('Loss')
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
         # plot accuracy during training
    #plt.subplot(122)
    #plt.ylabel('Accuracy')
    #plt.xlabel('epochs')
    #plt.plot(history.history['accuracy'], label='train')
    #plt.plot(history.history['val_accuracy'], label='test')
    #plt.legend()
    #plt.savefig('temp_training_history.png')
    #plt.show()


    return model, history, test_acc



def get_high_confidence_predictions(y_gt, y_pred, X, conf):
    """ for a given data set, keep only those samples for which we achieved acceptable confidence levels
    """

    dp = np.sum(y_pred*y_gt, axis=1)
    print(dp.shape)

    X_conf = X[dp > conf,:]
    y_conf = y_gt[dp > conf,:]

    print('Confidence >' + str(conf) + ' filtering. Keeping ' + str(X_conf.shape[0]) + ' samples out of ' + str(X.shape[0]))
    
    return X_conf, y_conf



def prepare_data_multivariate(X, factors, cv_flag):

    # get the target variables column names
    y_col_names = factors.columns
    
    """
    1. Shuffle the data

    the data comes in very structured, so we better shuffle it
    """
    y = np.array(factors)
    X, y = shuffle(X, y, random_state=0)
    
    """
    2. cross-validation splitting (if requested)
    """
    if cv_flag:

        # perform a stratified k=5-fold cross validation split
        skf = KFold(n_splits=5)
        skf.get_n_splits(X, y)
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for train_index, test_index in skf.split(X, y):
            # print(test_index)
            X_train_temp, X_test_temp = X[train_index], X[test_index]
            y_train_temp, y_test_temp = y[train_index], y[test_index]

            # make the labels a pd data frame again
            y_train_temp = pd.DataFrame(y_train_temp, columns=y_col_names)
            y_test_temp = pd.DataFrame(y_test_temp, columns=y_col_names)
            
            # append to list of CV data sets
            X_train.append(X_train_temp)
            X_test.append(X_test_temp)
            y_train.append(y_train_temp)
            y_test.append(y_test_temp)

    else:

        # do a stratified split (75-25)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    test_size=0.25)
        
        y_train = pd.DataFrame(y_train, columns=y_col_names)
        y_test = pd.DataFrame(y_test, columns=y_col_names)

    return X_train, X_test, y_train, y_test

