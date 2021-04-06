# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:23:27 2019

@author: Darkhorrow
"""

import pandas as pd
import numpy as np
import datetime as dt

class DataCleaner(object):
    
    def __init__(self, dataset):
        
        # Define the labels of the dataset
        labels =  ['vendor_id', 'pickup_datetime', 'dropoff_datetime',
                   'passenger_count', 'trip_distance', 'pickup_longitude', 'pickup_latitude',
                   'ratecode_id', 'store_and_fwd_flag', 'dropoff_longitude', 'dropoff_latitude',
                   'payment_type', 'fare_amount', 'extra_charge', 'mta_tax', 'tip_amount',
                   'tolls_amount', 'improvement_surcharge', 'total_amount',
                   'pickup_location_id', 'dropoff_location_id']


        # Read the excel dataset using the pre-made labels
        dataset = pd.read_csv(dataset, names=labels, index_col=False)
        
        # Replace columns with whitespaces with NaN numpy data
        dataset.replace('', np.nan, inplace=True)
        # Drop the columns that have NaN columns ('pickup_location_id' and 'dropoff_location_id')
        dataset.dropna(inplace=True, axis='columns')
        # Drop rows with negative tip_amount, total_amount
        dataset = dataset.drop(dataset[dataset['tip_amount'] < 0].index)
        dataset = dataset.drop(dataset[dataset['total_amount'] < 0].index)
        # Drop rows with invalid number of passengers
        dataset = dataset.drop(dataset[dataset['passenger_count'] < 1].index)
        # Drop invalid locations
        dataset.drop(dataset[dataset['dropoff_longitude'] == 0].index)
        dataset.drop(dataset[dataset['dropoff_latitude'] == 0].index)
        # Drop invalid locations
        dataset.drop(dataset[dataset['pickup_longitude'] == 0].index)
        dataset.drop(dataset[dataset['pickup_latitude'] == 0].index)
        # Drop rows with invalid ratecode
        dataset = dataset.drop(dataset[dataset['ratecode_id'] < 1].index)
        dataset = dataset.drop(dataset[dataset['ratecode_id'] > 6].index)
        # Replace store_and_fwd_flag notation to 0 and 1
        dataset.replace(to_replace='N', value=0, inplace=True)
        dataset.replace(to_replace='Y', value=1, inplace=True)

        dataset['length_time'] = pd.to_datetime(dataset['dropoff_datetime']) - pd.to_datetime(dataset['pickup_datetime'])
        dataset['length_time'] = dataset['length_time'].dt.total_seconds()
        
        # Datetimes to seconds since epoch
        dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'])
        dataset['pickup_datetime'] = (dataset['pickup_datetime'] - dt.datetime(1970,1,1)).dt.total_seconds()
        dataset['dropoff_datetime'] = pd.to_datetime(dataset['dropoff_datetime'])
        dataset['dropoff_datetime'] = (dataset['dropoff_datetime'] - dt.datetime(1970,1,1)).dt.total_seconds()
        
        self.full_dataset = dataset

    '''
    Normalize to a normal distribution.
    '''
    def normalize(self, dataset, exclude_columns=[]):
        dataset_aux = dataset.copy()
        for column in exclude_columns:
            dataset_aux.drop([column], axis=1, inplace=True)

        medias = dataset_aux.mean(axis=0)
        desviaciones = dataset_aux.std(axis=0)
        dataset_aux = (dataset_aux - medias) / desviaciones

        for column in exclude_columns:
            dataset_aux = pd.concat([dataset[column], dataset_aux], axis=1)

        return dataset_aux
