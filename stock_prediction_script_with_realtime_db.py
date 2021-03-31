# https://www.sharemarkethub.com/2020/05/09/data-science-projects/

# Importing Libraries
import numpy as np
import pandas as pd
import math

import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime, timedelta
# import streamlit as st   

plt.style.use('fivethirtyeight')
from nsepy import get_history
from datetime import date

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

databaseURL = 'https://stonks-7b3ab-default-rtdb.firebaseio.com/'
# Fetch the service account key JSON file 
cred = credentials.Certificate('stonks-7b3ab-firebase-adminsdk-h2p7m-7106810010.json')
# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': databaseURL
})

def create_db(data_db1):
    ref = db.reference('stocks')
    ref.update(data_db1)

def create_db_branch(data_db2):
    ''' Function that creates a new branch in the database everytime 
        a new company's stocks are predicted
    '''
    ref = db.reference('stocks')
    ref.set(data_db2)

def add_new_data(future_pred, symbol, num, date_of_pred, label):
    ''' Function to add a new row of prediction to specific 
        document_id of specific stock_symbol
    '''
    ref = db.reference('stocks').child(symbol).child('historical').child(num)
    # ref = db.reference('/')
    # ref.set(data_db) #  .
    ref.update({
        'close': future_pred,
        'date': date_of_pred,
        'label': label
        })

def update(stock_symbol, stock_pred_id, field, updated_value):
    ''' Function to update specific value of specific field of 
        specific document_id of specific stock_symbol
    '''
    ref = db.reference('stocks')
    stock_ref = ref.child(symbol).child(stock_pred_id)
    stock_ref.update({
    field : updated_value
})

def retreive(stock_symbol):
    ''' Function to retreive all the data of a specific stock_symbol
    '''
    ref = db.reference('stocks')
    return ref.child(stock_symbol).get()

def main():
    # Importing Historical Stock Price Data using nsepy's get_history function
    symbol = input("Enter Stock Symbol:")
    n = int(input("Enter number of days for which you want prediction:")) # 'cipla', 2  
    start = date(2015, 1, 1) 
    end = date.today() 
    df = get_history(symbol = symbol, start = start, end = end)
    df['Date'] = df.index

    # Plotting the Close price data
    plt.figure(figsize = (18, 9))
    plt.title('Close Price History')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price', fontsize = 18)
    plt.fill_between(df['Date'], df['Close'], color = "skyblue", alpha = 0.2)
    plt.plot(df['Date'], df['Close'], color = "skyblue", alpha = 0.6)
    plt.show()

    # Create a new dataframe to store the close prices
    data = df.filter(['Close'])
    # Converting the dataframe to a numpy array
    dataset = data.values
    # divide the dataset into training data and testing data 
    training_data_len = math.ceil(len(dataset)*0.75)

    # Scale data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Creating the training data set from scaled dataset
    prediction_days = 30
    train_data = scaled_data[0:training_data_len, :]
    # Split the data into x_train and y_train
    x_train = []
    y_train = []
    for i in range(prediction_days, len(train_data)):
        x_train.append(train_data[i - prediction_days:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the model
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(units = 50, return_sequences = False))
    model.add(Dense(units = 25))
    model.add(Dense(units = 1))

    # Compile the model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size = 32, epochs = 100)

    # Save the model
    model.save(symbol+'.model')

    # Test Data set
    test_data = scaled_data[training_data_len - prediction_days:, : ]
    # Create the x_test and y_test data sets
    x_test = []
    y_test = dataset[training_data_len : , : ]
    for i in range(prediction_days, len(test_data)):
        x_test.append(test_data[i - prediction_days: i, 0])

    # Convert x_test to a numpy array
    x_test = np.array(x_test)
    # Reshape the data into the shape accepted by the LSTM model
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Getting the model's predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    print(len(predictions))
    # Calculate RMSE Value
    rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
    print("RMSE:", rmse)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Visualizing the data
    plt.figure(figsize = (18, 9))
    plt.title('Predicted Stock price and Real Stock Price')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price', fontsize = 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
    plt.show()

    valid[-5:]

    # Next Day Price Predcition 
    newdf = df.filter(['Close'])
    # newdf['Predictions']


    # Leapfrog 
    for i in range(n):
        last_30_days = newdf[-30:].values
        last_30_days_scaled = scaler.transform(last_30_days)
        print("last_30_days:\n")
        print(last_30_days)
        
        x_test = []
        x_test.append(last_30_days_scaled)
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        pred_price = model.predict(x_test)
        pred_price = scaler.inverse_transform(pred_price)
        
        print("Predicted Price:{}".format(pred_price))
        
        last_date = date.today()
        next_date = last_date + timedelta(days = i)
        
        newdf.loc[next_date, 'Close'] = pred_price

    print(newdf.dtypes)
    print("Next {} day predictions are: {}".format(n, newdf[-(n * -1):]))
    newdf2 = newdf[(n*-1):].Close.astype(int)
    print("newdf2:", newdf2)
    print("newdf2[1]:", newdf2[1])

    for i in range(n):
        last_date = date.today()
        next_date = last_date + timedelta(days = i)
        ''' stock_pred_id = symbol + '_' + str(next_date.strftime('%d%m%Y'))   #'cipla_18032021' # (stock_symbol + _ + ddmmyy)
        print("stock_pred_id:", stock_pred_id) '''
        num = str(i)

        date_of_pred = str(next_date.strftime('%Y-%m-%d'))
        
        label = next_date.strftime('%b %d, %Y')
        data_db1 = {
            'stocks': 
                {
                    symbol:
                    {
                        'historical':
                        {
                            num:
                            {
                            'close': int(newdf2[i]),
                            }
                        }
                    }
                }
            }
        '''if i == 0:
            create_db(data_db1)
        else:'''
        add_new_data(int(newdf2[i]), symbol, num, date_of_pred, label) # symbol, stock_pred_id


if __name__ == '__main__':
    main()
