from django.shortcuts import render, redirect
from stock.forms import RegistrationForm
# For machine learning

import pandas as pd
import numpy as np
import math, datetime
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, svm
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


# from django.contrib.auth.forms import UserCreationForm

# Create your views here.
def enquiry(request):
    return render(request, 'accounts/enquiry.html')


def index(request):
    return render(request, 'accounts/home.html')


def register(request):
    if request.method == 'GET':
        context = {

        }
        return render(request, 'accounts/regform.html', context)
    else:
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('/home/login')
        else:
            return render(request, 'accounts/regform.html', {'form': form})


def ml_functions():
    df = pd.read_csv('data/agriculture-development-bank-data.csv', index_col='date',
                     parse_dates=True)
    df = df[['maxprice', 'minprice', 'closingprice', 'previousclosing', 'amount']]
    df['HL_PCT'] = (df['maxprice'] - df['minprice']) / df['minprice'] * 100.0
    df['PCT_change'] = (df['closingprice'] - df['previousclosing']) / df['previousclosing'] * 100.0

    forecast_col = 'closingprice'
    df.fillna(-99999, inplace=True)
    forecast_out = int(math.ceil(0.05 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)

    # Data exploration
    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    df.dropna(inplace=True)
    y = np.array(df['label'])
    y = preprocessing.scale(y)

    # Preparing for machine learning
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    # Write a file for train classifier for future use
    # with open('linearregression.pickle', 'wb') as f:
    #     pickle.dump(clf, f)
    #
    # pickle_in = open('linearregression.pickle', 'rb')
    # clf = pickle.load(pickle_in)

    accuracy = clf.score(X_test, y_test)
    # print(accuracy)
    forecast_set = clf.predict(X_lately)
    # print(forecast_set, accuracy, forecast_out)
    df['forecast'] = np.nan

    # Find the last date and next date
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day

    # Create a datetime index
    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    df['forecast'].dropna(inplace=True)
    final = df['forecast']
    result = final.reset_index()
    print("hello")
    print(result)

    # # Visualization using matplotlib
    # df['closingprice'].plot()
    # df['forecast'].plot()
    # plt.legend(loc=4)
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.show()

    return result


def prediction(request):
    result = ml_functions()
    date_list = list(result['date'])
    # print(date_list)
    forecast_list = list(result['forecast'])
    # print(forecast_list)
    result = zip(date_list, forecast_list)
    return render(request, 'accounts/prediction.html', {'result': result})
