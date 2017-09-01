'''
This file loads data, cleans data, generates additional features, and calls
the model_class file in order to output a classification model.
'''

import pandas as pd
import numpy as np
from model_class import *
# import Models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def load_file(fname):
    df = pd.read_csv(fname)

    # Shuffle the training data
    df = df.iloc[np.random.permutation(df.shape[0])]

    df = modify_features(df)

    # Impute missing values
    df = impute(df)

    # Remove rows that have not been imputed yet
    # df = remove_nulls(df)

    return df


def modify_features(df):
    # Convert dates
    df.last_trip_date = pd.to_datetime(df.last_trip_date, format='%Y-%m-%d')
    df.signup_date = pd.to_datetime(df.signup_date, format='%Y-%m-%d')

    # Add new features
    # df['days_elapsed'] = create_days_elapsed(df)
    df['days_since_creation'] = create_days_since_creation(df)
    df['avg_dist_per_day'] = df.avg_dist / df.days_since_creation
    df['y'] = get_y(df)
    # Drop the date columns
    df.drop(['last_trip_date','signup_date'], axis=1, inplace=True)

    # Get Dummies: convert string and boolean columns to 1 or 0
    df = pd.get_dummies(df, drop_first=True, columns=['phone', 'city', 'luxury_car_user'])
    # leave nothing to chance with dummy column naming: put it back to the original
    df.rename(columns={'phone_iPhone': "phone", 'phone_Android': 'phone'}, inplace=True)
    df.rename(columns={'luxury_car_user_True': "luxury_car_user", 'luxury_car_user_False': "luxury_car_user"}, inplace=True)

    return df


def impute(df):
    for col in ['avg_rating_by_driver', 'avg_rating_of_driver']:
        df[col].fillna(df[col].mean(), inplace=True)

    df['phone'].fillna(round(df['phone'].mean()), inplace=True)
    return df


# Destroy this function once imputation is up and running
def remove_nulls(df):
    # Get nulls for avg_rating_by_driver
    by_drv = df.avg_rating_by_driver.isnull()
    # Get nulls for phone
    phn = df.phone.isnull()
    # Get nulls for avg_rating_of_driver
    of_drv = df.avg_rating_of_driver.isnull()
    # Get average distance is 0
    dist = (df.avg_dist==0)

    # all entries are filled out: 33132
    all_full = df[~by_drv & ~phn & ~of_drv & ~dist]

    return all_full


def get_y(df):
    max_date = max(df.last_trip_date)
    delta_days = max_date - df.last_trip_date
    delta_days = delta_days.astype('timedelta64[D]')
    delta_days = delta_days.astype(int)
    return (delta_days > 30)

def create_days_elapsed(df):
    days_elapsed = df.last_trip_date - df.signup_date
    days_elapsed = days_elapsed.astype('timedelta64[D]')
    days_elapsed = days_elapsed.astype(int)
    return days_elapsed

def create_days_since_creation(df):
    max_date = max(df.last_trip_date)
    days_since_creation = max_date - df.signup_date
    days_since_creation = days_since_creation.astype('timedelta64[D]')
    days_since_creation = days_since_creation.astype(int)
    return days_since_creation


def transform_data(df, model, data):
    df_num = df.select_dtypes(exclude=['uint8','bool'])
    df_objects = df.select_dtypes(exclude=['float64','int64'])

    df_num_colnames = list(df_num.columns.values)
    df_objects_colnames = list(df_objects.columns.values)
    scaler = model
    if data == 'train':
        scaler.fit(df_num)
        df_num = scaler.fit_transform(df_num)
        df_num = pd.DataFrame(df_num)

        df_num.colnames = df_num_colnames

        df = pd.concat([df_objects, df_num], axis=1)

        df.columns = df_objects_colnames + df_num_colnames

    else:
        df_num = scaler.transform(df_num)
        df_num = pd.DataFrame(df_num)

        df_num.colnames = df_num_colnames

        df = pd.concat([df_objects, df_num], axis=1)

        df.columns = df_objects_colnames + df_num_colnames
    return df



def expected_profit(model, y, X):
    tn, fp, fn, tp = confusion_matrix(y, model.predict(X)).ravel()
    conf_m = np.array([[tp, fp], [fn, tn]])
    conf_m = conf_m/sum(conf_m)

    # Calculated from average of 2 trips per month & estimated $10 profit per ride
    profit = 20
    # Estimated cost to outreach to customer for retention
    outreach = -12.25
    # Cost benefit matrix; no cost for those predicted not to churn
    profit_m = np.array([[profit+outreach, outreach], [0, 0]])

    expected_profit = np.sum(conf_m * profit_m)

    return expected_profit


if __name__ == '__main__':
    train_X = load_file("../data/churn_train.csv")
    test_X = load_file("../data/churn_test.csv")

    scaler = StandardScaler()
    train_X = transform_data(train_X, scaler, 'train')
    test_X = transform_data(test_X, scaler, 'test')

    train_y = train_X.pop('y')
    test_y = test_X.pop('y')

    model = Models(model_type=LogisticRegression)
    model.fit(train_X, train_y)
    test_y_predicted = model.predict(test_X)
    score = model.score(test_y_predicted, test_y, accuracy_score)
    model.feature_importance()

    print("\nExpected profit: ",expected_profit(model, test_y, test_X))
