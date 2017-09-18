import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pickle


class Data_Cleaning(object):

    def __init__(self, filepath):
        self.df = pd.read_json(filepath)

    def _cleaning_data(self):
        # Change any data types that are incorrect
        self.df.show_map = self.df.show_map.astype('object')
        self.df.has_logo = self.df.has_logo.astype('object')
        self.df.has_analytics = self.df.has_analytics.astype('object')
        self.df.fb_published = self.df.fb_published.astype('object')
        self.df.delivery_method = self.df.delivery_method.astype('object')

        # Changing date strings to readable format
        self.df.event_created = pd.to_datetime(self.df.event_created, unit='s')
        self.df.event_end = pd.to_datetime(self.df.event_end, unit='s')
        self.df.event_published = pd.to_datetime(self.df.event_published, unit='s')
        self.df.event_start = pd.to_datetime(self.df.event_start, unit='s')

        # Fill numerical null values
        numerical_vals = self.df.select_dtypes(include=['float64', 'int64'])
        for col in numerical_vals.columns:
            self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Fill categorical null values
        categorical_vals = self.df.select_dtypes(include=['object'])
        for col in categorical_vals.columns:
            self.df[col].fillna('Missing', inplace=True)

    def _creating_features(self):
        # Creating feature for number of previous events
        self.df['num_previous_events'] = [len(x) for x in self.df['previous_payouts']]

        # Creating new feature to combine sale duration columns
        self.df['max_sale_duration'] = self.df[['sale_duration', 'sale_duration2']].max(axis=1)

        # Creating feature for total previous payouts
        final = []
        for x in self.df['previous_payouts']:
            # x = [x]
            temp = []
            for d in x:
                temp.append(d['amount'])
            final.append(sum(temp))
        self.df['total_previous_payouts'] = final

        # Creating boolean target variable
        target_list = []
        for x in self.df['acct_type']:
            if x == 'premium':
                target_list.append(0)
            else:
                target_list.append(1)
        self.df['target'] = target_list

        # Creating boolean variable for country
        self.df['US'] = self.df['country'] == 'US'

        # Creating new variables for time related measurements
        self.df['hours_between_published_and_created'] = [i.seconds / 3600.0 for
                                                          i in (self.df.event_published - self.df.event_created)]
        self.df['hours_between_event_start_and_end'] = [i.seconds / 3600.0 for
                                                        i in (self.df.event_end - self.df.event_start)]

        self.df['hour_of_day_event_published'] = [i.hour for i in self.df.event_published]

        # Creating new feature based on all caps name
        final = []
        for x in self.df['name']:
            if x == '':
                final.append(False)
            else:
                final.append(x[len(x) - 1].istitle())
        self.df['last_letter_name_caps'] = final

        # Creating new feature based on all lowercase name
        final = []
        for x in self.df['name']:
            if x == '':
                final.append(False)
            else:
                final.append(x[0].islower())
        self.df['first_letter_name_lowercase'] = final

        # Creating features for blank info
        self.df['desc_blank'] = self.df.org_desc == ''
        self.df['org_name_blank'] = self.df.org_name == ''
        self.df['payee_blank'] = self.df.payee_name == ''
        self.df['venue_name_blank'] = self.df.venue_name == 'Missing'
        self.df['venue_state_blank'] = self.df.venue_state == 'Missing'
        self.df['payout_type_blank'] = self.df.payout_type == ''

        # Fill numerical null values
        numerical_vals = self.df.select_dtypes(include=['float64', 'int64'])
        for col in numerical_vals.columns:
            self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Creating dummy variables manually for streaming data
        self.df['currency_CAD'] = self.df.currency == 'CAD'
        self.df['currency_EUR'] = self.df.currency == 'EUR'
        self.df['currency_GBP'] = self.df.currency == 'GBP'
        self.df['currency_MXN'] = self.df.currency == 'MXN'
        self.df['currency_NZD'] = self.df.currency == 'NZD'
        self.df['currency_USD'] = self.df.currency == 'USD'

        self.df['delivery_method_1.0'] = self.df.delivery_method == 1.0
        self.df['delivery_method_3.0'] = self.df.delivery_method == 3.0

        self.df['listed_y'] = self.df.listed == 'y'

        self.df['payout_type_ACH'] = self.df.payout_type == 'ACH'
        self.df['payout_type_CHECK'] = self.df.payout_type == 'CHECK'

    def _removing_data(self):
        # Drop some unneeded columns, add back in 'acct_type' if training
        self.df.drop(['object_id', 'currency', 'has_header', 'acct_type', 'venue_longitude', 'delivery_method', 'payout_type', 'listed', 'venue_longitude', 'venue_latitude',
                      'previous_payouts', 'venue_address', 'ticket_types', 'approx_payout_date', 'event_created',
                      'email_domain', 'event_end', 'event_start', 'event_published', 'user_created', 'description', 'name', 'org_desc', 'org_name',
                      'payee_name', 'venue_name', 'venue_state', 'venue_country', 'country'], inplace=True, axis=1)

        # Dropping highly correlated features
        self.df.drop(['sale_duration2', 'sale_duration',
                      'num_payouts', 'gts'], inplace=True, axis=1)

        # Getting dummy variables
        # self.df = pd.get_dummies(self.df, drop_first=True)
        # columns = list(self.df.columns)
        # self.df[columns[27:]] = self.df[columns[27:]].astype('bool')
        # self.df.reset_index()

    def _modeling_prep(self):
        # Creating X & y variables
        y = self.df.pop('target')
        X = self.df

        # Sorting df into numerical and not for scaler
        numerical_vals = X.select_dtypes(include=['float64', 'int64'])
        categorical_vals = X.select_dtypes(exclude=['float64', 'int64'])

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

        # Scaling train data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[numerical_vals.columns])
        X_train_scaled = np.concatenate([X_train_scaled, X_train[categorical_vals.columns]], axis=1)

        # Scaling test data
        X_test_scaled = scaler.transform(X_test[numerical_vals.columns])
        X_test_scaled = np.concatenate([X_test_scaled, X_test[categorical_vals.columns]], axis=1)

        # Using SMOTE to generate extra synthetic samples of the smaller class
        # X_train_resampled, y_train_resampled = SMOTE(
        #     k_neighbors=3, m_neighbors=5).fit_sample(X_train_scaled, y_train)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def run_pipeline(self):
        self._cleaning_data()
        self._creating_features()
        self._removing_data()

        # Code for creating and compressing the X & y data
        # X_train, X_test, y_train, y_test = self._modeling_prep()
        # args = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
        # np.savez_compressed('../data/Xycompressed', **args)

        # Opening pickled model
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Predicting probabilities of fraud
        X = np.array(self.df)
        y_test_probs = model.predict_proba(X)

        # Printing probability of fraud (column 0 is probability of not fraud)
        print(y_test_probs[:, 1])

        # Printing category of probability prediction
        if y_test_probs[:, 1] < .2:
            print('Low Risk')
        elif y_test_probs[:, 1] < .8:
            print('Medium Risk')
        else:
            print('High Risk')


if __name__ == '__main__':
    dc = Data_Cleaning('../data/data.json')
    dc.run_pipeline()
