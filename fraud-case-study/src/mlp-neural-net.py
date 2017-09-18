import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from pprint import pprint


def create_model(num_neurons=10, optimizer='adam', kernel_initializer='uniform', activation='relu'):
    # available activation functions at: options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    model = Sequential()  # sequence of layers

    # Set input_dim to the number of features
    # Dense class defines new layers
    # First argument = number of neurons in the layer
    # Set kernel_initializer to the chosen intialization method for the layer
    # Either use uniform or normal method
    # Set activation to the chosen activation function
    # Use sigmoid on the last output layer to map predictions between 0 and 1

    num_inputs = X_train.shape[1]  # number of features
    # num_classes = len(np.unique(y_train))  # number of classes of target, can be 0-9
    num_neurons_in_layer = num_neurons

    # 1st layer - input layer
    # number of neurons = number of features
    model.add(Dense(input_dim=num_inputs,
                    units=num_neurons_in_layer,
                    kernel_initializer=kernel_initializer,
                    activation=activation))

    # 2nd layer - hidden layer; very rarely would you use more than one layer
    model.add(Dense(input_dim=num_neurons_in_layer,
                    units=num_neurons_in_layer,
                    kernel_initializer=kernel_initializer,
                    activation=activation))

    # 3rd layer - output layer
    # Number of neurons = 1 for regression
    # Number of neurons = 1 for classication unless using softmax,
    # then equal to number of classes
    model.add(Dense(input_dim=num_neurons_in_layer,
                    units=1,
                    kernel_initializer=kernel_initializer,
                    activation='sigmoid'))

    # Compile model
    # Change loss and metrics for a regression problem
    # Can play with the optimizer but adam is a good place to start
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Setting random seed for reproducibility
    np.random.seed(42)

    npz = np.load('../data/Xycompressed.npz')
    X_train = npz['X_train']
    X_test = npz['X_test']
    y_train = npz['y_train']
    y_test = npz['y_test']

    # Start with setting to mean of the neurons in the input and output (classes) layers, then tune
    # num_neurons = int(np.mean([X_train.shape[1], len(np.unique(y_train))]))

    # Grid searching hyperparameters
    model = KerasClassifier(build_fn=create_model, verbose=0)
    param_grid = {'epochs': [20], 'batch_size': [40],
                  'optimizer': ['Adagrad', 'Adam', 'SGD'],
                  'num_neurons': [10, 20, 35], 'kernel_initializer': ['normal', 'uniform'],
                  'activation': ['linear', 'sigmoid']}
    g = GridSearchCV(estimator=model, param_grid=param_grid,
                     scoring='f1_weighted', verbose=10, n_jobs=3)
    g.fit(X_train, y_train)
    results = g.cv_results_
    print('\n\n')
    pprint(results)
    print('\n\n')
    print('Best Params: {}, Best Score: {}'.format(g.best_params_, g.best_score_))

    ################

    # # Fitting the model using best params
    # model = create_model(num_neurons=15, optimizer='Adagrad',
    #                      kernel_initializer='normal', activation='linear')
    #
    # # Epochs: number of iterations to run through the neural net
    # # Batch size: number of samples that are evaluated before a weight update
    # # Validation split is for cross validation
    # model.fit(X_train, y_train, epochs=20, batch_size=40, verbose=10, validation_split=0.3)
    #
    # y_test_pred = model.predict_classes(X_test, verbose=10)
    # print('\n')
    # print('Recall: {}'.format(recall_score(y_test, y_test_pred)))
    # print('Precision: {}'.format(precision_score(y_test, y_test_pred)))
    # print('Accuracy: {}'.format(accuracy_score(y_test, y_test_pred)))
    # print('F1 score: {}'.format(f1_score(y_test, y_test_pred)))
