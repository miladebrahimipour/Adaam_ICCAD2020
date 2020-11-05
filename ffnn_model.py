import argparse
import os
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
from sklearn.model_selection import train_test_split
import time
import numpy as np
import random

class feed_forward_neural_network:

    def __init__(self, filename, delimiter, test_size, model_filename, number_of_epochs = 2000, error_type = 'rmspe', training = 'dynamic'):
        self.read_file(filename, delimiter, test_size)
        print("read file")
        self.data_normalization()
        print("split data")
        if training == 'dynamic':
            self.create_model_dynamic(model_filename, number_of_epochs, error_type)
        elif training == 'static':
            self.build_model()
            self.train_data(number_of_epochs)
            if not '.h5' in filename:
                save_file = filename + '.h5'
            else:
                save_file = filename
            self.model.save(save_file)
            self.test_data(error_type)


    def read_file(self, filename, delimiter, test_size):
        Rand=np.loadtxt(filename, delimiter= delimiter)
        number_of_columns = np.size(Rand, axis=1)
        X = Rand[:,0:number_of_columns - 1]
        y = Rand[:,number_of_columns - 1] * 10**11
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    

    def data_normalization(self):
        self.mean_train=np.mean(self.X_train,axis=0)
        self.std_train=np.std(self.X_train,axis=0)
        self.X_train=(self.X_train - self.mean_train) / self.std_train
        self.X_test =(self.X_test - self.mean_train) / self.std_train

        print((self.X_train.shape,self.y_train.shape,self.X_test.shape,self.y_test.shape))


    def create_model_dynamic(self, filename, number_of_epochs, error_type):

        min_error = 999999
        neurons = 0
        layers_size = 0
        number_of_neurons = [64, 128, 256, 384, 512]
        for i in number_of_neurons:
            for j in range(2, 21):
                layers_sizes = [i for k in range(j)]
                keras.backend.clear_session()
                self.build_model_dynamic(layers_sizes)
                print("model")
                self.train_data(number_of_epochs)
                print("train")
                error = self.test_data(error_type)
                if error < min_error:
                    min_error = error
                    neurons = i
                    layers_size = j
        for i in range(5, 12):
            layers_sizes = [1024 for k in range(i)]
            keras.backend.clear_session()
            self.build_model_dynamic(layers_sizes)
            print("model")
            self.train_data(number_of_epochs)
            print("train")
            error = self.test_data(error_type)
            if error < min_error:
                min_error = error
                neurons = i
                layers_size = j
        
        layers_sizes = [neurons for i in range(layers_size)]
        self.build_model_dynamic(layers_sizes)
        self.train_data(number_of_epochs)
        error = self.test_data(error_type)
        if not '.h5' in filename:
            save_file = filename + '.h5'
        else:
            save_file = filename
        self.model.save(save_file)


    def build_model_dynamic(self, layers_sizes):
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(layers_sizes[0], activation = tf.nn.relu, input_shape=(self.X_train.shape[1],)))
        for i in range(1, len(layers_sizes)):
            self.model.add(keras.layers.Dense(layers_sizes[i], activation = tf.nn.relu))
        self.model.add(keras.layers.Dense(1))

        #optimizer = tf.train.RMSPropOptimizer(0.001)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
        self.model.compile(loss=tf.keras.metrics.mean_squared_error,#'mean_squared_error',
                    optimizer=optimizer,
                    metrics=['mean_squared_error','mean_absolute_error'])


    def build_model(self):
        self.model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(self.X_train.shape[1],)),                 
        keras.layers.Dense(64,activation=tf.nn.relu ),
        keras.layers.Dense(1)
        ])
        #optimizer = tf.train.RMSPropOptimizer(0.001)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
        self.model.compile(loss=tf.keras.metrics.mean_squared_error,#'mean_squared_error',
                    optimizer=optimizer,
                    metrics=['mean_squared_error','mean_absolute_error'])
    

    def train_data(self, number_of_epochs = 2000):
        self.model.summary()
        self.model.fit(self.X_train, self.y_train, epochs=number_of_epochs, validation_split=0.2, verbose=0)
    

    def test_data(self, error_type):
        print((self.model.predict(self.X_test[0,:].reshape((1,-1)))))    
        print((self.y_test[0]))            
        # plot_history(history)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        [loss, mae ,mse] = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        print(("Testing set Mean Abs Error: ${:7.2f}".format(mae)))
        test_predictions = self.model.predict(self.X_test).flatten()

        print(('Mean Absolute Error:' + str(metrics.mean_absolute_error(self.y_test, test_predictions)) + '\n'))
        print(('Mean Squared Error:' + str(metrics.mean_squared_error(self.y_test, test_predictions)) + '\n'))
        print(('Root Mean Squared Error:' + str(np.sqrt(metrics.mean_squared_error(self.y_test, test_predictions))) + '\n'))
        print(("r^2 score:" + str(metrics.r2_score(self.y_test,test_predictions)) + '\n'))
        print(('Mean Absolute Percentage Error: ' + str(self.mean_absolute_percentage_error(self.y_test,test_predictions)) + '\n'))
        print(('Root Mean Square Percentage Error: ' + str(self.root_mean_square_percentage_error(self.y_test,test_predictions)) + '\n'))

        if error_type == 'mae':
            return metrics.mean_absolute_error(self.y_test, test_predictions)
        elif error_type == 'mse':
            return metrics.mean_squared_error(self.y_test, test_predictions)
        elif error_type == 'rmse':
            return np.sqrt(metrics.mean_squared_error(self.y_test, test_predictions))
        elif error_type == 'r2':
            return self.mean_absolute_percentage_error(self.y_test,test_predictions)
        elif error_type == 'mape':
            pass
        elif error_type == 'rmspe':
            return self.root_mean_square_percentage_error(self.y_test,test_predictions)

    
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        return np.sum((np.abs(y_true-y_pred)/y_true)/len(y_true)) * 100
    

    def root_mean_square_percentage_error(self, y_true, y_pred):
        return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='path to dataset file', type=str)
    parser.add_argument('-e', '--delimiter', help="kind of delimiter to separate data inside dataset file", type=str, default=',')
    parser.add_argument('-s', '--test_size', type=float, default=0.33, help="ratio of test dataset size")
    parser.add_argument('-m', '--model_name', type=str, help="model file name to store")
    parser.add_argument('-n', '--number_of_epochs', type=int, default=2000, help="number of epochs for training")
    parser.add_argument('-t', '--error_type', type=str, default='rmspe', choices=['mse', 'mae', 'rmse', 'r2', 'mape', 'rmspe'], help="type of error to identify accuracy of model over test data set")
    parser.add_argument('-r', '--training_type', type=str, default='dynamic', choices=['dynamic', 'static'], help="run over a set of neural network to find the bets network (dynamic) or run in a specific network(static)")


    #args = parser.parse_args()   

    print("Hello")

    
    ffnn = feed_forward_neural_network('/home/uk/Programming/Python/fpga_marm/input_files/AND_rand.txt', ',', 0.3, '/home/uk/Programming/Python/fpga_marm/input_files/AND.h5', 20, 'rmspe', 'dynamic')
    #ffnn = feed_forward_neural_network(args.dataset, args.delimiter, args.test_size, args.model_name, args.number_of_epochs, args.error_type, args.training_type)
