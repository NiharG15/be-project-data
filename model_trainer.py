""" Class for Training Effectiveness Regression Models """

import os
import sys
import time

import numpy as np
import pandas as pd

from tqdm import tqdm

import keras

from tensorpack.dataflow import MapDataComponent, BatchData

import dfdataflow as dfd

import gc

import logging
logging.basicConfig(level=logging.DEBUG, handlers=[
    logging.FileHandler('trainer_logs/model_trainer.{}.log'.format(time.strftime("%d.%m.%y-%I.%M.%S"))),
    logging.StreamHandler(sys.stdout)
    ])

class ModelTrainer(object):
    """
        A class used for training an effectiveness regression model.
    """

    def __init__(self, name, train_path, val_path, test_path, base_path='/home/nihargajre/datasets/ads/'):
        self.name = name
        self.train_df = pd.read_pickle(train_path)
        self.val_df = pd.read_pickle(val_path)
        self.test_df = pd.read_pickle(test_path)
        self.base_path = base_path


    def build_densenet_model(self, summary=False):
        """
            Build a DenseNet121 model using keras and add a linear output layer.
        """
        base_model = keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
        
        for layer in base_model.layers:
            layer.trainable = False
        
        x = base_model.layers[-1].output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1, activation='linear')(x)
        
        model = keras.models.Model(inputs=base_model.inputs, outputs=x)
        
        if summary:
            model.summary()
        
        return model

    def densenet_preprocess(self, img):
        """ Preprocessing function to be used with MapDataComponent """
        img = np.expand_dims(img, axis=0)
        img = keras.applications.densenet.preprocess_input(img.astype('float32'))
        return np.squeeze(img)

    def get_data_flow(self, df, batch_size=32):
        """ Returns a data flow for a DataFrame with preprocessing and batching """
        base_df = dfd.DFBaseDataFlow(df, lambda row: '{}{}/{}'.format(self.base_path, row.folder, row.file), 'similarity', resize=(224, 224))
        prepro_df = MapDataComponent(base_df, func=self.densenet_preprocess)
        batch_df = BatchData(prepro_df, batch_size)
        return batch_df
    
    def train(self):
        """ Name says it all """
        logging.info('Training: {}'.format(self.name))
        logging.info('Train Shape: {}'.format(self.train_df.shape))
        logging.info('Val Shape: {}'.format(self.val_df.shape))
        logging.info('Test Shape: {}'.format(self.test_df.shape))

        train_data = self.get_data_flow(self.train_df)
        val_data = self.get_data_flow(self.val_df)

        train_gen = train_data.get_data()
        val_gen = val_data.get_data()

        model = self.build_densenet_model()

        for layer in model.layers:
            layer.trainable = True
        
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001, decay=0.0001), loss='logcosh', metrics=['mse'])

        def lr_schedule(epoch):
            if epoch < 3:
                return 0.001
            elif epoch < 6:
                return 0.0001
            else:
                return 0.00001
        
        model_name_base = 'ft-regression-models/densenet-effectiveness-{}-'.format(self.name)
        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        checkpointer = keras.callbacks.ModelCheckpoint(model_name_base + '10e-{val_mean_squared_error:.03f}.hdf5',
                                                        monitor='val_mean_squared_error',
                                                        mode='min',
                                                        save_best_only=True)
        csvlogger = keras.callbacks.CSVLogger('csv_logs/train_{}.csv'.format(self.name))
        tensorboard = keras.callbacks.TensorBoard(log_dir='./reg-logs/model_trainer/{}'.format(self.name))

        history = model.fit_generator(generator=train_gen,
                                      steps_per_epoch=train_data.size(),
                                      validation_data=val_gen,
                                      validation_steps=val_data.size(),
                                      epochs=10,
                                      callbacks=[lr_scheduler, checkpointer, csvlogger, tensorboard])

        logging.info('-- Final Training Loss --')
        logging.info('Loss: {:.5f}, MSE: {:.5f}'.format(history.history['loss'][0], history.history['mean_squared_error'][0]))
        logging.info('-- Final Validation Loss --')
        logging.info('Loss: {:.5f}, MSE: {:.5f}'.format(history.history['val_loss'][0], history.history['val_mean_squared_error'][0]))

        test_data = self.get_data_flow(self.test_df)
        test_gen = test_data.get_data()

        pred_dfs = list()
        for _ in tqdm(range(test_data.size())):
            datas, labels = next(test_gen)
            pred_labels = model.predict(datas)
            pred_df = pd.DataFrame({'labels': labels, 'pred_labels': pred_labels.squeeze()})
            pred_df = pred_df.assign(error=abs(pred_df['labels'] - pred_df.pred_labels))

            pred_dfs.append(pred_df)
        
        final_pred_df = pd.concat(pred_dfs)
        logging.info('-- Test Prediction --')
        logging.info(str(final_pred_df.describe()))
        logging.info(str(final_pred_df.corr()))


def main():
    """ Driver function for all categories """
    categories =['restaurant', 'media', 'sports', 'chips', 'shopping',
                'alcohol', 'soda', 'electronics', 'clothing', 'beauty',
                'chocolate', 'travel']

    FILE_PREFIX = 'with_sim_class_df.pickle'

    for category in categories:
        gc.collect()
        train_path = 'regression_data/train_{}_{}'.format(category, FILE_PREFIX)
        val_path = 'regression_data/val_{}_{}'.format(category, FILE_PREFIX)
        test_path = 'regression_data/test_{}_{}'.format(category, FILE_PREFIX)

        trainer = ModelTrainer(category, train_path, val_path, test_path)
        trainer.train()


if __name__ == '__main__':
    main()