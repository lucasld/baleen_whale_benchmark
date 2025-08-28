import datetime
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers
# import visualkeras

from PIL import ImageFont

import dataset
import metrics
from custom_loss_function import custom_cross_entropy
from models import build_model_architecture, get_available_models


class Model:
    def __init__(self, save_path, categories, model_name):
        """
        Initialize the model with the folder
        """
        self.save_path = save_path

        self.model_name = model_name
        self.log_path = save_path.joinpath(model_name)
        if not self.log_path.exists():
            os.mkdir(str(self.log_path))

        detection_metrics = metrics.ImbalancedDetectionMatrix(noise_class_name='Noise', classes_names=categories)
        tf.config.run_functions_eagerly(True)
        self.metrics = [
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            detection_metrics.imbalanced_metric,
            detection_metrics.noise_misclas_rate,
            detection_metrics.call_avg_tpr
        ]
        # TODO: Create metrics_dict for load_model which needs a dict, not a list of metrics
        self.metrics_dict = {m.name if hasattr(m, 'name') else m.__name__: m for m in self.metrics}
        # TODO: Add custom loss function to the custom objects dictionary for loading the model.
        self.metrics_dict['custom_cross_entropy'] = custom_cross_entropy

        # self.metrics = {'imbalanced_metric': detection_metrics.imbalanced_metric,
        #                 'noise_misclas_rate': detection_metrics.noise_misclas_rate,
        #                 'call_avg_tpr': detection_metrics.call_avg_tpr,
        #                 'f1_score': metrics.f1_score,
        #                 'recall_score': metrics.recall_score,
        #                 'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')}
        # Model still to be created or loaded
        self.model = None

    def create(self, n_classes, batch_size, model_architecture="IdilCNN"):
        """
        Create the model. Stores a summary of the model in the log_path, called model_summary.txt
        :param n_classes: number of classes
        :param batch_size: int, batch size
        :param model_architecture: string, architecture name (e.g., "IdilCNN", "SimpleCNN", "ResNet50")
        :return: created model
        """
        try:
            # Use the model factory to build the specified architecture
            model = build_model_architecture(model_architecture, n_classes, batch_size)
        except ValueError as e:
            # If architecture not found, fall back to original IdilCNN and show available options
            print(f"Error: {e}")
            print(f"Falling back to original IdilCNN architecture.")
            print(f"Available architectures: {get_available_models()}")
            
            # Original IdilCNN implementation as fallback
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Conv2D(batch_size, kernel_size=(3, 3), activation='relu', padding="same",
                                             kernel_initializer='he_normal', input_shape=(dataset.IMAGE_WIDTH,
                                                                                          dataset.IMAGE_HEIGHT, 1)))

            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Conv2D(batch_size, kernel_size=(3, 3), activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(batch_size, kernel_size=5, strides=2, padding='same', activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(
                tf.keras.layers.Conv2D(batch_size * 2, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(
                tf.keras.layers.Conv2D(batch_size * 4, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.Dense(batch_size * 2, kernel_regularizer=regularizers.l2(0.001)))
            model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.Dropout(0.3))
            model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

        model_summary_path = self.log_path.joinpath('model_summary.txt')
        # Open the file
        with open(model_summary_path, 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        self.model = model

    def load_existing(self):
        # Load model without compilation (for evaluation only)
        self.model = tf.keras.models.load_model(self.log_path.joinpath('model'), compile=False)
        # Re-compile with standard metrics for evaluation
        self.model.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def train(self, x_train, y_train, x_valid, y_valid, batch_size, epochs, loss_function, early_stop,
              monitoring_metric, monitoring_direction, class_weights, learning_rate):
        """
        Train the model. Will store the logs of the training inside the folder model_log_path in a file called logs.csv
        :param x_train: x to train
        :param y_train: y to train
        :param x_valid: x to validate
        :param y_valid: y to validate
        :param batch_size: batch size
        :param epochs: number of epochs
        :param early_stop: number of no increase performance to stop
        :param monitoring_metric: string, metric to use to monitor performance
        :param class_weights: string, None or dict
        :param loss_function: string, loss function to use
        :param monitoring_direction: string, "auto", "min" or "max"
        :param learning_rate: float, initial learning rate for the exponential decay
        :return: history
        """
        # compute the class weights
        class_labels = np.unique(y_train)
        class_weights = compute_class_weight(class_weights, classes=class_labels, y=y_train)
        class_weights_dict = {}
        for label, weight in zip(class_labels, class_weights):
            class_weights_dict[label] = weight
        print('used class weights:', class_weights_dict)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=(len(x_train) / batch_size) * 10,
            decay_rate=0.5)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # metrics.NoiseMisclassificationRate(noise_class_name='Noise', classes_names=categories, name='noise_misclass'),
        # metrics.CallAvgTPR(noise_class_name='Noise', classes_names=categories, name='call_tpr')

        if loss_function == 'custom_cross_entropy':
            loss_function = custom_cross_entropy
        self.model.compile(loss=loss_function,
                           optimizer=opt,
                           metrics=self.metrics,
                           )

        model_save_filename = self.log_path.joinpath('checkpoints.weights.h5')  # Changed from 'checkpoints' to 'checkpoints.weights.h5' --- this is a change by lucas

        early_stopping_cb = keras.callbacks.EarlyStopping(monitor=monitoring_metric, mode=monitoring_direction,
                                                          patience=early_stop, restore_best_weights=True)
        mdl_checkpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename, save_weights_only=True,
                                                            monitor=monitoring_metric, save_best_only=True)
        # min_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-5,
        #     decay_steps=10000,
        #     decay_rate=1.0)
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=monitoring_metric,
        # factor=0.2,patience=8, min_lr= min_learning_rate)
        history_cb = tf.keras.callbacks.CSVLogger(self.log_path.joinpath('logs.csv'), separator=',', append=False)

        # TODO: Keras default verbose=1 prints per-batch progress lines that clutter Slurm logs; switch to per-epoch lines.
        # history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
        #                          validation_data=(x_valid, y_valid),
        #                          callbacks=[early_stopping_cb, mdl_checkpoint_cb, history_cb],
        #                          class_weight=class_weights_dict)
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=(x_valid, y_valid),
                                 callbacks=[early_stopping_cb, mdl_checkpoint_cb, history_cb],
                                 class_weight=class_weights_dict,
                                 verbose=2)

        return history

    def train_with_datasets(self, train_dataset, valid_dataset, y_train_labels, steps_per_epoch, validation_steps,
                            epochs, loss_function, early_stop, monitoring_metric, monitoring_direction,
                            class_weights, learning_rate):
        """
        TODO: New method to train using tf.data datasets to avoid full in-memory loads.
        Keeps original `train` method intact for backward compatibility.
        """
        # compute the class weights
        class_labels = np.unique(y_train_labels)
        class_weights_vals = compute_class_weight(class_weights, classes=class_labels, y=y_train_labels)
        class_weights_dict = {}
        for label, weight in zip(class_labels, class_weights_vals):
            class_weights_dict[label] = weight
        print('used class weights:', class_weights_dict)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=steps_per_epoch * 10,
            decay_rate=0.5)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        if loss_function == 'custom_cross_entropy':
            loss_function = custom_cross_entropy
        self.model.compile(loss=loss_function,
                           optimizer=opt,
                           metrics=self.metrics,
                           )

        model_save_filename = self.log_path.joinpath('checkpoints.weights.h5')

        early_stopping_cb = keras.callbacks.EarlyStopping(monitor=monitoring_metric, mode=monitoring_direction,
                                                          patience=early_stop, restore_best_weights=True)
        mdl_checkpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename, save_weights_only=True,
                                                            monitor=monitoring_metric, save_best_only=True)
        history_cb = tf.keras.callbacks.CSVLogger(self.log_path.joinpath('logs.csv'), separator=',', append=False)

        # TODO: Keras default verbose=1 prints per-batch progress lines that clutter Slurm logs; switch to per-epoch lines.
        # history = self.model.fit(train_dataset,
        #                          epochs=epochs,
        #                          steps_per_epoch=steps_per_epoch,
        #                          validation_data=valid_dataset,
        #                          validation_steps=validation_steps,
        #                          callbacks=[early_stopping_cb, mdl_checkpoint_cb, history_cb],
        #                          class_weight=class_weights_dict)
        # TODO: Added concise training header with key metrics names once at start.
        print(f"Metrics: {', '.join([m if isinstance(m, str) else (m.name if hasattr(m, 'name') else m.__name__) for m in self.model.metrics_names])}")
        history = self.model.fit(train_dataset,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=valid_dataset,
                                 validation_steps=validation_steps,
                                 callbacks=[early_stopping_cb, mdl_checkpoint_cb, history_cb],
                                 class_weight=class_weights_dict,
                                 verbose=2)

        # TODO: Added concise run footer summarizing key results for .out readability.
        best_epoch = np.argmin(history.history['val_loss']) + 1 if 'val_loss' in history.history else None
        if best_epoch is not None:
            print(f"Best epoch (val_loss): {best_epoch} | val_loss: {history.history['val_loss'][best_epoch-1]:.4f}")
        return history

    def test(self, x_test, y_test, categories, images_to_test):
        """
        Test the model on x_test and y_test.

        :param x_test: X
        :param y_test: y
        :param categories: names of the categories to create the confusion matrix
        :return: scores_df (DataFrame) and con_mat_df (DataFrame)
        """
        scores = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred = self.model.predict(x_test)
        preds = pd.concat(
            [pd.DataFrame(y_pred).reset_index(drop=True), pd.DataFrame(images_to_test).reset_index(drop=True)], axis=1)
        con_mat = self.get_scores(y_test, y_pred, preds, categories)

        con_mat_df = pd.DataFrame(con_mat, index=categories, columns=categories)
        scores_df = pd.DataFrame([scores], columns=self.model.metrics_names)
        return scores_df, con_mat_df, preds

    def new_test(self, ds, data_split_df, batch_size=32):
        """
        Test the model using tf.data streaming approach for consistency with training.
        
        :param ds: SpectrogramDataSet instance
        :param data_split_df: DataFrame with 'path' and 'set' columns
        :param batch_size: batch size for testing
        :return: scores_df (DataFrame), con_mat_df (DataFrame), preds (DataFrame)
        """
        # Create test dataset using streaming approach
        print("[DEBUG] Creating test dataset from DataFrame ...")
        test_ds = ds.create_tf_dataset(data_split_df, 'test', batch_size, shuffle=False)
        # Get test paths for images_to_test
        print("[DEBUG] Extracting test paths from DataFrame ...")
        test_paths = data_split_df.loc[data_split_df['set'] == 'test', 'path'].values
        
        # DEBUG: Print types and samples of test_paths and labels
        print("[DEBUG] test_paths type:", type(test_paths))
        print("[DEBUG] test_paths sample:", test_paths[:5] if len(test_paths) > 5 else test_paths)
        y_test = ds.read_labels_from_file_list(test_paths)
        print("[DEBUG] y_test type:", type(y_test))
        print("[DEBUG] y_test sample:", y_test[:5] if len(y_test) > 5 else y_test)
        print("[DEBUG] test_ds type:", type(test_ds))
        # DEBUG: Print a single batch from test_ds
        for batch in test_ds.take(1):
            print("[DEBUG] test_ds batch types:", [type(x) for x in batch])
            print("[DEBUG] test_ds batch shapes:", [x.shape for x in batch])
        
        # Print class sample counts for the current test set (for debugging)
        if hasattr(ds, 'print_sample_counts'):
            print("[DEBUG] Class sample counts for this test set:")
            ds.print_sample_counts(data_split_df, partition_name="test")
        # Evaluate using tf.data (gets proper metrics like original test method)
        print("[DEBUG] Calling model.evaluate on test_ds ...")
        scores = self.model.evaluate(test_ds, verbose=0)
        print("[DEBUG] model.evaluate returned:", scores)
        print("[DEBUG] model.metrics_names:", self.model.metrics_names)
        # Get predictions
        print("[DEBUG] Calling model.predict on test_ds ...")
        y_pred = self.model.predict(test_ds, verbose=0)
        print("[DEBUG] model.predict returned shape:", y_pred.shape)
        # Get true labels for confusion matrix
        y_test = ds.read_labels_from_file_list(test_paths)
        print("[DEBUG] y_test shape for confusion matrix:", y_test.shape)
        # Build predictions DataFrame (same format as original test method)
        preds = pd.concat(
            [pd.DataFrame(y_pred).reset_index(drop=True), 
             pd.DataFrame(test_paths, columns=['path']).reset_index(drop=True)], axis=1)
        print("[DEBUG] preds DataFrame shape:", preds.shape)
        # Use existing confusion matrix method (since get_scores is missing)
        con_mat_df = self.get_confusion_matrix(y_test, y_pred, categories=ds.int2class)
        print("[DEBUG] con_mat_df shape:", con_mat_df.shape)
        # Build scores DataFrame (same format as original test method)
        scores_df = pd.DataFrame([scores], columns=self.model.metrics_names)
        print("[DEBUG] scores_df shape:", scores_df.shape)
        return scores_df, con_mat_df, preds

    def save(self):
        self.model.save(self.log_path.joinpath('model'))

    @staticmethod
    def get_confusion_matrix(y_test, y_pred, categories):
        y_pred = np.argmax(y_pred, axis=1)
        con_mat = confusion_matrix(y_test, y_pred, labels=np.arange(len(categories)))

        con_mat_df = pd.DataFrame(con_mat, index=categories, columns=categories)
        return con_mat_df

    def test_in_batches(self, ds, data_split_df):
        y_pred_total = None
        y_test_total = None
        for x_test, y_test, ds, images_for_test in ds.batch_load_from_df(data_split_df, data_split='test'):
            y_pred = self.model.predict_on_batch(x_test)
            if y_pred_total is None:
                y_pred_total = y_pred
                y_test_total = y_test
            else:
                y_pred_total = np.concatenate([y_pred_total, y_pred])
                y_test_total = np.concatenate([y_test_total, y_test])

        con_mat_df = self.get_confusion_matrix(y_test_total, y_pred_total, categories=ds.int2class)

        preds = pd.concat([pd.DataFrame(y_pred_total), pd.DataFrame(images_for_test)], axis=1)
        scores = pd.DataFrame(y_pred_total)
        #preds.to_csv(self.log_path.joinpath('prediction.csv'))
        #con_mat_df.to_csv(self.log_path.joinpath('confusion_matrix'))
        return scores, con_mat_df, preds

    def plot_training_metrics(self, history, chosen_metric):
        """
        Plot the history evolution of the metrics of the model and store the images in the folder save_path with the
        correct name indicating which fold was it run

        :param history: history, output from tensorflow
        :param fold: which fold was it run on
        :param chosen_metric: monitoring metric
        :return:
        """
        training_acc = history.history[chosen_metric.split('val_')[1]]
        val_acc = history.history[chosen_metric]
        training_loss = history.history['loss']
        val_loss = history.history['val_loss']

        epoch_count = range(1, len(training_acc) + 1)
        plt.figure(figsize=(8, 8))
        plt.plot(epoch_count, training_acc, 'r--')
        plt.plot(epoch_count, val_acc, 'b-')
        plt.legend(['Training %s' % chosen_metric, 'Validation %s' % chosen_metric])
        plt.xlabel('Epoch')
        plt.ylabel(chosen_metric)
        plt.yticks(np.arange(0, 1, 0.05))
        plt.xticks(np.arange(0, max(epoch_count), 2))
        # plt.text(10, 0.01, "Test acc: " + str(scores[1]))
        plt.grid()
        plt.savefig(self.log_path.joinpath('training_%s_%s.png' % (chosen_metric, self.model_name)))
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.plot(epoch_count, training_loss, 'g--')
        plt.plot(epoch_count, val_loss, 'r-')
        plt.legend(['Training loss', 'Validation loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yticks(np.arange(0, max(val_loss), 0.2))
        plt.xticks(np.arange(0, max(epoch_count), 2))
        plt.grid()
        plt.savefig(self.log_path.joinpath('training_loss_%s.png' % self.model_name))
        plt.close()


def plot_confusion_matrix(con_mat_df, save_path):
    """
    Plot the confusion matrix and save it to save_path
    :param con_mat_df: DataFrame, output from get_model_scores
    :param save_path: str or path
    :return:
    """
    con_mat_norm = con_mat_df.astype('float').div(con_mat_df.sum(axis=1), axis=0)

    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(con_mat_norm, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    plt.savefig(save_path)
    plt.close()
