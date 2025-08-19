import pandas as pd
import os

import model


def create_and_train_model(save_path, paths_df, ds, config, model_name):
    """
    Create and train model, from config

    :param save_path:
    :param paths_df:
    :param ds:
    :param config:
    :param model_name:
    :return:
    """
    print(config['CATEGORIES'])
    m = model.Model(save_path=save_path, categories=config['CATEGORIES'], model_name=model_name)

    m.create(n_classes=ds.n_classes, batch_size=config['BATCH_SIZE'])
    # TODO: Switched to tf.data streaming to avoid loading entire datasets into memory; original code below kept for reference.
    # x_train, y_train = ds.load_set_from_df(paths_df, 'train')
    # x_valid, y_valid = ds.load_set_from_df(paths_df, 'valid')
    # history = m.train(x_train, y_train, x_valid, y_valid, batch_size=config['BATCH_SIZE'], epochs=config['EPOCHS'],
    #                   loss_function=config['loss_function'], early_stop=config['early_stop'],
    #                   monitoring_metric=config['monitoring_metric'],
    #                   monitoring_direction=config['monitoring_direction'],
    #                   class_weights=config['CLASS_WEIGHTS'], learning_rate=config['learning_rate'])

    # Build streaming datasets
    train_ds = ds.create_tf_dataset(paths_df, 'train', config['BATCH_SIZE'], shuffle=True)  # TODO: use tf.data to stream training batches
    valid_ds = ds.create_tf_dataset(paths_df, 'valid', config['BATCH_SIZE'], shuffle=False)  # TODO: stream validation batches

    # Compute labels and step counts without loading images
    train_paths = paths_df.loc[paths_df['set'] == 'train', 'path'].values  # TODO: derive labels for class weights only
    valid_paths = paths_df.loc[paths_df['set'] == 'valid', 'path'].values
    y_train = ds.read_labels_from_file_list(train_paths)  # TODO: compute class weights from labels

    steps_per_epoch = int((len(train_paths) + config['BATCH_SIZE'] - 1) / config['BATCH_SIZE'])  # TODO: steps for dataset-based training
    validation_steps = int((len(valid_paths) + config['BATCH_SIZE'] - 1) / config['BATCH_SIZE'])

    # TODO: Added concise run header for this model to improve .out readability (sizes and steps).
    print(f"Train samples: {len(train_paths)}, Valid samples: {len(valid_paths)}, Batch size: {config['BATCH_SIZE']}, Steps/epoch: {steps_per_epoch}, Val steps: {validation_steps}")

    history = m.train_with_datasets(
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        y_train_labels=y_train,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=config['EPOCHS'],
        loss_function=config['loss_function'],
        early_stop=config['early_stop'],
        monitoring_metric=config['monitoring_metric'],
        monitoring_direction=config['monitoring_direction'],
        class_weights=config['CLASS_WEIGHTS'],
        learning_rate=config['learning_rate']
    )
    m.plot_training_metrics(history, chosen_metric=config['monitoring_metric'])
    m.save()
    paths_df.to_csv(m.log_path.joinpath('data_used_%s.csv' % model_name))
    return m


def run_multiple_models(log_path, paths_df, config, fold, ds, perform_test=False):
    """
    Create all the models with the specified noise on the training set
    train it and test it according to the specifications in config
    """
    if type(config['NOISE_RATIO']) == list:
        noise_to_train = config['NOISE_RATIO']
    else:
        noise_to_train = [config['NOISE_RATIO']]

    scores = pd.DataFrame()
    con_mat_df = pd.DataFrame()
    for i, noise in enumerate(noise_to_train):
        if i == 0:
            noise_before = noise
        else:
            noise_before = noise_to_train[i - 1]
        model_name = 'fold_%s_noise_%s' % (fold, noise)
        paths_df1, noise = select_more_noise(paths_df, 'train', noise_before, noise, config, ds)
        paths_df2, noise = select_more_noise(paths_df1, 'valid', noise_before, noise, config, ds)
        # TODO: Added concise fold header to mark start of a model run with key parameters.
        print(f"=== Model: {model_name} | Fold: {fold} | Noise train: {noise} ===")
        cnn_model = create_and_train_model(log_path, paths_df2, ds, config, model_name=model_name)
        if perform_test:
            scores_i, con_mat_i = test_model_multiple_noise(cnn_model, paths_df, config, ds, fold, log_path)
            scores = pd.concat([scores, scores_i])
            con_mat_df = pd.concat([con_mat_df, con_mat_i])

    return scores, con_mat_df


def test_model_multiple_noise(cnn_model, paths_df, config, ds, fold, log_path):
    if type(config['NOISE_RATIO_TEST']) == list:
        noise_to_test = config['NOISE_RATIO_TEST']
    else:
        noise_to_test = [config['NOISE_RATIO_TEST']]

    last_noise = noise_to_test[0]
    scores_i = pd.DataFrame()
    con_mat_df = pd.DataFrame()
    for noise_test in noise_to_test:
        paths_df, train_noise = select_more_noise(paths_df, 'test', last_noise, noise_test, config, ds)
        last_noise = noise_test
        scores_noise, con_mat_noise, predictions = cnn_model.new_test(ds, data_split_df=paths_df)

        model.plot_confusion_matrix(con_mat_noise, log_path.joinpath('confusion_matrix_fold%s_noise%s_noise%s.png' %
                                                                     (fold, train_noise, noise_test)))
        # Add the metadata
        scores_noise['noise_percentage_train'] = train_noise
        scores_noise['noise_percentage_test'] = noise_test

        scores_i = pd.concat([scores_i, scores_noise])
        con_mat_df = pd.concat([con_mat_df, con_mat_noise])

        paths_df.to_csv(
            log_path.joinpath('data_used_fold%s_noise%s_noise%s.csv' % (fold, train_noise, noise_test)))
        predictions.to_csv(
            log_path.joinpath('predictions_fold%s_noise%s_noise%s.csv' % (fold, train_noise, noise_test)))

    return scores_i, con_mat_df


def test_model_from_folder(folder_path, ds):
    print("Loading model...")
    print("    model_name: ", folder_path.name)
    print("    save_path: ", folder_path.parent)
    print("    categories: ", ds.categories)
    m = model.Model(save_path=folder_path.parent, categories=ds.categories, model_name=folder_path.name)
    m.load_existing()
    print("Model loaded successfully. ")
    csv_split_file = m.log_path.joinpath('data_used_%s.csv' % m.model_name)
    # Original (non-existent in our Model API):
    # con_mat = m.predict_full_ds(ds, csv_split_file)
    # TODO: Use existing batch tester; it expects a DataFrame
    data_split_df = pd.read_csv(csv_split_file)
    print("Testing model...")
    # _, con_mat, _ = m.test_in_batches(ds, data_split_df)
    _, con_mat, _ = m.new_test(ds, data_split_df)

    return con_mat


def select_more_noise(paths_df, phase, noise, new_noise, config, ds):
    if noise == new_noise:
        return paths_df, noise
    elif new_noise == 'all':
        paths_df = ds.select_more_noise(paths_df, new_noise, phase)
    elif noise != 'all':
        if phase == 'test':
            if new_noise > noise:
                paths_df = ds.select_more_noise(paths_df, new_noise, phase)
            elif new_noise < config['NOISE_RATIO'][0]:
                print(
                    'Noise percentage lower than in first training. '
                    'Not considering it and testing on the first training ratio'
                )
                noise2 = config['NOISE_RATIO'][0]

        else:
            if new_noise > noise:
                paths_df = ds.select_more_noise(paths_df, new_noise, phase)

    return paths_df, noise2
