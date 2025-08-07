import json
import argparse  # TODO: Added argparse for command-line arguments
import pathlib   # TODO: Added pathlib for path handling
import pandas as pd

import dataset
import training
import model


def test_multiple_models(mother_folder, ds):
    con_mat = pd.DataFrame()
    for folder in mother_folder.glob('*'):
        if folder.is_dir():  # TODO: Changed isdir() to is_dir() for pathlib compatibility
            con_mat_i = training.test_model_from_folder(folder, ds)
            # TODO: Changed from `reset_index(names='label')` to a two-step reset and rename for backwards compatibility with older pandas versions.
            con_mat_i = con_mat_i.reset_index(drop=False)
            con_mat_i = con_mat_i.rename(columns={'index': 'label'})
            con_mat_i['fold'] = folder.name
            con_mat = pd.concat([con_mat, con_mat_i])
    con_matrix = con_mat.drop(columns=['fold'])
    con_matrix_avg = con_matrix.groupby('label').mean()
    model.plot_confusion_matrix(con_matrix_avg, save_path=mother_folder.joinpath('mean_confusion_matrix.png'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO: Added command-line argument parsing
    parser = argparse.ArgumentParser(description='Test trained CNN models')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to the folder containing trained models')
    args = parser.parse_args()
    
    # TODO: Commented out input() calls and replaced with command-line arguments
    # config_file = input('Where is the config file?')
    config_file = args.config
    
    # Read the config file
    with open(config_file) as f:  # TODO: Changed to use with statement for file handling
        config = json.load(f)

    # model_folder = input('Where is the folder to test?')
    model_folder = pathlib.Path(args.model_folder)  # TODO: Convert to Path object

    #  Load the test dataset
    spectro_ds = dataset.SpectrogramDataSet(data_dir=config['DATA_DIR'],
                                            categories=config['CATEGORIES'], join_cat=config["CATEGORIES_TO_JOIN"],
                                            locations=config['LOCATIONS'],
                                            corrected=config['USE_CORRECTED_DATASET'],
                                            samples_per_class=config['SAMPLES_PER_CLASS'])

    test_multiple_models(model_folder, spectro_ds)
