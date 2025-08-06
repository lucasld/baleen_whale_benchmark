# Import required modules
import json
import pandas as pd
import os
import numpy as np
import argparse  # TODO: Added argparse for command-line arguments


def main():
    # TODO: Added command-line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate CNN model predictions')
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions (csv file)')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth (csv)')
    parser.add_argument('--label_list', type=str, required=True, help='Path to list of classes (json)')
    parser.add_argument('--output_path', type=str, required=True, help='Where to store the results (folder)')
    args = parser.parse_args()
    
    # TODO: Commented out input() calls and replaced with command-line arguments
    # Output file from test with the noise class listed in the last column
    # raw_detections_path = input('Path to predictions (csv file)')
    raw_detections_path = args.predictions

    # Ground truth file with the noise class listed in the last column
    # test_annots_path = input('Path to ground truth (csv)')
    test_annots_path = args.ground_truth

    # Label list file where the order of labels is according to the columns in the predictions
    # and ground truth and noise being the last label in the list
    # label_list_path = input('Path to list of classes (json)')
    label_list_path = args.label_list

    # Output path for results file
    # output_path = input('Where should we store the results (folder)?')
    output_path = args.output_path

    # Read true coverage, predictions and list of labels
    f = open(label_list_path)
    label_list = json.load(f)

    # TODO: Read predictions and ground truth CSVs into pandas DataFrames.
    # The predictions file contains the model's output, and the ground truth file contains the actual labels.
    # We are no longer using index_col=0 as the first column is an unnamed index.
    predictions_df = pd.read_csv(raw_detections_path)
    ground_truth_df = pd.read_csv(test_annots_path)

    # TODO: The ground truth file contains train, validation, and test sets.
    # We filter it to only include the test set to match the predictions file.
    ground_truth_df = ground_truth_df[ground_truth_df['set'] == 'test']

    # TODO: The ground truth file does not contain one-hot encoded labels.
    # We need to create them from the file paths.
    # We'll create a reverse mapping from the label string to the integer representation.
    class_to_int = label_list
    int_to_class = {i: v for v, i in class_to_int.items()}
    
    # TODO: Create a reverse map from joined categories to original categories.
    # This is needed to extract the label from the filename.
    # This logic is based on the dataset.py script.
    join_cat = {
        "20Plus": "20Hz20Plus",
        "20Hz": "20Hz20Plus",
        "A": "ABZ",
        "B": "ABZ",
        "Z": "ABZ",
        "D": "DDswp",
        "Dswp": "DDswp",
        "Noise": "Noise"
    }
    
    # Add other classes that are not joined
    for cat in class_to_int.keys():
        if cat not in join_cat.values():
             join_cat[cat] = cat
    
    # Create reverse map
    map_join = {}
    for k, v in join_cat.items():
        if v not in map_join:
            map_join[v] = []
        map_join[v].append(k)

    def get_label_from_path(path):
        # e.g. 1427_BallenyIslands2015_20Hz.png -> 20Hz
        label_str = os.path.basename(path).split('_')[2].split('.')[0]
        # e.g. 20Hz -> 20Hz20Plus
        joined_label = None
        for jl, ol in join_cat.items():
            if label_str == jl:
                joined_label = ol
                break
        if joined_label is None:
             # It might be that the label is already a joined label
             if label_str in class_to_int:
                 joined_label = label_str
             else:
                raise ValueError(f"Cannot find a joined label for {label_str}")

        return class_to_int[joined_label]

    ground_truth_df['label'] = ground_truth_df['path'].apply(get_label_from_path)
    
    # TODO: Convert the single label column to a one-hot encoded format (true_coverage).
    true_coverage = np.zeros((len(ground_truth_df), len(label_list)))
    for i, label in enumerate(ground_truth_df['label']):
        true_coverage[i, label] = 1

    # TODO: Merge the predictions and ground truth DataFrames on the 'path' column.
    # This ensures that each prediction is aligned with its corresponding ground truth label.
    merged_df = pd.merge(predictions_df, ground_truth_df, on='path')

    # TODO: Extract the numpy arrays for labels (predictions) and true_coverage from the merged dataframe.
    # The columns with integer names ('0', '1', etc.) are the prediction probabilities.
    label_columns = [str(i) for i in range(len(label_list))]
    labels = merged_df[label_columns].to_numpy()

    # Create confusion matrix for segments with potentially multiple labels
    confusion_matrix = np.zeros((len(label_list) + 1, len(label_list) + 1), dtype=int)
    for true_label in range(len(label_list)):
        for predicted_label in range(len(label_list)):
            confusion_matrix[true_label, predicted_label] = sum(
                labels[(true_coverage[:, true_label] == 1) & (true_coverage[:, predicted_label] != 1), predicted_label])

        confusion_matrix[true_label, true_label] = sum(labels[true_coverage[:, true_label] == 1, true_label])
        confusion_matrix[true_label, -1] = sum(true_coverage[:, true_label])
        confusion_matrix[-1, true_label] = sum(labels[:, true_label])

    # Extract diagonal and non-diagonal from confusion matrix
    dia = np.diag(confusion_matrix)
    non_dia_ind = ~np.eye(confusion_matrix[0:-2, 0:-2].shape[0], dtype=bool)
    non_dia = confusion_matrix[0:-2, 0:-2]
    non_dia = non_dia[np.array(non_dia_ind)]
    non_dia = non_dia.reshape([(len(label_list) - 1), int(len(non_dia) / (len(label_list) - 1))])

    # Extract single TCRs, NMRs, and CMRs per entry in the confusion matrix
    TCRs = []
    NMRs = []
    CMRs = []
    for classes in range(len(label_list) - 1):
        TCRs.append(dia[classes] / confusion_matrix[classes, -1])
        NMRs.append(confusion_matrix[-2, classes])
        for cols in range(np.shape(non_dia)[1]):
            CMRs.append(non_dia[classes, cols] / confusion_matrix[classes, -1])

    # Calculate TCR, NMR, CMR, and F value
    TCR = np.average(TCRs)
    NMR = sum(NMRs) / confusion_matrix[-2, -1]
    CMR = np.average(CMRs)
    F = np.average([TCR, (1 - NMR), (1 - NMR), (1 - CMR)])

    # Save confusion matrix and evaluation metrics to csv
    DF = pd.DataFrame(confusion_matrix)
    model_predictions = os.path.splitext(os.path.splitext(os.path.basename(raw_detections_path))[0])[0]
    DF.to_csv(
        os.path.join(output_path, '.'.join([''.join([model_predictions, '_confusion', '_NMR=', NMR.round(2).astype(str),
                                                     '_CMR=', CMR.round(2).astype(str), '_TCR=',
                                                     TCR.round(2).astype(str), '_F=', F.round(2).astype(str)]),
                                            'csv'])))
    
    # TODO: Added print statement to show metrics
    print(f"Evaluation complete. Results saved to {output_path}")
    print(f"TCR: {TCR:.4f}, NMR: {NMR:.4f}, CMR: {CMR:.4f}, F: {F:.4f}")


if __name__ == "__main__":
    main()
