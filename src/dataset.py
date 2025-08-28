import os
import cv2
import sys  # TODO: used to detect non-interactive environments (e.g., Slurm) and disable tqdm to reduce log noise
import tensorflow as tf  # TODO: Added TensorFlow for tf.data pipeline to stream data without loading all into RAM
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 42

IMAGE_HEIGHT = 90
IMAGE_WIDTH = 30
N_CHANNELS = 1

CORRECTED_SAMPLES = 2500


class SpectrogramDataSet:
    def __init__(self, data_dir, categories, join_cat, locations, corrected,
                 samples_per_class='all'):
        """
        :param data_dir:
        :param categories:
        :param join_cat:
        :param locations:
        :param n_channels:
        :param corrected:
        :param samples_per_class: int, number of samples per class

        """
        self.locations = locations
        self.data_dir = data_dir
        self.categories = categories
        self.corrected = corrected
        self.join_cat = join_cat
        self.samples_per_class = samples_per_class

        # Create an understandable map for joined categories
        # and their corresponding int representation
        self.map_join = {}
        self.classes2int = {}
        self.int2class = []
        for join_class_name, classes_list in join_cat.items():
            self.classes2int[join_class_name] = len(self.classes2int.keys())
            self.int2class.append(join_class_name)
            for class_name in classes_list:
                self.map_join[class_name] = join_class_name

        for cat_name in self.categories:
            if cat_name not in self.map_join.keys():
                self.map_join[cat_name] = cat_name
                self.classes2int[cat_name] = len(self.classes2int)
                self.int2class.append(cat_name)

        self.n_classes = len(self.classes2int)
        print('These are the classes: ', self.classes2int)
        print('Which are formed by doing: ', self.map_join)

    def how_many_samples(self):
        """
        Return the number of samples if all is selected
        :return:
        """
        samples = 0
        for cat_i, category in enumerate(self.categories):
            # If Noise, select a random amount
            if category != 'Noise':
                path = os.path.join(self.data_dir, category)

                # Read all the images of that category
                samples += len(pd.Series(os.listdir(path)))
        return samples / (len(self.int2class) - 1)

    @staticmethod
    def reshape_images(images):
        """
        Reshape all the images to the specified height, width and channels during init of object

        :param images: list of images
        :return: array with normalized images (0 to 1) with the correct shape
        """
        X = np.array(images).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS)
        x = X / 255.0
        return x

    @staticmethod
    def join_paths_to_df(paths_train, paths_valid, paths_test):
        train_df = pd.DataFrame({'path': paths_train})
        valid_df = pd.DataFrame({'path': paths_valid})
        test_df = pd.DataFrame({'path': paths_test})
        train_df = train_df.assign(set='train')
        valid_df = valid_df.assign(set='valid')
        test_df = test_df.assign(set='test')
        paths_df = pd.concat([train_df, valid_df, test_df])
        return paths_df

    def prepare_all_dataset(self, test_size, valid_size, noise_ratio):
        """
        Will load all the labeled data (up to samples_per_class for each class)
        and some noise samples up to a certain noise_ratio

        :param test_size: float (0 to 1), percentage from the total data loaded to split randomly to test
        :param valid_size: float (0 to 1), percentage from the model (not test) data loaded to split randomly to
        validation
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list (of all the data)
        """
        paths_list = self.select_data(locations_to_exclude=None, noise_ratio=noise_ratio)
        x, y = self.load_from_file_list(file_list=paths_list)
        x_model, x_test, y_model, y_test, paths_model, paths_test = train_test_split(x, y, paths_list,
                                                                                     test_size=test_size, shuffle=True)
        x_train, x_valid, y_train, y_valid, paths_train, paths_valid = train_test_split(x_model, y_model, paths_model,
                                                                                        test_size=valid_size,
                                                                                        shuffle=True)
        paths_df = self.join_paths_to_df(paths_train, paths_valid, paths_test)

        return paths_df

    def prepare_blocked_dataset(self, blocked_location, valid_size, noise_ratio, noise_ratio_test):
        """
        Same than prepare_all_dataset but the test is decided by the blocked location
        :param blocked_location: string, name of the location to use for test and NOT for training or validation
        :param valid_size: float (0 to 1), percentage from the model (not test) data loaded to split randomly to
        validation
        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: x_train, y_train, x_valid, y_valid, x_test, y_test, paths_list (of all the data)
        """
        selected_locs = list(set(self.locations) - {blocked_location})
        print('selecting model data...')
        paths_list_model = self.select_data(locations_to_exclude=[blocked_location], noise_ratio=noise_ratio)
        y = self.read_labels_from_file_list(file_list=paths_list_model)
        paths_train, paths_valid = train_test_split(paths_list_model, stratify=y, test_size=valid_size, shuffle=True)

        print('selecting test data...')
        paths_test = self.select_data(locations_to_exclude=selected_locs, noise_ratio=noise_ratio_test)
        paths_df = self.join_paths_to_df(paths_train, paths_valid, paths_test)
        return paths_df

    def folds(self, noise_ratio, n_folds, valid_size):
        """
        Loop through the folds. The data will be first loaded (x, y) according to the samples per class and noise ratio.
        Once the data is loaded, it will be split between model and test according to the folds.
        Inside every fold, model split is split further into train and test, but this time randomly.

        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :param n_folds: number of folds to loop through
        :param valid_size: float (0 to 1) validation split from the model split
        :return: fold, x_train, y_train, x_vali, y_valid, x_test, y_test, paths_list (for all together)
        """
        paths_list = self.select_data(noise_ratio=noise_ratio, locations_to_exclude=None)
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
        y = self.read_labels_from_file_list(file_list=paths_list)
        for fold, (train_index, test_index) in enumerate(kfold.split(paths_list, y)):
            paths_model = paths_list[train_index]
            y_model = y[train_index]
            paths_test = paths_list[test_index]

            paths_train, paths_valid = train_test_split(paths_model, stratify=y_model,
                                                        test_size=valid_size, shuffle=True)
            paths_df = self.join_paths_to_df(paths_train, paths_valid, paths_test)
            yield fold, paths_df

    def select_files_category(self, category, samples_to_load, locations_to_exclude=None,
                              samples_to_exclude=None):
        """
        The function will return the selected data.
        For non-noise classes, the data included will be the first samples_to_load of the dataset (ordered),
        excluding the ones listed in samples_to_exclude and the ones corresponding to the locations to exclude,
        if any.

        :param category: string, category to load
        :param samples_to_load: int, number of samples per category (sum of all subclasses)
        :param locations_to_exclude:
        :param samples_to_exclude:
        :return:
        """
        subcats = dict((key, value) for (key, value) in self.map_join.items() if value == category)
        # Load all the available samples from the folder
        num_samples = []
        for subcat in subcats:
            path = os.path.join(self.data_dir, subcat)
            num_samples.append(len(os.listdir(path)))
        num_samples = np.array(num_samples)

        if samples_to_load != 'all':
            n_subcategories = len(subcats)
            samples_per_subcategory = round(samples_to_load / n_subcategories)
            # The sum of all the samples of the subclasses is smaller than the samples to load
            if num_samples.sum() < samples_to_load:
                print(f"[WARNING] Requested {samples_to_load} samples for category '{category}', but only {num_samples.sum()} available. Returning all available samples.")
                samples_to_load_per_subcat = num_samples

            # There is enough data at each subclass
            elif all(num_samples >= samples_per_subcategory):
                samples_to_load_per_subcat = np.repeat(samples_per_subcategory, n_subcategories)

            # If there is one of the subclasses which has less than its proportional part, check if we can load more
            # of the other subclasses, and do it iteratively until all the samples are reached
            #elif any(num_samples < samples_per_subcategory):
            #    samples_df = pd.DataFrame(index=subcats.keys(), columns=['samples_to_load', 'available', 'needed'])
            #    samples_df['available'] = num_samples
            #    samples_df['needed'] = samples_per_subcategory
            #    samples_df['samples_to_load'] = samples_df[['available', 'needed']].min()
            #    total_loaded = samples_df.samples_to_load.sum()
            #    while total_loaded < samples_to_load:
            #        leftover = samples_to_load - total_loaded
            #        samples_df['needed'] += round(leftover / (samples_df['available'] > samples_df['needed']).sum())
            #        samples_df['samples_to_load'] = samples_df[['available', 'needed']].min(axis=1)
            #        total_loaded = samples_df.samples_to_load.sum()
            #    samples_to_load_per_subcat = samples_df['samples_to_load'].values
            # Elegant redistribution: allocate fairly based on capacity
            elif any(num_samples < samples_per_subcategory):
                subcat_names = list(subcats.keys())
                print(f"  REDISTRIBUTION: {subcat_names} have {num_samples.tolist()} available")
                
                # Start with minimum allocation, then distribute remainder proportionally
                samples_to_load_per_subcat = np.minimum(num_samples, samples_per_subcategory)
                remainder = samples_to_load - samples_to_load_per_subcat.sum()
                print(f"  Initial: {dict(zip(subcat_names, samples_to_load_per_subcat))}, remainder: {remainder}")
                
                # Distribute remainder proportionally by capacity
                capacities = num_samples - samples_to_load_per_subcat
                if capacities.sum() > 0 and remainder > 0:
                    # Proportional distribution + handle rounding
                    extra = (capacities * remainder / capacities.sum()).astype(int)
                    samples_to_load_per_subcat += extra
                    
                    # Distribute any leftover from rounding to highest-capacity subcategories
                    leftover = remainder - extra.sum()
                    for idx in np.argsort(capacities)[::-1][:leftover]:
                        if samples_to_load_per_subcat[idx] < num_samples[idx]:
                            samples_to_load_per_subcat[idx] += 1
                    
                    print(f"  Added: {dict(zip(subcat_names, extra))}")
                
                print(f"  Final: {dict(zip(subcat_names, samples_to_load_per_subcat))}, total: {samples_to_load_per_subcat.sum()}")
        else:
            samples_to_load_per_subcat = num_samples

        # DEBUG: Detailed sample allocation information
        print(f"=== DEBUG: Category {category} ===")
        print(f"  Requested total: {samples_to_load}")
        print(f"  Subcategories: {list(subcats.keys())}")
        print(f"  Available per subcat: {dict(zip(subcats.keys(), num_samples))}")
        print(f"  Allocated per subcat: {dict(zip(subcats.keys(), samples_to_load_per_subcat))}")
        print(f"  Total allocated: {samples_to_load_per_subcat.sum()}")
        if samples_to_load != 'all':
            print(f"  Target vs Actual: {samples_to_load} vs {samples_to_load_per_subcat.sum()}")
            if samples_to_load_per_subcat.sum() != samples_to_load:
                print(f"  *** MISMATCH DETECTED! Difference: {samples_to_load_per_subcat.sum() - samples_to_load} ***")
        print()

        total_selected_subcat = []
        for i, subcat in enumerate(subcats):
            path = os.path.join(self.data_dir, subcat)

            # Read all the images of that category
            images_per_subcat = pd.Series(os.listdir(path))
            images_per_subcat = shuffle(images_per_subcat, random_state=SHUFFLE_SEED)

            # If there are one or more locations to exclude, exclude them from the list!
            if locations_to_exclude is not None:
                for loc in locations_to_exclude:
                    images_per_subcat = images_per_subcat.loc[~images_per_subcat.str.contains(loc)]

            if samples_to_exclude is not None:
                images_per_subcat = images_per_subcat.loc[~images_per_subcat.isin(samples_to_exclude)]

            # If the dataset is corrected, exclude the corrections
            if len(images_per_subcat) > 0:
                if samples_to_load == 'all':
                    last_img = -1
                else:
                    if self.corrected:
                        # Sort the images, we only want the n first ones
                        order = images_per_subcat.str.split('_', expand=True)[0].astype(int)
                        order = order.sort_values()
                        images_per_subcat = images_per_subcat.reindex(order.index)
                    last_img = samples_to_load_per_subcat[i]

                selected_images = images_per_subcat.iloc[:last_img]
            else:
                selected_images = images_per_subcat
            # If using the corrected dataset, eliminate the ones that are not correct
            if self.corrected and subcat != 'Noise':
                correction_path = os.path.join(self.data_dir, subcat + '2Noise.csv')
                if not os.path.exists(correction_path):
                    raise Exception(
                        'If you want to use the corrected dataset you should provide a csv file with the '
                        'corrections for each of the original classes')
                correction_csv = pd.read_csv(correction_path, header=None)
                all_images_joined_names = selected_images.str.split('_').str.join('')
                selected_images = selected_images.loc[~all_images_joined_names.isin(correction_csv[0])]
            total_selected_subcat += list(selected_images)
        return total_selected_subcat

    def select_data(self, noise_ratio, locations_to_exclude=None):
        """
        The function will return the selected data for all the classes together, shuffled.
        For non-noise classes, the data included will be the first samples_per_class of the dataset (ordered),
        excluding the ones corresponding to the locations to exclude, if any.

        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :param locations_to_exclude: list of locations to not load (for blocked testing)
        :return: x, y and paths
        """
        print("!-- Selecting data --")
        non_noise_paths = []
        # First, collect all non-noise samples
        for cat_i, category in enumerate(self.int2class):
            if category == 'Noise':
                continue
            samples_to_load = self.samples_per_class
            selected_paths = self.select_files_category(category, samples_to_load, locations_to_exclude)
            non_noise_paths += selected_paths

        n_non_noise = len(non_noise_paths)
        # Calculate the number of noise samples needed for the requested ratio
        if noise_ratio == 'all':
            n_noise = 'all'
        else:
            n_noise = int((noise_ratio * n_non_noise) / (1 - noise_ratio) + 0.5) if n_non_noise > 0 else 0

        # Now select noise samples
        noise_paths = self.select_files_category('Noise', n_noise, locations_to_exclude)

        total_paths = non_noise_paths + noise_paths
        total_paths = shuffle(total_paths)
        return total_paths

    def create_tf_dataset(self, paths_df, partition, batch_size, shuffle=True):
        """
        TODO: New method (does not exist in original) to build a tf.data pipeline for on-demand image loading.
        Keeps original in-memory loaders intact; this streams data efficiently with parallel I/O and prefetch.

        :param paths_df: DataFrame with 'path' and 'set' columns
        :param partition: 'train' | 'valid' | 'test'
        :param batch_size: batch size for batching the dataset
        :param shuffle: whether to shuffle (typically True for train, False for valid/test)
        :return: tf.data.Dataset yielding (image, label)
        """
        # Extract file names for the chosen partition
        paths_list = paths_df.loc[paths_df['set'] == partition, 'path'].values

        # Build full absolute paths and corresponding integer labels using existing label parsing
        # The class folder is derived from the filename (e.g., 1427_Location_Class.png -> 'Class')
        full_paths = [os.path.join(self.data_dir, p.split('_')[2].split('.')[0], p) for p in paths_list]
        labels = self.read_labels_from_file_list(paths_list)

        def _load_and_preprocess(path, label):
            # Read and decode as single-channel grayscale. If source is RGB, TF converts to one channel.
            image_bytes = tf.io.read_file(path)
            image = tf.image.decode_png(image_bytes, channels=1)
            # TODO: Model expects (IMAGE_WIDTH, IMAGE_HEIGHT, 1) i.e., (30,90,1). Decoded PNG is (H,W,1)=(90,30,1).
            # Transpose to match expected input.
            image = tf.transpose(image, perm=[1, 0, 2])
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]
            # Ensure static shape matches model expectation (IMAGE_WIDTH, IMAGE_HEIGHT, 1)
            image = tf.ensure_shape(image, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])
            return image, label

        ds = tf.data.Dataset.from_tensor_slices((full_paths, labels))
        # Parallel map decode + normalize
        ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        # TODO: Cache decoded tensors in memory so subsequent epochs don't re-read from disk.
        # This keeps first-epoch I/O, then serves from RAM, reducing repeated shuffle-buffer stalls.
        ds = ds.cache()

        # New: shuffle only for training in the tf.data pipeline (no equivalent in original codebase)
        if shuffle:
            # TODO: Tuned shuffle buffer to balance randomness and startup latency
            ds = ds.shuffle(buffer_size=min(2048, len(full_paths)))

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def read_labels_from_file_list(self, file_list):
        labels = []
        for img_name in file_list:
            category = img_name.split('_')[2].split('.')[0]
            joined_cat = self.map_join[category]

            # This part is for joined classes
            labels.append(self.classes2int[joined_cat])

        y = np.array(labels)

        return y

    def load_from_file_list(self, file_list):
        labels = []
        images = []
        # TODO: original tqdm without Slurm-aware disabling was too verbose for sbatch logs
        # for img_name in tqdm(file_list, total=len(file_list)):
        for img_name in tqdm(
                file_list,
                total=len(file_list),
                disable=(not sys.stdout.isatty()) or (os.environ.get('SLURM_JOB_ID') is not None),  # TODO: disable progress bar under sbatch
                leave=False,
                mininterval=5
        ):
            category = img_name.split('_')[2].split('.')[0]
            joined_cat = self.map_join[category]
            # TODO: Switch to grayscale decode to avoid RGB decode + manual averaging; reduces I/O and CPU.
            # img_array = cv2.imread(os.path.join(self.data_dir, category, img_name))
            # grey_image = np.mean(img_array, axis=2)
            # images.append(grey_image)
            img_gray = cv2.imread(os.path.join(self.data_dir, category, img_name), cv2.IMREAD_GRAYSCALE)  # TODO(lucas): direct grayscale decode for efficiency
            images.append(img_gray)  # TODO: already grayscale; no need to average channels
            # This part is for joined classes
            labels.append(self.classes2int[joined_cat])

        x = self.reshape_images(images)
        y = np.array(labels)

        return x, y

    def load_set_from_df(self, paths_df, partition):
        print('loading %s set in memory...' % partition)
        paths_list = paths_df.loc[paths_df['set'] == partition, 'path'].values
        return self.load_from_file_list(paths_list)

    def select_more_noise(self, paths_df, new_noise_ratio, partition):
        """
        Append to x_test and y_test more noise, NOT repeated (not the same samples).
        The amount of noise added is according to the new_noise_ratio.

        :param paths_df: pd.DataFrame with all the paths of the images corresponding used
        :param new_noise_ratio: new ratio (0 to 1) of noise from the total dataset
        :param partition: train, valid or test
        :return: updated x_test and y_test
        """
        noise_samples = self.get_noise_samples(new_noise_ratio)
        y_test = self.read_labels_from_file_list(paths_df.loc[paths_df['set'] == partition, 'path'].values)
        new_noise_samples = noise_samples - (y_test == self.classes2int['Noise']).sum()
        selected_paths = self.select_files_category('Noise', samples_to_load=new_noise_samples,
                                                    locations_to_exclude=None,
                                                    samples_to_exclude=paths_df['path'].values)

        new_paths_df = pd.DataFrame({'path': selected_paths})
        new_paths_df['set'] = partition

        paths_df = pd.concat([paths_df, new_paths_df])

        return paths_df

    def get_noise_samples(self, noise_ratio):
        """
        Compute how many noise samples are necessary to get the specified noise_ratio if each class has an amount of
        samples of samples_per_class

        :param noise_ratio: float (0 to 1), ratio of noise of the total dataset.
        :return: number of samples
        """
        if noise_ratio == 'all':
            return noise_ratio
        else:
            if self.samples_per_class == 'all':
                samples_per_class = self.how_many_samples()
            else:
                samples_per_class = self.samples_per_class
            return ((len(self.int2class) - 1) * samples_per_class * noise_ratio) / (1 - noise_ratio)

    def batch_load_from_df(self, data_split_df, data_split='test'):
        batch_size = 16

        data_split_test = data_split_df[data_split_df['set'] == data_split]
        images_for_test = pd.Series(data_split_test['path'])

        labels = []
        images = []

        # TODO: original tqdm without Slurm-aware disabling was too verbose for sbatch logs
        # for i, img_path in tqdm(enumerate(images_for_test), total=len(images_for_test)):
        for i, img_path in tqdm(
                enumerate(images_for_test),
                total=len(images_for_test),
                disable=(not sys.stdout.isatty()) or (os.environ.get('SLURM_JOB_ID') is not None),  # TODO: disable progress bar under sbatch
                leave=False,
                mininterval=5
        ):
            if (i % batch_size == 0) and (i != 0):
                x = self.reshape_images(images)
                y = np.array(labels)
                images = []
                labels = []

                yield x, y, self, images_for_test

            cat_folder = os.path.splitext(os.path.basename(img_path))[0].split('_')[2]
            # TODO: Switch to grayscale decode to avoid RGB decode + manual averaging during test batch loading.
            # img_array = cv2.imread(os.path.join(self.data_dir, cat_folder, img_path))

            # Not necessary if images already on the correct format
            # resized_image = cv2.resize(img_array, (self.image_width, self.image_height))
            img_gray = cv2.imread(os.path.join(self.data_dir, cat_folder, img_path), cv2.IMREAD_GRAYSCALE)  # TODO(lucas): direct grayscale decode for efficiency
            images.append(img_gray)  # TODO: already grayscale; no need to average channels

            # This part is for joined classes
            labels.append(self.classes2int[self.map_join[cat_folder]])

        x = self.reshape_images(images)
        y = np.array(labels)
        yield x, y, self, images_for_test

    def print_sample_counts(self, paths_df, partition_name="all"):
        """
        Print the number of samples for each class in the specified partition
        
        :param paths_df: DataFrame with 'path' and 'set' columns
        :param partition_name: 'train', 'valid', 'test', or 'all' to show counts for specific partition or all
        """
        if partition_name == "all":
            paths_list = paths_df['path'].values
            print(f"\n=== Sample counts for ALL partitions ===")
        else:
            paths_list = paths_df.loc[paths_df['set'] == partition_name, 'path'].values
            print(f"\n=== Sample counts for {partition_name.upper()} partition ===")
        
        labels = self.read_labels_from_file_list(paths_list)
        
        # Count samples per class
        class_counts = {}
        for class_name, class_int in self.classes2int.items():
            count = (labels == class_int).sum()
            class_counts[class_name] = count
            print(f"{class_name}: {count:,} samples")
        
        total_samples = len(labels)
        print(f"Total: {total_samples:,} samples")
        
        # Show percentages
        print("\nPercentages:")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"{class_name}: {percentage:.1f}%")
        print()
