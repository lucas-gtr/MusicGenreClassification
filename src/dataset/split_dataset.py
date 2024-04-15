import os
import random
import shutil


def get_files_dict(dataset_path: str,
                   train_percent: float, val_percent: float, test_percent: float) -> tuple:
    """
    Divide the files in the dataset into training, validation, and test sets based on given percentages

    Args:
        dataset_path (str): Path to the dataset directory
        train_percent (float): Percentage of data to be used for training
        val_percent (float): Percentage of data to be used for validation
        test_percent (float): Percentage of data to be used for testing

    Returns:
        tuple: A tuple containing dictionaries representing the training, validation, and test sets
            Each dictionary has class labels as keys and lists of file paths as values
    """

    total_percent = round(train_percent + val_percent + test_percent, 2)
    assert total_percent == 1, "The sum of the percentages must be equal to 1."

    train_set = {}
    val_set = {}
    test_set = {}

    for label in os.listdir(dataset_path):
        label_folder = os.path.join(dataset_path, label)
        if os.path.isdir(label_folder):
            label_files = os.listdir(label_folder)
            random.shuffle(label_files)

            num_files = len(label_files)

            num_train = int(train_percent * num_files)
            num_val = int(val_percent * num_files)

            # Split files into train, val, and test
            train_set[label] = [os.path.join(label_folder, file) for file in label_files[:num_train]]
            val_set[label] = [os.path.join(label_folder, file) for file in label_files[num_train:num_train + num_val]]
            test_set[label] = [os.path.join(label_folder, file) for file in label_files[num_train + num_val:]]

    return train_set, val_set, test_set


def move_files_to_folders(dataset_path: str, train_dict: dict, val_dict: dict, test_dict: dict):
    """
    Move files to their respective folders (train, val, test) based on the provided dictionaries

    Args:
        dataset_path (str): Path to the dataset directory
        train_dict (dict): Dictionary containing class labels as keys and lists of file paths for training as values
        val_dict (dict): Dictionary containing class labels as keys and lists of file paths for validation as values
        test_dict (dict): Dictionary containing class labels as keys and lists of file paths for testing as values
    """
    for key, file_list in train_dict.items():
        target_folder = os.path.join(dataset_path, 'train', key)
        os.makedirs(target_folder, exist_ok=True)
        for file_path in file_list:
            shutil.move(file_path, target_folder)

    for key, file_list in val_dict.items():
        target_folder = os.path.join(dataset_path, 'val', key)
        os.makedirs(target_folder, exist_ok=True)
        for file_path in file_list:
            shutil.move(file_path, target_folder)

    for key, file_list in test_dict.items():
        target_folder = os.path.join(dataset_path, 'test', key)
        os.makedirs(target_folder, exist_ok=True)
        for file_path in file_list:
            shutil.move(file_path, target_folder)
        os.rmdir(os.path.join(dataset_path, key))


def split_dataset(dataset_path: str,
                   train_percent: float, val_percent: float, test_percent: float):
    """
    Split the dataset into training, validation, and test sets and move files accordingly

    Args:
        dataset_path (str): Path to the dataset directory
        train_percent (float, optional): Percentage of data to be used for training
        val_percent (float, optional): Percentage of data to be used for validation
        test_percent (float, optional): Percentage of data to be used for testing
    """
    train_dict, val_dict, test_dict = get_files_dict(dataset_path,
                                                     train_percent, val_percent, test_percent)

    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    move_files_to_folders(dataset_path, train_dict, val_dict, test_dict)
