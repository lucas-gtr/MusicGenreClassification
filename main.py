import torch

from src.model.model import Model
from src.dataset.split_dataset import split_dataset
from src.dataset.load_dataset import get_dataloader, create_label_index
from src.train import train_model
from src.evaluate import eval_model
from src.query import query_model
from config import batch_size, epoch_number, train_percentage, val_percentage, test_percentage,\
    sample_rate, sample_length, use_augmentation, num_chunks, device


import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='AI Model Training and Evaluation')

    subparsers = parser.add_subparsers(dest='command', help='Choose command')

    train_parser = subparsers.add_parser('train', help='Train AI model')
    train_parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    train_parser.add_argument('model_name', type=str, help='Name of the model file to save (.ckpt)')

    eval_parser = subparsers.add_parser('eval', help='Evaluate AI model')
    eval_parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    eval_parser.add_argument('model_name', type=str, help='Name of the model file to use (.ckpt)')

    query_parser = subparsers.add_parser('query', help='Query AI model with audio file')
    query_parser.add_argument('audio_file_path', type=str, help='Path to the audio file')
    query_parser.add_argument('model_name', type=str, help='Name of the model file to use (.ckpt)')

    args = parser.parse_args()

    if not args.model_name.endswith('.ckpt'):
        args.model_name += ".ckpt"

    model_path = os.path.join("trained_models", args.model_name)
    if not os.path.exists(model_path):
        print(f"""Error: Model {args.model_name} not found in folder "trained_models".""")
        return

    if args.command == 'train':
        if not os.path.exists(args.dataset_path):
            print(f"Error: {args.dataset_path} folder not found.")
            return
        if not is_dataset_split(args.dataset_path):
            split_dataset(args.dataset_path, train_percentage, val_percentage, test_percentage)

        print(f"Training model for dataset {args.dataset_path}, saving as {args.model_name}")

        label_to_index, index_to_label = create_label_index(os.path.join(args.dataset_path, 'train'))
        class_number = len(label_to_index)
        m = Model(class_number)

        train_loader = get_dataloader(
            args.dataset_path, 'train', label_to_index, sample_rate,
            sample_length, num_chunks, batch_size, use_augmentation)
        valid_loader = get_dataloader(
            args.dataset_path, 'val', label_to_index, sample_rate,
            sample_length, num_chunks, batch_size, use_augmentation)

        train_model(m, device, epoch_number, train_loader, valid_loader, args.model_name, index_to_label)

        print("Training completed successfully.")
    elif args.command == 'eval':
        if not os.path.exists(args.dataset_path):
            print(f"Error: {args.dataset_path} folder not found.")
            return
        if not is_dataset_split(args.dataset_path):
            split_dataset(args.dataset_path, train_percentage, val_percentage, test_percentage)

        print(f"Evaluating model using dataset {args.dataset_path}, using model {args.model_name}")

        model, index_to_label = load_model(model_path)
        label_to_index = {label: index for index, label in index_to_label.items()}

        test_loader = get_dataloader(
            args.dataset_path, 'test', label_to_index, sample_rate,
            sample_length, num_chunks, batch_size, use_augmentation)

        eval_model(model, device, test_loader, index_to_label)
        print("Evaluation completed successfully.")
    elif args.command == 'query':
        if not os.path.exists(args.audio_file_path):
            print("Error: Audio file path does not exist.")
            return

        print(f"Querying model with audio file {args.audio_file_path.split('/')[-1]}, using model {args.model_name}")

        model, index_to_label = load_model(model_path)
        print(f"{args.model_name} loaded!\n")

        query_model(model, device, args.audio_file_path, sample_rate, sample_length, num_chunks, index_to_label)
    else:
        print("Invalid command. Use 'train', 'eval', or 'query'.")


def is_dataset_split(dataset_path):
    """
    Check if the dataset directory contains separate directories for training, validation, and testing sets.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        bool: True if the dataset is split into 'train', 'val', and 'test' directories, False otherwise.
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path) or not os.path.isdir(split_path):
            return False

    return True


def load_model(model_path):
    """
    Load a trained model from the specified path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        torch.nn.Module: The loaded model.
        dict: A dictionary mapping genre indices to genre labels.
    """
    model_save = torch.load(model_path, map_location=device)
    index_to_label = model_save['labels']

    class_number = len(index_to_label)

    m = Model(class_number)
    m.load_state_dict(model_save['model_state_dict'])

    return m, index_to_label


if __name__ == "__main__":
    main()
