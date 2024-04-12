import os
import random
import librosa
import numpy as np
import torch
from torch.utils import data
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)


class AudioDataset(data.Dataset):
    def __init__(self, dataset_list: list, split: str, sample_rate: int,
                 sample_length: int, num_chunks: int, use_augmentation: bool):
        """
        Dataset class for loading audio data.

        Args:
            dataset_list (list): List of tuples containing file paths and corresponding labels
            split (str): Split name, e.g., 'train', 'val', or 'test'
            sample_rate (int): Sample rate for audio files
            sample_length (int): Length of samples (in seconds) to be extracted from audio files
            num_chunks (int): Number of chunks to split audio into (for validation and test sets)
            use_augmentation (bool): Whether to apply data augmentation techniques
        """
        self.split = split
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * sample_length)
        self.num_chunks = num_chunks
        self.use_augmentation = use_augmentation
        self.dataset_list = dataset_list
        if use_augmentation:
            self.augmentation = self._get_augmentations()

    def _get_augmentations(self):
        """
        Define data augmentation transformations
        """
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=self.sample_rate)], p=0.8),
            RandomApply([Delay(sample_rate=self.sample_rate)], p=0.5),
            RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=self.sample_rate)], p=0.4),
            RandomApply([Reverb(sample_rate=self.sample_rate)], p=0.3),
        ]
        return Compose(transforms=transforms)

    def _adjust_audio_length(self, wav):
        """
        Adjust the length of audio samples and the number of chunks based on the split

        Args:
            wav (np.ndarray): Audio waveform

        Returns:
            np.ndarray: Adjusted audio waveform
        """
        if self.split == 'train':
            random_index = random.randint(0, len(wav) - self.num_samples - 1)
            wav = wav[random_index: random_index + self.num_samples]
        else:
            hop = (len(wav) - self.num_samples) // self.num_chunks
            wav = np.array([wav[i * hop: i * hop + self.num_samples] for i in range(self.num_chunks)])
        return wav

    def __getitem__(self, index: int):
        """
        Get audio sample and its corresponding label

        Args:
            index (int): Index of the sample

        Returns:
            tuple: Audio sample and its label
        """
        audio_filename, genre_index = self.dataset_list[index]

        wav, _ = librosa.load(audio_filename, sr=self.sample_rate)

        if wav.ndim == 2:
            wav = (wav[:, 0] + wav[:, 1]) / 2
        wav = self._adjust_audio_length(wav).astype('float32')

        if self.use_augmentation:
            wav = self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()

        return wav, genre_index

    def __len__(self):
        """
        Get the total number of samples in the dataset

        Returns:
            int: Total number of audio files in dataset
        """
        return len(self.dataset_list)


def get_dataloader(directory: str, split: str, label_to_index: dict, sample_rate: int, sample_length: int,
                    num_chunks: int, batch_size: int, use_augmentation: bool):
    """
    Create data loaders for training, validation, and testing

    Args:
        directory (str): Path to the dataset directory
        split (str): Split name, e.g., 'train', 'val', or 'test'
        label_to_index (dict): Mapping of class labels to indices
        sample_rate (int): Sample rate for audio files
        sample_length (int): Length of samples (in seconds) to be extracted from audio files
        num_chunks (int): Number of chunks to split audio into for validation and test sets
        batch_size (int): Batch size
        use_augmentation (bool): Whether to apply data augmentation techniques

    Returns:
        list: List of data loaders for training, validation, and testing
    """
    audio_extensions = [".mp3", ".wav", ".ogg", ".flac", ".aac", ".au"]

    split_directory = os.path.join(directory, split)
    dataset_list = []

    for root, dirs, files in os.walk(split_directory):
        audio_files = [f for f in files if os.path.splitext(f)[1].lower() in audio_extensions]
        if audio_files:
            genre_label = os.path.basename(root)
            genre_index = label_to_index[genre_label]

            # Add the audio file path and its genre index to the dataset list
            dataset_list.extend([(os.path.join(root, file), genre_index) for file in audio_files])

    is_shuffle = True if (split == 'train') else False
    batch_size = batch_size if (split == 'train') else (batch_size // num_chunks)

    data_loader = data.DataLoader(dataset=AudioDataset(dataset_list,
                                                       split,
                                                       sample_rate,
                                                       sample_length,
                                                       num_chunks,
                                                       use_augmentation),
                                  batch_size=batch_size,
                                  shuffle=is_shuffle,
                                  drop_last=False,
                                  num_workers=0)

    return data_loader


def create_label_index(directory):
    """
    Create mapping of class labels to indices and vice versa

    Args:
        directory (str): Path to the dataset directory

    Returns:
        tuple: Tuple containing label_to_index and index_to_label dictionaries
    """
    label_to_index = {}
    index_to_label = {}
    index = 0

    for folder_name in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, folder_name)):
            label_to_index[folder_name] = index
            index_to_label[index] = folder_name
            index += 1

    return label_to_index, index_to_label
