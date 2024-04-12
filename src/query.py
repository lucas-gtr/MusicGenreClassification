import librosa
import torch
import numpy as np
import torch.nn.functional as F


def query_model(model, device, file_path, sample_rate, sample_length, num_chunks, index_to_label):
    """
    Query the deep learning model with an audio file to predict its genre.

    Args:
        model (torch.nn.Module): The trained deep learning model.
        device (torch.device): The device to run the inference on (e.g., 'cuda' for GPU or 'cpu' for CPU).
        file_path (str): Path to the audio file.
        sample_rate (int): Sampling rate of the audio file.
        sample_length (float): Length of each audio sample (in seconds).
        num_chunks (int): Number of chunks to divide the audio file into for processing.
        index_to_label (dict): A dictionary mapping genre indices to genre labels.
    """
    model.eval()
    with torch.no_grad():
        wav = get_file(file_path, sample_rate, sample_length, num_chunks)
        wav = wav.to(device)

        chunks_logits = model(wav)

        logits = chunks_logits.mean(dim=0)

    probabilities = F.softmax(logits, dim=0).detach().numpy()

    print("Probabilities for each genre:")
    for index, prob in enumerate(probabilities):
        label = index_to_label[index]
        print(f"{label}: {prob:.2f} | ", end='')

    predicted_index = np.argmax(probabilities)
    predicted_label = index_to_label[predicted_index]

    print("\nPredicted label:", predicted_label)


def get_file(file_path, sample_rate, sample_length, num_chunks):
    """
    Load and preprocess the audio file.

    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Sampling rate of the audio file.
        sample_length (float): Length of each audio sample (in seconds).
        num_chunks (int): Number of chunks to divide the audio file into.

    Returns:
        torch.Tensor: Tensor containing the preprocessed audio data.
    """
    wav, _ = librosa.load(file_path, sr=sample_rate)

    if wav.ndim == 2:
        wav = (wav[:, 0] + wav[:, 1]) / 2

    num_samples = int(sample_rate * sample_length)
    wav = divide_chunks(wav, num_samples, num_chunks).astype('float32')

    return torch.tensor(wav)


def divide_chunks(wav, num_samples, num_chunks):
    """
    Divide the audio file into chunks for processing.

    Args:
        wav (np.ndarray): Array containing the audio data.
        num_samples (int): Number of samples per chunk.
        num_chunks (int): Number of chunks to divide the audio file into.

    Returns:
        np.ndarray: Array containing the divided audio chunks.
    """
    hop = (len(wav) - num_samples) // num_chunks

    wav = np.array([wav[i * hop:i * hop + num_samples] for i in range(num_chunks)])

    return wav
