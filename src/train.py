import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
import time


def train_model(m, device, num_epochs, train_loader, valid_loader, model_path, index_to_label):
    """
    Train a neural network model and save the best model.

    Args:
        m (torch.nn.Module): The neural network model to be trained.
        device (torch.device): The device to run the training on (e.g., 'cuda' for GPU or 'cpu' for CPU).
        num_epochs (int): Number of epochs for training.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        valid_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        model_path (str): Path to save the trained model.
        index_to_label (dict): A dictionary mapping genre indices to genre labels.
    """
    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)

    valid_loss_list = []

    start_time = time.time()
    for epoch in range(num_epochs):
        train_losses = []
        valid_losses = []

        y_true = []
        y_pred = []

        m.train()
        for (wav, genre_index) in train_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            out = m(wav)
            loss = loss_function(out, genre_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        print(f"{time.time() - start_time:.1f}s Epoch: [{epoch + 1}/{num_epochs}], Train loss: {train_loss:.4f}")

        m.eval()
        for wav, genre_index in valid_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            b, c, t = wav.size()
            logits = m(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)

            loss = loss_function(logits, genre_index)
            valid_losses.append(loss.item())

            _, pred = torch.max(logits.data, 1)

            y_true.extend(genre_index.tolist())
            y_pred.extend(pred.tolist())

        accuracy = accuracy_score(y_true, y_pred)
        valid_loss = np.mean(valid_losses)

        print(f"{time.time() - start_time:.1f}s Epoch: [{epoch+1}/{num_epochs}], Valid loss: {valid_loss:.4f}, "
              f"Valid accuracy: {accuracy:.4f}")

        valid_loss_list.append(valid_loss)
        if np.argmin(valid_loss_list) == epoch:
            print(f"Saving the best model at {epoch + 1} epochs!")
            torch.save({
                'model_state_dict': m.state_dict(),
                'labels': index_to_label
            }, f'trained_models/{model_path}')
