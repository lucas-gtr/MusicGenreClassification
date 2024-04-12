import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


def eval_model(m, device, test_loader, index_to_label):
    """
    Evaluate the performance of a deep learning model on a test dataset
    by computing its accuracy and displaying the confusion matrix.

    Args:
        m (torch.nn.Module): The trained deep learning model.
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' for GPU or 'cpu' for CPU).
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        index_to_label (dict): A dictionary mapping genre indices to genre labels.
    """
    m.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for wav, genre_index in test_loader:
            wav = wav.to(device)
            genre_index = genre_index.to(device)

            b, c, t = wav.size()
            logits = m(wav.view(-1, t))
            logits = logits.view(b, c, -1).mean(dim=1)

            _, pred = torch.max(logits.data, 1)

            y_true.extend(genre_index.tolist())
            y_pred.extend(pred.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    genres_list = []
    for i in range(len(index_to_label)):
        genres_list.append(index_to_label[i])

    sns.heatmap(cm, annot=True, xticklabels=genres_list, yticklabels=genres_list, cmap='YlGnBu')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted genres')
    plt.ylabel('Trues genres')

    print('Accuracy: %.3f' % accuracy)
    plt.show()
