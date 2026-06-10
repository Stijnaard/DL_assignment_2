from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(labels, predicted_indices, show=False, save_path=None):
    cm = confusion_matrix(labels.cpu(), predicted_indices.cpu())
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.colormaps['Blues'])
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
