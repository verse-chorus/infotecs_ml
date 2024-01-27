import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataAnalysis.EDA import config_data

def create_heatmap(conf_matrix):
    """
    Создание heatmap'ы по заданной confusion matrix

    Parameters:
        conf_matrix (array-like): Визуализируемая confusion matrix

    Returns:
        None
    """
    sns.heatmap(conf_matrix, annot=True, cmap='viridis', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.xticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'])
    # Save the heatmap as a PNG file
    plt.savefig(f'{config_data["results_path"]}/confusion_matrix.png')

def calculate_metrics(y, y_pred):
    """
    Подсчет метрик: precision, recall, accuracy, F1,
    а также подсчет TP, FP, FN, TN по confusion matrix
    
    Args:
    - y: истинное значение 
    - y_pred: предсказанное моделью значение
    
    Returns:
    - словарь с метриками
    """
    # Вычислим Матрицу ошибок (Confusion matrix)
    conf_matrix = confusion_matrix(y, y_pred)
    # Создадим heatmap'у и сохраним ее в results/confusion_matrix.png
    create_heatmap(conf_matrix)
    print('Confusion matrix saved as .png')

    tp, fp, fn, tn = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[0, 0]
    # Подсчет метрик: precision, recall, accuracy, F1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # вернуть словарь с метриками
    return {'tp': tp, 
            'fp': fp, 
            'fn': fn, 
            'tn': tn, 
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1}
    