import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from dataAnalysis.EDA import df_train, df_val, config_data
from utils import calculate_metrics
from sklearn.utils.class_weight import compute_class_weight

def trainModel(df_train: pd.DataFrame, df_val: pd.DataFrame, balanced: bool = False) -> None:
    """
    Обучение модели классификации с использованием Multinomial Naive Bayes на 
    предоставленных тренировочных данных и оценка ее на валидационных данных. 
    Производится сохранение весов модели и различных метрик оценки в файлы.
    
    Параметры:
        df_train (pd.DataFrame): Тренировочный набор данных, содержащий столбцы 'libs' и 'is_virus'.
        df_val (pd.DataFrame): Валидационный набор данных, содержащий столбцы 'libs' и 'is_virus'.
        
    Возвращает:
        None
    """
    # Обучающая выборка
    X_train, y_train  = df_train['libs'], df_train['is_virus']
    # Валидационная выборка
    X_val, y_val = df_val['libs'], df_val['is_virus']

    # Преобразование текста в матрицу TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)

    # Обучение модели классификации: Multinomial Naive Bayes классификатор 
    if balanced == True:
        # учитываем дизбаланс классов
        train_labels = list(y_train)
        class_weights = compute_class_weight('balanced', 
                                             classes=np.unique(train_labels), 
                                             y=train_labels)
        class_weights = {0: class_weights[0],
                        1: class_weights[1]}
        classifier = MultinomialNB(class_prior=list(class_weights.values()))
    else:
        # не учитываем
        classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)

    # Предсказание категорий на validation set
    y_pred = classifier.predict(X_val_tfidf)
    # calculate metrics
    metrics = calculate_metrics(y_val, y_pred)

    # Запишем метрики на валидации в файл validation.txt
    with open(f'{config_data["results_path"]}/validation.txt', 'w') as file:
        file.write(f'''True positive: {metrics['tp']}
False positive: {metrics['fp']}
False negative: {metrics['fn']}
True negative: {metrics['tn']}
Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1: {metrics['f1']:.4f}
''')
    print('Metrics saved as .txt')

    # Сохраним веса модели
    joblib.dump(classifier, 'multinomial_nb_model.pkl')
    print('Model saved as .pkl')

trainModel(df_train, df_val, balanced=True)