import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from dataAnalysis.EDA import df_test, df_train, config_data

def inference(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    X_train  = df_train['libs']
    X_test = df_test['libs']

    # Преобразование текста в матрицу TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Load the trained model
    classifier = joblib.load('multinomial_nb_model.pkl')

    # Perform inference
    y_pred_test = classifier.predict(X_test_tfidf)
    with open(f'{config_data["results_path"]}/prediction.txt', 'w') as file:
        for item in y_pred_test:
            file.write('%s\n' % item)
    print('Model predictions on test set saved as .txt')

inference(df_train, df_test)