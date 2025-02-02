import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def load_data():
    try:
        logging.info("Loading data from xdata.csv")
        df = pd.read_csv('xdata.csv')
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    try:
        logging.info("Starting preprocessing")
        df = pd.get_dummies(df, columns=['Surname', 'Geography', 'Gender'], drop_first=True)
        return df
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def split_data(df):
    try:
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error(f"Error training the model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        logging.info(f"Model Evaluation - Accuracy: {accuracy}, ROC AUC Score: {roc_auc}")
        logging.info(f"Confusion Matrix: \n{conf_matrix}")
    except Exception as e:
        logging.error(f"Error evaluating the model: {e}")
        raise

def main():
    try:
        logging.info("Starting the main function")
        df = load_data()
        preprocessed_df = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(preprocessed_df)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        logging.info("Main function completed successfully")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()