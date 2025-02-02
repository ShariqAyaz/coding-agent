import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
df = pd.read_csv('xdata.csv')
df['Age'] = df['Age'].astype('float64')  # Convert age to float for regression
df['Tenure'] = df['Tenure'].astype('int64')  # Convert tenure to integer
# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)  # Drop first dummy variable for encoded columns
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = LogisticRegression()
try:
    logging.info('Training model')
    model.fit(X_train, y_train)
except Exception as e:
    logging.error('Error training model', exc_info=True)
# Evaluate model
logging.info('Evaluating model')
y_pred = model.predict(X_test)
accuracy = round(model.score(X_test, y_test), 2)
logging.info(f'Accuracy: {accuracy}')
# Plot confusion matrix
from sklearn.metrics import confusion_matrix
logging.info('Plotting confusion matrix')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()
logging.info('Program completed successfully')