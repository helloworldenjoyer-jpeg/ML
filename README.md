# ML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier
from matplotlib import pyplot as plt
import seaborn as sns

#Upload data set
!wget https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv

##Data set
df = pd.read_csv('fake_news.csv')
X = df['text']
y = df['label']
#Learning and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=7)
#Vectorization text
vect = TfidfVectorizer()
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)
#Passive aggressive classifier
aggr = PassiveAggressiveClassifier(max_iter=50)
aggr.fit(X_train, y_train)
#Logreg model
model = LogisticRegression()
model.fit(X_train, y_train)
#Prediction and score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc}')
#Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])
#Matrix visualization
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels=['Predicted fake', 'Predicted real'], yticklabels=['Actual fake', 'Actual real'])
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
