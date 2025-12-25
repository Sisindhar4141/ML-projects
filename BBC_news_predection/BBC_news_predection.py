import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score ,f1_score,classification_report
data = pd.read_csv('bbc_text_cls new.csv')
df = pd.DataFrame(data)
print(df.columns)
vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(df['text'])
y = df['labels']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = SVC(kernel='rbf')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))
data1 = pd.DataFrame({'Actual_values' : y_test.values,'Predicted_values' : y_pred})
new_headlines = [input("Enter the head lines of news : ")]
type = vectorizer.transform(new_headlines)
prediction = model.predict(type)
for headlines, type in zip(new_headlines,prediction):
    print(f"Headlines : {headlines} ---> type :{type}")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print("F1_score is : "(f1_score(y_test, y_pred, average='weighted')))
