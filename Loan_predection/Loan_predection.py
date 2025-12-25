import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from scipy.sparse import hstack,csr_matrix
data = pd.read_csv('loan_data.csv')
df = pd.DataFrame(data)
le = LabelEncoder()
vectorizer = TfidfVectorizer(stop_words='english')
text = vectorizer.fit_transform(df['Text'])
employ = le.fit_transform(df['Employment_Status'])
employ = csr_matrix(employ.reshape(-1,1))
scaler = StandardScaler(with_mean=False)
numaric = scaler.fit_transform(df[['Income','Credit_Score','Loan_Amount','DTI_Ratio']])
x = hstack([text,csr_matrix(numaric),csr_matrix(employ)])
y = df['Approval']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = RandomForestClassifier(n_estimators=100,random_state=42,criterion='gini')
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
for actual_data,predicted_data in zip(y_test,y_pred):
    print(f'Actual_data : {actual_data} --------> Prideict_data : {predicted_data}')
print("Accuracy is : ",(accuracy_score(y_test,y_pred)))
print("F1_score is : ",(f1_score(y_test,y_pred ,average='weighted')))
