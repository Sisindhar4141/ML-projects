import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data2 ={
    "Temperature" : ['Hot','Hot','Mild','Cool','Cool','Cool','Mild','Hot','Cool','Mild','Mild','Mild','Hot','cool'],
    "Humidity" : ['Normal','High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal'],
    "Windy" : [False, True, False, False, False, True, True,False, False, False, True, True, False,True],
    "BuylceCream" : ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes','No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data2)
x = pd.get_dummies(df[['Temperature','Humidity','Windy']])
y = df['BuylceCream']
model = DecisionTreeClassifier(criterion="entropy",random_state=0)
model.fit(x,y)
while True:
    print("\n--Ice cream ")
    temp = input("Enter Temperature (Hot / Mild / Cool or 'exit' to stop): ")
    if temp.lower() == "exit":
        print(u"Exiting... Have a sweet day! 🍨")
        break
    humid = input("Enter Humidity (High / Normal):")
    windy_input = input("Is is Windy? (True / False):")
    windy = True if windy_input.lower() == "true" else False
    test_data = pd.DataFrame({
        "Temperature" : [temp.capitalize()],
        "Humidity" : [humid.capitalize()],
        "Windy" : [windy],
    })
    test_data_encoder = pd.get_dummies(test_data)
    test_data_encoder = test_data_encoder.reindex(columns=x.columns,fill_value=0)
    prediction = model.predict(test_data_encoder)
    print("Can I buy Icecream now?",prediction[0])

def prediction_icecream(temp,humid,windy):
    test_data =pd.DataFrame({
        "Temperature" : [temp.capitalize()],
         "Humidity" : [humid.capitalize()],
         "Windy" : [windy],
    })
    test_data_encoder = pd.get_dummies(test_data)
    test_data_encoder = test_data_encoder.reindex(columns=x.columns,fill_value=0)
    prediction = model.predict(test_data_encoder)[0]
    if prediction == "Yes":
        print("🍦 Yes! It's a good time to buy Ice Cream!")
    else:
        print("🚫 No, better skip ice cream now.")
