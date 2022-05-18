import pandas as pd
import plotly_express as pe
import plotly.graph_objects as pgo
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

# ------------------------------------------------

data = pd.read_csv("project.csv")

TOEFL_Score = data["TOEFL Score"].tolist()
GRE_Score = data["GRE Score"].tolist()
Chance_of_admit = data["Chance of admit"].tolist()

graph = pe.scatter(x = TOEFL_Score , y = GRE_Score)


# ------------------------------------------

colors = []

for i in GRE_Score :
    if i == 1 :
        colors.append("green")
    else :
        colors.append("red")

fig = pgo.Figure(data = pgo.Scatter(
    x = TOEFL_Score ,
    y = Chance_of_admit , 
    mode = "markers" ,
    marker=dict(color=colors)
))

fig.show()

# --------------------------------------------------------

factors = data[['TOEFL Score' , 'Chance of admit']]

GRE_Score = data['GRE Score']

TOEFL_Score_train , TOEFL_Score_test , GRE_Score_train , GRE_Score_test =  train_test_split(factors , GRE_Score , test_size = 0.25)

# -----------------------------------------------------

sc = StandardScaler()

TOEFL_Score_train = sc.fit_transform(TOEFL_Score_train)

TOEFL_Score_test = sc.fit_transform(TOEFL_Score_test)

# ---------------------------------------------------------------

anything = LogisticRegression(random_state = 0)

anything.fit(TOEFL_Score_train , GRE_Score_train)

GRE_Score_predict = anything.predict(TOEFL_Score_test)


print("accuracy : " , accuracy_score(GRE_Score_test ,GRE_Score_predict))

print("----------------------------------------------")










































































            






