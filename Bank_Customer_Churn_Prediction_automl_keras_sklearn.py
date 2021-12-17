from h2o.automl import H2OAutoML
import h2o
from sklearn.metrics import accuracy_score
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from google.colab import drive

drive.mount("/content/drive")
df = pd.read_csv("/content/drive/kongkea/churn.csv")
df.head()
df.drop(["UserId", "Row", "LastName"], axis="columns", inplace=True)
df.head()
df.dtypes
df.head()
df.nunique()
labels = "Churned", "Retained"
sizes = [df.Churned[df["Churned"] == 1].count(
), df.Churned[df["Churned"] == 0].count()]
explode = 0, 0.1
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(
    sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90
)
ax1.axis("equal")
plt.title("Stats of churned and retained customers", size=20)
plt.show()

fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x="Geography", hue="Churned", data=df, ax=axarr[0][0])
sns.countplot(x="Gender", hue="Churned", data=df, ax=axarr[0][1])
sns.countplot(x="HasCrCard", hue="Churned", data=df, ax=axarr[1][0])
sns.countplot(x="IsActiveMember", hue="Churned", data=df, ax=axarr[1][1])
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y="CreditScore", x="Churned", hue="Churned", data=df, ax=axarr[0][0])
sns.boxplot(y="Age", x="Churned", hue="Churned", data=df, ax=axarr[0][1])
sns.boxplot(y="Tenure", x="Churned", hue="Churned", data=df, ax=axarr[1][0])
sns.boxplot(y="Balance", x="Churned", hue="Churned", data=df, ax=axarr[1][1])
sns.boxplot(y="NumOfProducts", x="Churned",
            hue="Churned", data=df, ax=axarr[2][0])
sns.boxplot(y="EstimatedSalary", x="Churned",
            hue="Churned", data=df, ax=axarr[2][1])
tenure_churn_no = df[df.Churned == 0].Tenure
tenure_churn_yes = df[df.Churned == 1].Tenure
plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
plt.hist(
    [tenure_churn_yes, tenure_churn_no],
    rwidth=0.95,
    color=["red", "green"],
    label=["Churn=Yes", "Churn=No"],
)
plt.legend()
df["BalanceSalaryRatio"] = df.Balance / df.EstimatedSalary
sns.boxplot(y="BalanceSalaryRatio", x="Churned", hue="Churned", data=df)
plt.ylim(-1, 5)
df["TenureByAge"] = df.Tenure / df.Age
sns.boxplot(y="TenureByAge", x="Churned", hue="Churned", data=df)
plt.ylim(-1, 1)
plt.show()


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == "object":
            print(f"{column}: {df[column].unique()}")


print_unique_col_values(df)
df["Gender"].replace({"Male": 1, "Female": 0}, inplace=True)
df1 = pd.get_dummies(data=df, columns=["Geography"])
df1.head()
scale_var = [
    "Tenure",
    "CreditScore",
    "Age",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
]

scaler = MinMaxScaler()
df1[scale_var] = scaler.fit_transform(df1[scale_var])
df1.head()
X = df1.drop("Churned", axis="columns")
y = df1["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)
len(X_train.columns)

model = keras.Sequential(
    [
        keras.layers.Dense(12, input_shape=(32, 14), activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100)
model.evaluate(X_test, y_test)
yp = model.predict(X_test)
yp
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_pred

print(classification_report(y_test, y_pred))

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Truth")

print("Accuracy score: ", accuracy_score(y_test, y_pred) * 100, "%")

h2o.init(max_mem_size="16G")
df = h2o.import_file("/content/drive/kongkea/Churn_Modelling.csv")
df.head()
df_train, df_test = df.split_frame(ratios=[0.8])
df_train
y = "Churned"
x = df.columns
x.remove(y)
x.remove("UserId")
x.remove("Row")
x.remove("LastName")
aml = H2OAutoML(
    max_runtime_secs=300, max_models=10, seed=10, verbosity="info", nfolds=2
)
aml.train(x=x, y=y, training_frame=df_train)
lb = aml.leaderboard
lb
model_ids = list(aml.leaderboard["model_id"].as_data_frame().iloc[:, 0])
model_ids
aml.leader.model_performance(df_test)
h2o.get_model([mid for mid in model_ids if "StackedEnsemble" in mid][0])
output = h2o.get_model(
    [mid for mid in model_ids if "StackedEnsemble" in mid][0])
output.params
aml.leader
y_pred = aml.leader.predict(df_test)
y_pred
