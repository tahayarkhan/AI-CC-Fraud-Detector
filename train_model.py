import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv("data/creditcard.csv")


x = df.drop("Class", axis=1)
y = df["Class"]


x["Amount"] = StandardScaler().fit_transform(x[["Amount"]])


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

joblib.dump(clf, "models/model.pkl")


y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
