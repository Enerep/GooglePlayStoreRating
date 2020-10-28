import pandas as pd
from sklearn.tree import DecisionTreeClassifier

google_rating = pd.read_csv("googleplaystore.csv")
google_rating_top = google_rating.head()

print(google_rating_top)

X = google_rating.drop(columns=['Rating'])
y = google_rating['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dtmodel = DecisionTreeClassifier()
dtmodel.fit(X_train, y_train)

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)
