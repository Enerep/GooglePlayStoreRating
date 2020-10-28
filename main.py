import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

google_rating = pd.read_csv("googleplaystore.csv")
google_rating_top = google_rating.head()

# Printing the top 5 lines of the csv file
print(google_rating_top)

X = google_rating.drop(columns=['App', 'Category', 'Rating', "Last Updated",
								"Android Ver", "Current Ver", "Genres"])
y = google_rating['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

dtmodel = DecisionTreeClassifier()
dtmodel.fit(X_train, y_train)

predictions = dtmodel.predict(X_test)

# Outputting the score
score = accuracy_score(y_test, predictions)
print(score)


