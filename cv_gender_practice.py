import pandas as pd

df = pd.read_csv('combined_human_mouse_meta_v2.csv')

df = df.drop(columns = ['present', 'num_reads'])
print(df.head())

values_to_keep = ['male', 'female']
df = df[df['metadata_sex'].isin(values_to_keep)]
print(df.head())
#metadata_sex column
X = df.iloc[:,0:5]
#expression_sex column
y = df.iloc[:,6]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=17, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=17)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import classification_report

features = pd.DataFrame(rf.feature_importances_, index = X.columns)

#hyper parameters
rf2 = RandomForestClassifier(n_estimators = 1000,
                             criterion = 'entropy',
                             min_samples_split = 10,
                             max_depth = 14,
                             random_state = 42
)

rf2.fit(X_train, y_train)
#print(rf2.score(X_test, y_test))
ypred2 = rf2.predict(X_test)

#print(classification_report(y_test, ypred2))
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf2, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores for each fold:", cv_scores)