import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#read training data into a dataframe
df1 = pd.read_csv('train_data.csv')
test1 = pd.read_csv('test.csv')
#edit and analyze data frame

#create a copy of train data set
df = df1.copy()

#remove irrelevant copies
del df['PassengerId']
del df['Name']
del df['Ticket']
del df['Cabin']

#remove target 
del df['Survived']

#create binary variables from qualitative variables
df['Sex'] = df['Sex'] == 'male'
df['Embarked_S'] = df['Embarked'] == 'S'
df['Embarked_Q'] = df['Embarked'] == 'Q'
df['Embarked_C'] = df['Embarked'] == 'C'
df['Child'] = df['Age'] <= 12
del df['Embarked']
del df['Fare']

df_cat = df["Pclass"]
encoder = LabelEncoder()
df_encoded = encoder.fit_transform(df_cat)
encoder = OneHotEncoder(categories='auto')
df_cat = encoder.fit_transform(df_encoded.reshape(-1,1))
df_cat = df_cat.toarray()
df_cat = pd.DataFrame(df_cat)

df = df.join(df_cat)

del df['Pclass']
del df['Age']

#repeat steps for test data set
test = test1.copy()

del test['PassengerId']
del test['Name']
del test['Ticket']
del test['Cabin']


test['Sex'] = test['Sex'] == 'male'
test['Embarked_S'] = test['Embarked'] == 'S'
test['Embarked_Q'] = test['Embarked'] == 'Q'
test['Embarked_C'] = test['Embarked'] == 'C'
test['Child'] = test['Age'] <= 12
del test['Embarked']
del test['Fare']

test_cat = test["Pclass"]
encoder = LabelEncoder()
test_encoded = encoder.fit_transform(test_cat)
encoder = OneHotEncoder(categories='auto')
test_cat = encoder.fit_transform(test_encoded.reshape(-1,1))
test_cat = test_cat.toarray()
test_cat = pd.DataFrame(test_cat)

test = test.join(test_cat)

del test['Pclass']
del test['Age']

#train using decision tree
classifier = DecisionTreeClassifier(random_state=0)
train = classifier.fit(df, df1['Survived'])
training = train.predict(df)

#compute error on test set
error = abs(df1['Survived'] - training).sum()
predictions = train.predict(test)
predictions = pd.Series(predictions)
predictions = pd.DataFrame({'PassengerId' : range(892,  892 + 418 ,1), 'Survived': predictions})

#record predictions
predictions.to_csv('predictions1.csv', index = False)