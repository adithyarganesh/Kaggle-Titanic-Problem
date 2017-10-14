# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


titanic_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

titanic_df.head()
# In[]

titanic_df.info()
test_df.info()

# In[]

titanic_df = titanic_df.drop(["PassengerId", "Name", "Ticket"] , axis = 1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)

# In[ ]:

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanic_df.info()


# In[ ]:
sns.factorplot(x='Embarked',y='Survived', data=titanic_df)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

#sns.countplot(x= 'Embarked' , data = titanic_df , ax=axis1)
sns.countplot(x='Embarked', data=titanic_df, ax=axis1) 
sns.countplot(hue='Embarked',x = "Survived", data=titanic_df, ax=axis2)

embark_perc = titanic_df.groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc ,ax=axis3)


# In[]
split_embarked = pd.get_dummies(titanic_df["Embarked"])
split_embarked_test = pd.get_dummies(test_df["Embarked"])
split_embarked.info()
split_embarked_test.info()

# In[]


split_embarked.drop( ['S'] , axis = 1, inplace= True)
split_embarked_test.drop( ['S'] , axis = 1, inplace= True)
split_embarked.info()
split_embarked_test.info()
#split_embarked.to_csv('omg.csv', index=False)

# In[]

titanic_df = titanic_df.join(split_embarked)
titanic_df.info()
test_df = test_df.join(split_embarked_test)
test_df.info()
# In[]
titanic_df.drop(["Embarked"] , axis = 1, inplace = True)
titanic_df.info()
#titanic_df.to_csv('omxog.csv', index=False)
test_df.drop(["Embarked"] , axis = 1, inplace = True)
test_df.info()
#embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
#embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
#titanic_df = titanic_df.join(embark_dummies_titanic)

#titanic_df.drop(['Embarked'], axis=1,inplace=True)

#titanic_df.to_csv('lmao.csv', index=False)
# In[]

test_df["Fare"].fillna(test_df["Fare"].median() , inplace = True)
test_df.info()









