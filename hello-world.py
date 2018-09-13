from sklearn import tree

# 0 for apple, and 1 for orange
features = [[140,0],[150,0],[170,1],[180,1],[175,1],[160,1],[130,0]]

#lables = [1,1,0,0,0,0,1]
lables = ['orange','orange','apple','apple','apple','apple','orange']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, lables)

print(clf.predict([[150, 0]]))