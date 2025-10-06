from sklearn import svm

X = [[1, 0], [0, 1], [0, 0]]
Y = [0,1,0]
clf = svm.SVC()
clf.fit(X, Y)
Y1 = [[1, 0]]
prediciton = clf.predict(Y1)
print(prediciton)
print(clf.support_vectors_)
print(clf.support_)
print(clf.n_support_)