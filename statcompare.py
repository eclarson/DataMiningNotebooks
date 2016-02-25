clf1 = Pipeline(
    [('PCA',RandomizedPCA(n_components=100)),
     ('CLF',GaussianNB())]
)
clf2 = Pipeline(
    [('PCA',RandomizedPCA(n_components=500)),
     ('CLF',GaussianNB())]
)


from sklearn.cross_validation import cross_val_score
# is clf1 better or worse than clf2?
cv=StratifiedKFold(y,n_folds=10)
acc1 = cross_val_score(clf1, X, y=y, cv=cv)
acc2 = cross_val_score(clf2, X, y=y, cv=cv)

#=================================

t = 2.26 / np.sqrt(10)

e = (1-acc1)-(1-acc2)
# std1 = np.std(acc1)
# std2 = np.std(acc2)
stdtot = np.std(e)

dbar = np.mean(e)
print 'Range of:', dbar-t*stdtot,dbar+t*stdtot 
print np.mean(acc1), np.mean(acc2)


#===============================
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.imshow(cm_normalized,cmap=plt.get_cmap('Reds'),aspect='auto')