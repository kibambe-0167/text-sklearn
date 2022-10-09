# import data
from cgitb import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np


categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)

# for d in twenty_train.data[: 10]: print( d )
# print(  twenty_train.data[0].split("\n")[: 3] )
# see target col
# print( twenty_train.target )
print( ' Target Names : %s' % twenty_train.target_names )


# TOKENIZING TEXT WITH SKLEARN
count_vect = CountVectorizer()
x_train_count = count_vect.fit_transform( twenty_train.data )
# print( x_train_count.shape )

# from occurences to freg.
# tf_transfomer = TfidfTransformer(use_idf=False).fit( x_train_count )
# x_train_tf = tf_transfomer.transform( x_train_count )
# print( x_train_tf.shape )
# a shorter way is
tfidt_transformer = TfidfTransformer()
x_train_tfidf = tfidt_transformer.fit_transform( x_train_count )
# print( x_train_tfidf.shape )



# TRAINING A CLASSIFIER.
clf = MultinomialNB().fit( x_train_tfidf, twenty_train.target )

# new doc.
docs_new = ['God is love', 'OpenGL on the GPU is fast']
print( docs_new )

# extract features using transform, instead of fit_transform, bc the data have already be fit to the training set.
x_new_docs = count_vect.transform( docs_new )
x_new_tfidf = tfidt_transformer.transform( x_new_docs )

# predict
predict = clf.predict( x_new_tfidf )
print( 'Prediction Classes: %s' %  predict )

for doc, cat in zip( docs_new, predict ):
    print( '%r = %s' % ( doc, twenty_train.target_names[cat] ) )
    
    
    
    

# building a pipeline
print("\n\n using pipelines")
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer() ),
    ('clf', MultinomialNB() )
])

# train the model.
text_clf.fit( twenty_train.data, twenty_train.target );

# predict
print( text_clf.predict( docs_new ) )




# EVAULATING PERFORMANCE OF THE SET.
print("\n\nEval performance of test set.")
twenty_test = fetch_20newsgroups( subset='test', random_state=42, categories=categories, shuffle=True )
# use the pipeline clf
predicttions = text_clf.predict( twenty_test.data )
score =  np.mean( predicttions == twenty_test.target )
print( f"Prediction : {score:.2f}" )




# USING LINEAR SVM CLASSIFIER.
# use pipelines
print("\n\nUsing Linear SVM")
text_clf1 = Pipeline([
    ('vect', CountVectorizer() ),
    ('tfidf', TfidfTransformer() ),
    ('clf',  SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))
])

text_clf1.fit( twenty_train.data, twenty_train.target )
predicted = text_clf1.predict( twenty_test.data  )
score =  np.mean( predicted == twenty_test.target )
print( f"Prediction : {score:.2f}" )





# METRICS
print("\n\n METRICS")
print( metrics.classification_report( twenty_test.target, predicted, target_names=twenty_train.target_names ) )
print("\n\n Confusion METRICS")
print( metrics.confusion_matrix( twenty_test.target, predicted ) )



# USING GRID SEARCH 
print("\n\nGRID SEARCH, SEARCH FOR BEST PARAMS.")
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}
# n_jobs, detect how many cores are installed and use them all.
gs_clf = GridSearchCV( text_clf1, parameters, cv=5, n_jobs=1 )

gs_clf = gs_clf.fit( twenty_train.data, twenty_train.target )

print( twenty_train.target_names[gs_clf.predict(['God is love'])[0]] )

# print( gs_clf.cv_results_ )
print( f'Best Score : {gs_clf.best_score_:.2f}' )

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))