import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier
import MLfunctions as fu


app = Flask(__name__)

tf = pickle.load(open('/Users/mohsen/Documents/GitHub/Flask_project/pac_tfidf.pk','rb'))
saved_clf = pickle.load(open('/Users/mohsen/Documents/GitHub/Flask_project/fake_news_PAC.sav','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    test = request.form['news']
    t = fu.preprocessor(test)
    t = fu.stemming2(t)
    tfidf_Newtest=tf.transform([t])
    pred = saved_clf.predict(tfidf_Newtest)
    if pred == 0:
       output = "FAKE"
    else:
       output = "TRUE"

    return render_template('index.html', prediction_text='This news  "{}"  is {}'.format(test,output))


if __name__ == "__main__":
    app.run(debug=True)