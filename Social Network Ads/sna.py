from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model (2).pkl','rb'))

@app.route('/')
def home():
    return render_template('social_network_ads.html')

@app.route('/predict', methods=['POST'])
def predict():
    #features=[int(x) for x in request.form.values()]
    #print(features)
    gender=int(request.values['gender'])
    Age=int(request.values['age'])
    salary=int(request.values['salary'])
    
    
    features=([gender,Age,salary])
    new_features=np.array(features).reshape(1,3)
    prediction=model.predict(new_features)
    prediction_values={0:"Not Purchased",1:"Purchased"}
    submit=prediction_values[prediction[0]]
    print(submit)
    return render_template('social_network_ads.html',prediction_text="You Have {}".format(submit))

if __name__=="__main__":
    app.run(port=8000)