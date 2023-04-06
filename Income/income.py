from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('income_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('income.html')

@app.route('/predict', methods=['POST'])
def predict():
    age=request.values['age']
    print(age)
    age=np.reshape(age,(-1,1))
    output=model.predict(age)
    output=output.item()
    print(output)
    return render_template('income.html',prediction=output)

if __name__=="__main__":
    app.run()

