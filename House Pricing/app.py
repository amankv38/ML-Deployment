from flask import Flask,render_template,request
import pickle
import numpy as np


app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('House Pricing.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    area=float(request.values['housearea'])
    print(area)
    area=np.reshape(area,(-1,1))
    output=model.predict(area)
    output=output.item()
    output=round(output,2)
    print(output)
    return render_template('House Pricing.html',prediction=output)
    
   
if __name__ == '__main__':
    app.run()