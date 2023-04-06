from flask import Flask,render_template,request
import numpy as np
import pickle

app=Flask(__name__)
model=pickle.load(open('car1_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('car_price.html')

@app.route('/predict', methods=['POST'])
def predict():
    mileage=int(request.values['mileage_input'])
    years=int(request.values['years_input'])
    print(mileage)
    print(years)
    car=([mileage,years])
    car=np.reshape(car,(1,2))
    prediction=model.predict(car)
    # prediction=prediction_item()
    # prediction=round(prediction,2)

    return render_template('car_price.html',prediction="CAR VALUE IN THE RANGE OF {}".format(prediction))

if __name__=="__main__":
    app.run(port=8000)




   
    
   
