import pandas as pandas
from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('fake_job_knn_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('fake_job.html')

@app.route('/predict', methods=['POST'])
def predict():
    title=int(request.values['title'])
    location=int(request.values['location'])
    department=int(request.values['department'])
    salary_range=int(request.values['salary_range'])
    company_profile=int(request.values['company_profile'])
    description=int(request.values['description'])
    requirements=int(request.values['requirements'])
    benifits=int(request.values['benifits'])
    telecommuting=int(request.values['telecommuting'])
    company_logo=int(request.values['company_logo'])
    questions=int(request.values['questions'])
    employement=int(request.values['employement'])
    required_experince=int(request.values['required_experince'])
    required_education=int(request.values['required_education'])
    industry=int(request.values['industry'])
    function=int(request.values['function'])

    job=([title,location,department,salary_range,company_profile,description,requirements,benifits,telecommuting,company_logo,questions,employement,required_experince,required_education,industry,function])
    print(job)
    fake_job=np.reshape(job,(1,16))
    prediction=model.predict(fake_job)
    prediction_values={0:"GENUINE",1:"FAKE"}
    result=prediction_values[prediction[0]]
    return render_template('fake_job.html',prediction="THIS JOB POSTING IS {}.".format(result))


if __name__=="__main__":
    app.run(port=8000)