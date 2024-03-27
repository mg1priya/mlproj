from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

@app.route('/index',methods = ['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predictdata():
    print(request.method)
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
           ID=int(request.form.get('ID')),
        LIMIT_BAL=int(request.form.get('LIMIT_BAL')),
        SEX=int(request.form.get('SEX')),
        EDUCATION=int(request.form.get('EDUCATION')),
        MARRIAGE=int(request.form.get('MARRIAGE')),
        AGE=int(request.form.get('AGE')),
        PAY_0=int(request.form.get('PAY_0')),
        PAY_2=int(request.form.get('PAY_2')),
        PAY_3=int(request.form.get('PAY_3')),
        PAY_4=int(request.form.get('PAY_4')),
        PAY_5=int(request.form.get('PAY_5')),
        PAY_6=int(request.form.get('PAY_5')),
        BILL_AMT1=int(request.form.get('BILL_AMT1')),
        BILL_AMT2=int(request.form.get('BILL_AMT2')),
        BILL_AMT3=int(request.form.get('BILL_AMT3')),
        BILL_AMT4=int(request.form.get('BILL_AMT4')),
        BILL_AMT5=int(request.form.get('BILL_AMT5')),
        BILL_AMT6=int(request.form.get('BILL_AMT6')),
        PAY_AMT1=int(request.form.get('PAY_AMT1')),
        PAY_AMT2=int(request.form.get('PAY_AMT2')),
        PAY_AMT3=int(request.form.get('PAY_AMT3')),
        PAY_AMT4=int(request.form.get('PAY_AMT4')),
        PAY_AMT5=int(request.form.get('PAY_AMT5')),
        PAY_AMT6=int(request.form.get('PAY_AMT6'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.debug=True
    app.run()   

