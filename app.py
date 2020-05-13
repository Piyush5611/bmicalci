import numpy as np
import pickle
from flask import Flask,request,jsonify ,render_template
import pandas as pd
 
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    

    
    int_features=[float(x) for x in request.form.values()]
    final_value=[np.array(int_features)]
  
    
    features_name=['height','weight']
 
     
    df = pd.DataFrame(final_value, columns=features_name)
    output = model.predict(df)
    
    if output == 0:
        res_val = "** extreme weak**"
    elif output==1:
        res_val = "Weak"
    elif output==2:
        res_val="Normal"
    elif output==3:
        res_val="Overweight"
    elif output==4:
        res_value="Obesity"
    else:
        res_val="Extreme Obesity"
        
        
        

    return render_template('index.html', prediction_text='You are {}'.format(res_val))
    
    




if __name__=='__main__':
    app.run()