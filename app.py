from flask import Flask, render_template ,request
import pickle
import re

app = Flask(__name__)

#load Pickle
feature =pickle.load(open('feature.pkl','rb'))
model =pickle.load(open('model_SVM.pkl','rb'))


@app.route('/')
def home():
    return render_template('projectmcai.html')

@app.route('/predict', methods= ['POST'])
def predict():
  
  text = request.form['fname']
  #mengubah huruf pada kalimat 
  txt = text.lower()
  txt = re.sub("[^a-zA-Z]", ' ', txt)

  #ekstrak = feature.tranform(txt)

  if text != '':
    prediksi = model.predict(feature.transform([txt]).toarray())
    #Indikasi result 
    if prediksi ==0 :
      res_pred = "no racism"
    else :
      res_pred = "racism"
  else:
    res_pred= '-'

  return render_template('projectmcai.html', hasil=res_pred)
  

if __name__ == "__main__":
  app.run()
