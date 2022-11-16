import os
import numpy as np  # used for numerical analysis
from flask import Flask, request, render_template

from tensorflow.keras.models import load_model  
from tensorflow.keras.preprocessing import image

app = Flask(__name__)  
model = load_model('ECG.h5')  

@app.route("/") #default route
@app.route("/home") #Home page set to default page
def default():
    return render_template('index.html') #rendering index.html

@app.route("/info") #route to info page
def information():
    return render_template("info.html") #rendering info.html

@app.route("/about") #route to about us page
def about_us():
    return render_template('about.html')  #rendering about.html

@app.route("/contact") #route to contact us page
def contact_us():
    return render_template('contact.html')  #rendering contact.html

@app.route("/upload") #default route
def test():
    return render_template("predict.html")  #rendering contact.html
app.config['UPLOAD_FOLDER']="static/testing"
@app.route("/",methods=['GET','POST'])
def upload():
    if request.method=='POST':
        upload_image=request.files['upload_image']

        if upload_image.filename!='':
            filepath=os.path.join(app.config["UPLOAD_FOLDER"],upload_image.filename)
            upload_image.save(filepath)
            path=filepath
            return render_template('predict.html',data=path)
            flash("File Upload Successfully","success")
    
        img = image.load_img(path, target_size=(64, 64))
        print(img) # load and reshaping the image
        x = image.img_to_array(img)  # converting image to array
        x = np.expand_dims(x, axis=0)  # changing the dimensions of the image

        preds = model.predict(x)  # predicting classes
        pred = np.argmax(preds, axis=1)  # predicting classes
        print("prediction", pred)  # printing the prediction

        index = ['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction',
                 'Premature Ventricular Contractions', 'Right Bundle Branch Block', 'Ventricular Fibrillation']
        result = str(index[pred[0]])
        return result  # resturing the result
    return None



if __name__ == "__main__":
    app.run(debug=True)  
