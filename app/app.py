from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json
import requests
import nsvision as nv

CLASSES =['beige','black','blue','brown','cream','crimson','gold','green',
               'grey','navy-blue','orange','red','silver','titanium','white','yellow']
SIZE=300
# MODEL_URI='http://localhost:8501/v1/models/saved_model:predict'
# MODEL_URI='http://172.17.0.2:8501/v1/models/saved_model:predict'
MODEL_URI='http://tf_serving:8501/v1/models/saved_model:predict'



def rgb_to_xyz(rgb):

    if rgb.dtype == 'uint8':
        rgb = rgb.astype(float)
    # If not uint8, assume type is float64 or float32 

    # Inverse sRGB Gamma (convert to "linear RGB")
    lin_rgb = rgb / 12.92
    lin_rgb[rgb > 0.04045] = ((rgb[rgb > 0.04045] + 0.055) / 1.055) ** 2.4

    k = np.array([
                    [0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227],
                ], rgb.dtype)

    # Left multiply k by lin_rgb triplets. xyz[r, c] = k * lin_rgb[r, c] (use k.T and reverse order for using matmul).
    xyz = np.matmul(lin_rgb, k.T)

    return xyz

RATE = 0.5
def combine_rgb_xyz(rgb_image):
  xyz_image = rgb_to_xyz(rgb_image)
  return (rgb_image*RATE)+(xyz_image*(1-RATE))



def preprocess_img(path):
    img = nv.imread(path, resize = (SIZE,SIZE), color_mode='rgb',normalize=True)
    #img = image.load_img(path, target_size=(SIZE, SIZE))
    #img = image.img_to_array(img)
    img = combine_rgb_xyz(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    return img

app = Flask(__name__)

@app.route('/')
def entry_page():
    # Jinja template of the webpage
    return render_template('index.html')


@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    try:
        # Get image URL as input
        image_url = request.form['image_url']
        img_data = requests.get(image_url).content
        with open('img.jpg', 'wb') as handler:
            handler.write(img_data)
        img = preprocess_img('img.jpg')
        data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(MODEL_URI, data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']
        pred_class = CLASSES[np.argmax(predictions[0])]
        final = pred_class
        message = "Model prediction: {class_} ! ".format(class_ = str(pred_class))
        
        print('Python module executed successfully')
        print (str(pred_class))

    except Exception as e:
        # Store error to pass to the web page
        message = "Error encountered. Try another image. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(
            e.__class__, e.args, e.__doc__)
        final = pd.DataFrame({'A': ['Error'], 'B': [0]})

    # Return the model results to the web page
    return render_template('index.html',
                           message=message,
                           data=final,
                           image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True,host= '0.0.0.0', port=8080)
