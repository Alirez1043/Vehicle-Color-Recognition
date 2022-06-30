import tensorflow_io as tfio
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from keras.layers import Rescaling
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style('darkgrid')
from IPython.core.display import display, HTML


class DatasetBuilder():
  def __init__(self ,dataset_path,image_size=(224,224),batch_size=32):
    self.datasetPath = dataset_path
    self.image_size = image_size
    self.batch_size = batch_size

  def create(self ,type_):
      ds = tf.keras.utils.image_dataset_from_directory(
          (os.path.join(self.datasetPath,type_)) ,
          shuffle=True,
          label_mode = 'categorical',
          batch_size = self.batch_size,    
          image_size = self.image_size  )
      return ds

  def get_class_weights(self,ds):
    class_series = np.array([])
    for i , class_ in enumerate(ds.class_names) :
        shape = len(os.listdir(os.path.join(self.datasetPath,'train',class_)))
        class_series = np.concatenate((class_series, np.full((shape,) ,i)), axis=None)
    class_labels = np.unique(class_series).astype(int)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    return dict(zip(class_labels, class_weights))

  @staticmethod
  def rgb2ycbcr(image):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = image.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    ycbcr = np.uint8(ycbcr)
    return tf.convert_to_tensor(ycbcr, dtype=tf.uint8)
  
  @staticmethod
  def rgb2hsv(image):
    return tf.image.rgb_to_hsv(image)

  @staticmethod
  def rgb2xyz(image):
    return tfio.experimental.color.rgb_to_xyz(image)

#---------------------------

class EvaluateModel():
  def __init__(self ,model ,test_ds,class_names ,model_name):

    self.model = model
    self.test_ds = test_ds
    self.class_names = class_names
    self.model_name = model_name
    self.save_dir = os.path.join(os.getcwd(),'drive',"MyDrive", 'Car-Color-Recognition' ,'Logs', model_name)
    os.makedirs( os.path.join(os.getcwd(),'drive',"MyDrive", 'Car-Color-Recognition' ,'Logs', model_name)  ,exist_ok=True)

  @staticmethod
  def get_labels_and_predictions(model ,ds ,class_names):
    labels = []
    preds  = []
    for item in ds.take(-1) :
      image_batch = item[0]
      label_batch = item[1]
      y_pred= model.predict(image_batch)
      for y in np.array(y_pred) :
          preds.append(y.argmax())

      for lbl in np.array(label_batch) :
          labels.append(lbl.argmax())

    return np.array(labels) ,np.array(preds)

  def report(self,):
    labels ,preds = EvaluateModel.get_labels_and_predictions(self.model ,self.test_ds,self.class_names)
    result_dic = classification_report( labels, preds, target_names=self.class_names ,output_dict=True )
    df = pd.DataFrame.from_dict(result_dic).T
    df.to_csv(os.path.join(self.save_dir+'/report.csv'))
    print("Report CSV file added to logs .")
    return df

    return result_dic
  def plot_classes_error(self,):
    classes=self.class_names
    labels ,preds= EvaluateModel.get_labels_and_predictions(self.model ,self.test_ds ,self.class_names)
    true_class = np.array(classes)[labels]
    pred_class = np.array(classes)[preds]
    error_indices = labels[(labels!=preds)].tolist()
    errors = len(error_indices)
    class_dict = { key:value for key ,value in enumerate(classes) }
    if errors>0:
        plot_bar=[]
        plot_class=[]
        for  key, value in class_dict.items():        
            count=error_indices.count(key) 
            if count!=0:
                plot_bar.append(count) # list containg how many times a class c had an error
                plot_class.append(value)   # stores the class 
        fig=plt.figure()
        fig.set_figheight(len(plot_class)/3)
        fig.set_figwidth(10)
        plt.style.use('fivethirtyeight')
        for i in range(0, len(plot_class)):
            c=plot_class[i]
            x=plot_bar[i]
            plt.barh(c, x, )
            plt.title( ' Errors by Class on Validation Set')
        dest = os.path.join(self.save_dir,'error_list.png')
        plt.savefig( dest )
        plt.show()

  def get_confusion(self,):
    labels ,preds= EvaluateModel.get_labels_and_predictions(self.model ,self.test_ds ,self.class_names)
    return confusion_matrix( labels, preds ,normalize = 'true')   

  def plot_confusion(self,):
    cm = self.get_confusion()*100       
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='.1f', xticklabels=self.class_names, yticklabels=self.class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    dest = os.path.join(self.save_dir+'/confusion_matrix.png')
    plt.savefig( dest )
    plt.show(block=False)



class ModelUtils():

    def __init__(self,model_name ,model_type):
      self.model_name = model_name
      self.model_type = model_type
      self.save_dir = os.path.join(os.getcwd(),'drive',"MyDrive", 'Car-Color-Recognition')
      os.makedirs( self.save_dir  ,exist_ok=True)

    def get_callbacks(self , early_stop=True ,tensorboard=True ,checkpoint=True ,lr_scheduler=True,run_index=1):
      callbacks = []

      if early_stop:

        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4 ,restore_best_weights=True)
        callbacks.append(early_stop_cb)

      if tensorboard:

        dest = os.path.join(self.save_dir,"Callbacks","Tensorboards",self.model_name)
        os.makedirs(dest ,exist_ok=True)       
        run_logdir = os.path.join( dest ,f"run_{run_index:03d}" )
        tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
        callbacks.append(tensorboard_cb)

      if lr_scheduler:

        lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1)
        callbacks.append(lr_scheduler_cb)

      if checkpoint:

        checkpoint_base = os.path.join(self.save_dir,"Callbacks","Checkpoints",self.model_name)
        os.makedirs(checkpoint_base ,exist_ok = True)
        postFix = "_{epoch}"
        if self.model_type=='HUB' :
          postFix = "_{epoch}.h5"
        checkpoint_dir = os.path.join(checkpoint_base ,postFix)
        check_point=keras.callbacks.ModelCheckpoint( filepath= checkpoint_dir, save_best_only=True, monitor="val_loss", verbose=1 )
        callbacks.append(check_point)

      return  callbacks

    @staticmethod
    def get_compile_model(model ,optimizer_type ,lr = 1e-2):
      if optimizer_type=='Adam' :
          lr_schedule =tf.keras.optimizers.schedules.ExponentialDecay(lr, 602, .9)
          optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
      else:
          optimizer = keras.optimizers.SGD(learning_rate=lr,momentum=.9)
      loss_fn = tf.keras.losses.CategoricalCrossentropy()
      model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
      return model

    def make_or_restore_model(self,model,optimizer_type="SGD",lr = 1e-2) :
        checkpoint_dir = os.path.join(self.save_dir,"Callbacks","Checkpoints",self.model_name)
        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)
          return ModelUtils.get_compile_model(model,optimizer_type,lr)
        else:
          checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
          if checkpoints:
              latest_checkpoint = max(checkpoints, key=os.path.getctime)
              print("Restoring from", latest_checkpoint)
              if self.model_type=="HUB" :
                return keras.models.load_model(latest_checkpoint ,custom_objects={'KerasLayer':hub.KerasLayer})
              return keras.models.load_model(latest_checkpoint)
          print("Creating a new model")
          return ModelUtils.get_compile_model(model,optimizer_type, lr)      

    def save_model_and_history(self,model,run_index) :
        models_path = os.path.join(self.save_dir ,"Models",self.model_name)
        if not os.path.exists(models_path) :
          os.makedirs(models_path)
        history_name = self.model_name+'_history_'+str(run_index)+'.npy'
        history_path = os.path.join(models_path ,history_name)
        np.save(history_path , model.history)
        model_path = os.path.join( models_path,str(run_index))
        if not os.path.exists(model_path):
          os.makedirs(model_path)
        if self.model_type =="HUB":
          model.save(model_path)
        else:
          tf.keras.models.save_model(model ,model_path)
    #history1=np.load('history1.npy',allow_pickle='TRUE').item() #This use for model analysis
