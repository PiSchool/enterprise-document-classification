# Data Loading
import preproc
import pandas as pd
from time import time

C={'letter': 0,
   'form': 1,
   'email': 2,
   'handwritten': 3,
   'advertisement': 4,
   'scientific report': 5,
   'scientific publication': 6,
   'specification': 7,
   'file folder': 8,
   'news article': 9,
   'budget': 10,
   'invoice': 11,
   'presentation': 12,
   'questionnaire': 13,
   'resume': 14,
   'memo': 15}

# Trial Parameters
Nr= 764
Nc= 764
f_method='phash'
n_train=14
n_val= 5
c_method='ParForest'
hash_s= 256
dataset_path= "/Users/robi/AnacondaProjects/AutoDocProject/rvl-cdip/"
results_path= "/Users/robi/AnacondaProjects/AutoDocProject/Results_Images/"

# Read the training set file and put it in a dictionary: "Class name-List of files"
df_train= pd.read_csv(dataset_path+"labels/train.txt", delim_whitespace=True,
                dtype={'Filename': str,'Class': int})
df_test= pd.read_csv(dataset_path+"labels/test.txt", delim_whitespace=True,
                dtype={'Filename': str,'Class': int})
df_val= pd.read_csv(dataset_path+"labels/val.txt", delim_whitespace=True,
                dtype={'Filename': str,'Class': int})

# create_data: Read the dataset filenames per class and count number of images
dataset, nitems= preproc.create_data(df_train, df_test,df_val)


# create_training: Get batch of Nb images per class, full resolution
train_set, y_train= preproc.create_sample(dataset, nitems, dataset_path, Nr, Nc, label='train', Nbatch=n_train)

#%% Image Test
import matplotlib.pyplot as plt
import numpy as np

plt.imshow( np.reshape( train_set[88,:], (Nr,Nc) ), cmap=plt.get_cmap('gray') ),
plt.colorbar()

plt.show()

#t0= time()
#%% Feature Extraction
import features

trfeat_set= features.feat_extr(train_set,Nr,Nc, feat_type=f_method, hash_size=hash_s)
#del train_set

#%% Image Test
import matplotlib.pyplot as plt
import numpy as np

NN= 210
rN= int( np.sqrt( int((hash_s**2)/4) ) )

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow( np.reshape( train_set[NN,:], (Nr,Nc) ), cmap=plt.get_cmap('gray'), aspect='auto' ), ax1.set_title("Original Document "+str(Nr)+"x"+str(Nc))
ax2.imshow( np.reshape( trfeat_set[NN,:rN**2], (rN,rN) ), cmap=plt.get_cmap('gray'), aspect='auto' ), ax2.set_title("Perceptual Hash "+str(rN)+"x"+str(int(rN/2)))
plt.colorbar()

plt.show()

#%% Learning on training set
import learning
# Only in case of phash or ahash or whash0;des
clasf= learning.learn(trfeat_set,y_train, method=c_method)

    
#%% Prediction on test set

# create_validation: Get batch of Nb images per class, full resolution
test_set, y_test,Nr,Nc= preproc.create_sample(dataset, nitems, label='test', Nbatch=n_val)
tefeat_set= features.feat_extr(test_set,Nr,Nc, feat_type=f_method, hash_size=hash_s)
del test_set

# Prediction
yp= clasf.predict(tefeat_set)

#print("Done in %0.3fs" % (time()-t0))
#%% Validation
import validation
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yp)
np.set_printoptions(precision=2)
	
# Plot non-normalized confusion matrix
#plt.figure()
#validation.plot_confusion_matrix(cnf_matrix, classes=C,
#	                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
validation.plot_confusion_matrix(cnf_matrix, classes=C, normalize=True,
	                      title='Normalized confusion matrix - '+f_method+'+'+c_method)
	
plt.show()

    

