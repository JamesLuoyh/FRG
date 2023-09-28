# format_data.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.utils.io_utils import load_pickle,save_pickle


N=23700 # Clips off 5 samples (at random) to make total divisible by 150,
# the desired batch size

savename_features = './face_recog/features.pkl'
savename_race_labels = './face_recog/race_labels.pkl'
savename_age_labels = './face_recog/age_labels.pkl'
savename_gender_labels = './face_recog/gender_labels.pkl'
savename_sensitive_attrs = './face_recog/sensitive_attrs.pkl'

print("loading data...")
data = pd.read_csv('./face_recog/age_gender.csv')
# Shuffle data since it is in order of age, then gender
data = data.sample(n=len(data),random_state=42).iloc[:N]
# Convert pixels from string to numpy array
print("Converting pixels to array...")
data['pixels']=data['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))

# normalize pixels data
print("Normalizing and reshaping pixel data...")
data['pixels'] = data['pixels'].apply(lambda x: x/255)

# Reshape pixels array
X = np.array(data['pixels'].tolist())

## Converting pixels from 1D to 4D
features = X.reshape(X.shape[0],1,48,48)

# Extract gender labels
gender_labels = data['gender'].values
race_labels = data['ethnicity'].values
age_labels = data['age'].values

# The median of age is 29. We split the age label into two categories
# If it is above or equal 30, we label it with 1.
# If it is below 30, we label it with 0.

mask = age_labels >= 30
age_labels = mask.astype('int64')

# Make one-hot sensitive feature columns
race=data['ethnicity'].values.astype(int)
n_classes = race.max() + 1
sensitive_attrs = np.zeros([len(race), n_classes])
for i in range(len(race)):
  sensitive_attrs[i, race[i]] = 1
# mask=~(M.astype("bool"))
# F=mask.astype('int64')
# sensitive_attrs = np.hstack((M.reshape(-1,1),F.reshape(-1,1)))

# Save to pickle files
print("Saving features, labels, and sensitive_attrs to pickle files")
save_pickle(savename_features,features)
save_pickle(savename_race_labels,race_labels)
save_pickle(savename_gender_labels,gender_labels)
save_pickle(savename_age_labels,age_labels)
save_pickle(savename_sensitive_attrs,sensitive_attrs)
