# Ranked Outlier Space Stratification (R.O.S.S.)

Module which takes any float-valued NxM array and stratifies the data by levels of outlierness. This method is novel in that it, unlike all other outlier detection methods-- e.g., Isolation Forests, One Class SVM's, etc.-- is *entirely non-parametric and not reliant on a random seed.* You do not need to tell its contamination rate like IsoForest nor the NN's to check in KNN or Local Outlier Factor. It will dynamically determine the pollution rate of the set, *and determine the amount of outlier strata that exist unique to that data.* 

These gains make it asymptotically O(N) linear run time. Beyond its computational efficiencies gained through no hyperparameter tuning, this method has the novel gain of being specifically designed at handling three major issues of outlier detection:

1. Swamping/Masking
2. Multiple centroids of varying density
3. Connected centroids

![download (1)](https://user-images.githubusercontent.com/47681284/121797110-66542900-cbdb-11eb-99a2-703370657fe8.png)

If you are uninterested in 'levels' of outlierness, a simple keyword `binary_flag = True` will assert the results into an inlier/outlier dynamic.

# User Guide

### CLI installation (Linux/OSX)
```
git clone git@github.com:BrianCote/Ranked-Outlier-Space-Stratification.git
cd Ranked-Outlier-Space-Stratification
sudo pip3 install . 
pip3 show -f RankedOutliers #Confirm successful installation
```

### Python Use
```
from ross import model

df = pd.read_csv('/path/to/data.csv')
df['labels'] = model.fit_predict(np.array(df))
```
