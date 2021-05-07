# Ranked-Outlier-Space-Stratification

Optimized, **non-parametric**, numpy-coded package which takes any NxM data array and stratifies data by levels of outlierness. It will dynamically determine the amount of outlier strata by itself, depending on the underlying data. 

Usage:
```
from ross import ross

df['label'] = ross.fit_predict(np.array(df))
```
Higher valued labels = more of an outlier / "more strange"

![image](https://user-images.githubusercontent.com/47681284/117406023-b8fa4100-aec9-11eb-8806-5a43d76a600c.png)

Keyword binary_flag=True turns into a straightforward outlier detection algo.

![image](https://user-images.githubusercontent.com/47681284/117404897-feb60a00-aec7-11eb-9287-ca05f56ad6d9.png)


