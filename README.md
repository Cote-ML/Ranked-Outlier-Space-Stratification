# Ranked-Outlier-Space-Stratification

Currently used in other personal projects, figured it might be time to give it its own repo. 

TODO: Lots and lots of busywork. 

- App-ify into a PyPi package (need a src/main/python/model/, __main__, app.py, src/test/python/model/, data/, notebooks/ (examples), and whatever else.)
- Upload test examples (use market model stuff)
- Take the time to make some unit & integration tests
- Expand this ReadMe (what is ROSS? What is an influence space and an INFLO score? Why does stratification work? Pretty pictures & stuff too, why not)
- Integrate to dynamically allow numpy and pandas use without outside fiddling. 
  - Mimic SciKit standard of allowing to pass in a distance matrix too, decompose squareform stuff into its own func. 
  - This is gonna suck, but needs to be done. 
