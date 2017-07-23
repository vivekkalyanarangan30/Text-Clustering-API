# Text-Clustering-API
Implementation of a text clustering algorithm using Kmeans clustering in order to derive quick insights from unstructured text.
Please check the below links for details - 

+ https://machinelearningblogs.com/2017/01/26/text-clustering-get-quick-insights-from-unstructured-data/
+ https://machinelearningblogs.com/2017/01/26/text-clustering-get-quick-insights-from-unstructured-data/

## Docker Setup
0. Install [Docker](https://docs.docker.com/engine/installation/)
1. Run `git clone https://github.com/vivekkalyanarangan30/Text-Clustering-API`
2. Run `cd Text-Clustering-API/`
3. Open docker terminal and navigate to `/path/to/Text-Clustering-API`
4. Run `docker build -t clustering-api .`
5. Run `docker run -p 8180:8180 clustering-api

## Native Setup
1. Anaconda distribution of python 2.7
2. `pip install -r requirements.txt`
3. Some dependencies from *nltk* (`nltk.download()` from python console and download averaged perceptron tagger)

### Run it
1. Place the script in any folder
2. Open command prompt and navigate to that folder
3. Type "python CLAAS.py"and hit enter
4. Go over to http://localhost:8180/apidocs/index.html in your browser (preferably Chrome) and start using.