FROM continuumio/anaconda:4.4.0
MAINTAINER Vivek Kalyanarangan, https://machinelearningblogs.com/about/
COPY python/ /usr/local/python/
EXPOSE 8180
WORKDIR /usr/local/python/
RUN pip install -r requirements.txt \
    && python -m nltk.downloader averaged_perceptron_tagger
CMD python CLAAS_public.py
