FROM continuumio/anaconda3
MAINTAINER adsferreira
EXPOSE 8000
WORKDIR /var/www/text_clustering_api/
COPY . /var/www/text_clustering_api/
WORKDIR /var/www/text_clustering_api/
RUN apt-get update && apt-get install -y apache2 \
    apache2-dev \
    vim \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt \
    && /opt/conda/bin/mod_wsgi-express install-module \
    && mod_wsgi-express setup-server text_clustering_api.wsgi --port=8000 \
       --user www-data --group www-data \
       --server-root=/etc/mod_wsgi-express-80
CMD /etc/mod_wsgi-express-80/apachectl start -D FOREGROUND
