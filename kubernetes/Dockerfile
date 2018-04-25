# Read from Ubuntu Base Image
FROM python:2.7
RUN mkdir -p /service
# Copy over all the files of interest
ADD app /service/app
ADD app.py /service/app.py
ADD config.py /service/config.py
ADD manage.py /service/manage.py
ADD requirements.txt /service/requirements.txt
# more added files
ADD nltk.txt /service/nltk.txt
ADD political_leanings.json /service/political_leanings.json
# end added files 
WORKDIR /service/
RUN pip install -r requirements.txt
#add nltk downloads
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet 
#end nltk downloads
CMD python -u app.py $APP_SETTINGS $DATABASE_URL
