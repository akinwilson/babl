# Serving models 

## Overview 
FastAPI has been used to wrap an application server around our trained language models allowing their endpoints to be tested via an interactive API Documentation page. 

## Usage 
To develop upon the API, run 
```
pip install -r requirements.txt
```
To run the server in **development** mode, run 
```
fastapi dev api/main.py
```
To deploy the API via a containerised workflow, first build the image 
```
docker build . -f Dockerfile.serve -t serve:latest
```
then run the image to spin up a container 
```

```

