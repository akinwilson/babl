# !/bin/bash


echo "Running server with either 2 or ${WORKERS} process workers . Set with WORKER env variable ..."
echo "Running on port 80 with process worker lifetime of either 300 seconds or ${TIMEOUT}. Set with TIMEOUT env variable  ... "

gunicorn -w ${WORKERS:=2} \ 
  -b :80 -t ${TIMEOUT:=300} \
  -k uvicorn.workers.UvicornWorker \
  main:app
