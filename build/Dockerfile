FROM tensorflow/tensorflow:latest-jupyter

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN apt update && apt upgrade -y 
RUN apt-get update && apt-get upgrade -y 
RUN apt-get install iputils-ping -y

# Install pip requirements
COPY build/requirements.txt .
RUN pip uninstall -y -r requirements.txt --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir

WORKDIR /tf
