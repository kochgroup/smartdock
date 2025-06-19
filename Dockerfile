FROM python:3.9

# set the working directory
WORKDIR /code

# Create a folder for aplication
RUN mkdir /appl
COPY ./requirements.txt /appl

# Install OpenGL libraries
RUN apt-get update && apt-get install -y \
libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libglapi-mesa \
    libgles2 \
    libopengl0 \
    libxkbcommon0 \
    libdbus-1-3 \
    libnss3 \
    mesa-utils \
    qtbase5-dev 
    
# install dependencies
RUN python3.9 -m pip install --no-cache-dir --upgrade -r /appl/requirements.txt
RUN python3.9 -m pip install --extra-index-url https://pip.ccdc.cam.ac.uk/ csd-python-api 

# Install Nano y Vim
RUN apt-get update && apt-get install -y nano vim

# Create a new work folder
RUN mkdir work 

# set the working directory
WORKDIR /code/work


