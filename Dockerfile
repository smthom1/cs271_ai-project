# use python 3.10 slim image
FROM python:3.10-slim

# keep python output visible in logs
ENV PYTHONUNBUFFERED=1

# set the working folder
WORKDIR /app

# install system tools for gui (pygame and tkinter)
RUN apt-get update && apt-get install -y \
    python3-tk \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    libjpeg-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# install python libraries found in your imports
RUN pip install --no-cache-dir \
    numpy \
    gymnasium \
    pygame \
    six \
    pandas \
    matplotlib

# copy all files into the container
COPY . /app

# install the local package (setup.py)
RUN pip install -e .

# default to bash shell so you can choose which agent to run
CMD ["/bin/bash"]