# Python image to use. #alpine
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .

RUN apt-get update && apt-get install swig -y

# Install any needed packages specified in requirements.txt
#RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Run the script to create embeddings
#RUN python create_embeddings.py

# Run app.py when the container launches
ENTRYPOINT ["python", "app.py"]
