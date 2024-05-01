# Use the latest Python image as the base image
FROM python:3.10.12

# Set the working directory inside the container
WORKDIR /home/suraj/Desktop/KCDH2/websiteForKCDH

# Copy the current directory contents into the container at the working directory
COPY . .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install necessary system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    poppler-utils \ 
 && rm -rf /var/lib/apt/lists/*




# Copy the requirements file into the container at a temporary location
COPY requirements.txt /tmp/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r /tmp/requirements.txt

# Command to run the Streamlit application
CMD ["streamlit", "run", "app4.py"]

EXPOSE 8001

