# Base Image
FROM python:3.9

# Create working directory
WORKDIR /paraph

# Copy requirements to working dir
COPY requirements.txt .

# install requir
RUN pip install -r requirements.txt

# copy all files
COPY . .

# run app
CMD ["python", "./app.py"]
