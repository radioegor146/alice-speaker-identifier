FROM python:3.12-slim
WORKDIR /app
RUN apt update && apt install -y g++
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]