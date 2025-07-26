FROM python:3.12-slim
WORKDIR /app
RUN apt install g++
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]