FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN echo "estoy aqui"

CMD ["python3", "./starter/mi_primer_feature_eng.py"]