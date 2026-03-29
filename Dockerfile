FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN plotly_get_chrome

COPY . .

CMD ["python", "BTC_perp_data.py"]
