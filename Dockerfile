FROM python:3.7-slim AS compile-image
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

COPY requirements.txt .
RUN pip install --user -r requirements.txt


FROM python:3.7-slim AS build-image
COPY --from=compile-image /root/.local /root/.local

COPY predict.py /root
COPY config.py /root
COPY led_model /root

# Make sure scripts in .local are usable:
WORKDIR /root
CMD ["python", "predict.py"]