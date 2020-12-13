FROM python:3.6

RUN apt-get update && apt-get install -y \
	python3-dev \
	build-essential \
	nano \
    libhunspell-dev \
    hunspell \
    gunicorn \
	python3-enchant libicu-dev \
	libenchant1c2a gettext \
	libaio-dev libaio1 --no-install-recommends && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Set env variables used in this Dockerfile
ENV PYTHONUNBUFFERED 1
ENV APPLICATION_PORT 5000

# Set the application directory
WORKDIR /classify_v2_service

# Install our requirements.txt
ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy our code from the current folder to /app inside the container
ADD . /classify_v2_service

ENTRYPOINT ["/classify_v2_service/entrypoints/celery_entrypoint.sh"]
ENTRYPOINT ["/classify_v2_service/entrypoints/entrypoint.sh"]

CMD ["flask","run"]
HEALTHCHECK --interval=5s --timeout=5s --retries=5 \
    CMD wget http://localhost:$APPLICATION_PORT/health_check -O /dev/null
