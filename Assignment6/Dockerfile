FROM python:3.10.8

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
    mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# copying the files
COPY baseball.sql .
COPY Assignment6.sql .
COPY Assignment6.sh .

# Running the script
ENTRYPOINT ["/bin/bash", "Assignment6.sh"]