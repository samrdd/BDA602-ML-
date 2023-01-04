FROM python:3.10
USER root

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
     libmariadb3 \
     libmariadb-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*


# Get Necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# copying the files
COPY finalproject .



# Running the script
ENTRYPOINT ["/bin/bash", "finalproject.sh"]