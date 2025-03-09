FROM python:3.10-slim-bullseye as build

# ADD "https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e" /code/osnet_x0_25_msmt17.pt

RUN apt update && apt install --no-install-recommends -y \
    curl \
    git \
    python3-dev \
    gcc \
    g++ \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libjpeg-dev \
    libgl1 \
    libturbojpeg0

ARG POETRY_VERSION
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Copy only files that are necessary to install dependencies
COPY poetry.lock poetry.toml pyproject.toml /code/

WORKDIR /code
RUN poetry install --no-root
    
# Copy the rest of the project
COPY . /code/

WORKDIR /code
ENV PATH="/code/.venv/bin:$PATH"
CMD [ "python", "main.py" ]

# RUN: sudo docker build -t mcvy_yq/object_tracker_arm64:v1.0 --platform linux/arm64 .