FROM nadrino/gundam:latest

WORKDIR /workspace

COPY . /workspace

RUN python -m venv /opt/gundam-interface-venv

ENV PATH="/opt/gundam-interface-venv/bin:${PATH}"

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]"

CMD ["python", "-m", "pytest", "-v"]
