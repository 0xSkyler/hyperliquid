FROM python:3.12-slim
WORKDIR /srv
COPY pyproject.toml README.md ./
COPY app app
COPY backtest backtest
RUN pip install --no-cache-dir ".[fast,postgres]" && useradd -r -u 10001 trader && mkdir data && chown trader data
USER trader
ENV HL_DASHBOARD_HOST=0.0.0.0
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8787/api/state', timeout=4)"
CMD ["python", "-m", "app.main"]
