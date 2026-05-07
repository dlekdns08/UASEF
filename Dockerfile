# UASEF — reproducibility container (audit 6.10)
#
# 사용:
#   docker build -t uasef:latest .
#   docker run --rm \
#       -e OPENAI_API_KEY=$OPENAI_API_KEY \
#       -v "$PWD/data/raw:/app/data/raw:ro" \
#       -v "$PWD/results:/app/results" \
#       uasef:latest \
#       python experiments/run_all_experiments.py --backend openai --n-cal 100 --n-test 20

FROM python:3.11-slim

# 비-루트 사용자 생성 (권장)
RUN useradd -m -u 1000 uasef
WORKDIR /app

# 시스템 의존성 (matplotlib build, 일부 datasets dep)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# 의존성 캐시를 위해 pyproject만 먼저 복사
COPY pyproject.toml uv.lock* ./
# uv 사용 (lock 파일 활용)
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache .

# 소스 복사
COPY models/      ./models/
COPY agent/       ./agent/
COPY data/        ./data/
COPY experiments/ ./experiments/

# 실행 시 결과 디렉토리는 volume mount 권장
RUN mkdir -p results data/raw \
    && chown -R uasef:uasef /app

USER uasef

# 기본 진입점: --help로 사용법 안내
ENTRYPOINT ["python"]
CMD ["experiments/run_all_experiments.py", "--help"]
