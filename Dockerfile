# 1. RunPod 공식 캐시 이미지 사용 (PyTorch 2.2, CUDA 12.1, Python 3.10 포함)
# 이 이미지는 RunPod 서버에 이미 있어서 다운로드/압축해제 시간이 0초에 가깝습니다.
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# 2. 쉘 설정 (에러 발생 시 즉시 중단)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 3. 작업 경로 설정
WORKDIR /

# 4. 필수 라이브러리 설치
# (주의: requirements.txt에서 torch, xformers, torchvision은 꼭 지웠는지 확인하세요!)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# 5. 소스 코드 복사
# download_weights.py는 이제 필요 없으니 뺍니다.
COPY handler.py schemas.py test_input.json /

# 6. 실행 명령어
CMD ["python", "-u", "/handler.py"]