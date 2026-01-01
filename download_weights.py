import os
import urllib.request
import shutil
import hashlib
import re
import json

# ==========================================
# [설정] 환경 변수 및 경로
# ==========================================
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/noobai-xl-1.1.safetensors")
LORA_CACHE_DIR = "/runpod-volume/models/loras"
DOWNLOAD_TIMEOUT = 300

# ==========================================
# [핵심] 리다이렉트 제어 핸들러
# ==========================================
class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        return None
    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302

def get_download_url(api_url, token=None):
    print(f"🔗 다운로드 링크 추출 중: {api_url}")
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    
    # 공백 제거 후 토큰 적용
    if token and "civitai.com" in api_url:
        headers["Authorization"] = f"Bearer {token.strip()}"

    req = urllib.request.Request(api_url, headers=headers)
    opener = urllib.request.build_opener(NoRedirectHandler)
    
    try:
        response = opener.open(req)
        return api_url
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307):
            redirect_url = e.headers.get('Location')
            
            # [수정] 로그인 페이지로 튕기는 경우 감지
            if "/login" in redirect_url or "auth" in redirect_url:
                print("❌ 오류: 인증 실패! 토큰이 없거나 만료되어 로그인 페이지로 리다이렉트되었습니다.")
                print(f"   리다이렉트 URL: {redirect_url}")
                return None
                
            if redirect_url:
                print("✅ 실제 다운로드 URL 확보 완료 (Cloudflare R2)")
                return redirect_url
        
        print(f"❌ URL 추출 실패: {e.code} {e.reason}")
        return None
    except Exception as e:
        print(f"❌ URL 추출 중 에러: {e}")
        return None

def download_file(url, destination, token=None):
    if "civitai.com/api/download" in url:
        real_url = get_download_url(url, token)
        if not real_url:
            return False
        target_url = real_url
        use_headers = {"User-Agent": "Mozilla/5.0"}
    else:
        target_url = url
        use_headers = {"User-Agent": "Mozilla/5.0"}
        if token and "civitai.com" in url:
            use_headers["Authorization"] = f"Bearer {token.strip()}"

    print(f"⬇️ 다운로드 시작 (URL 숨김처리)...")
    print(f"📂 저장 경로: {destination}")

    req = urllib.request.Request(target_url, headers=use_headers)
    
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as response:
            total_size = response.headers.get('content-length')
            if total_size:
                print(f"📦 파일 크기: {int(total_size) / (1024*1024):.2f} MB")
            
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, 'wb') as f:
                shutil.copyfileobj(response, f)

            if os.path.getsize(destination) < 10240:
                print("⚠️ 경고: 파일이 너무 작습니다 (에러 페이지 가능성). 삭제합니다.")
                os.remove(destination)
                return False
            
            print(f"✅ 다운로드 성공!")
            return True
            
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def get_lora_cache_path(lora_source, custom_name=None):
    """
    LoRA가 저장될 경로를 결정합니다.
    custom_name이 있으면 그것을 우선하여 파일명으로 사용합니다.
    """
    os.makedirs(LORA_CACHE_DIR, exist_ok=True)
    
    # 1. 사용자가 이름을 지정한 경우 (최우선)
    if custom_name:
        # 확장자가 없으면 붙여줌
        if not custom_name.endswith('.safetensors'):
            filename = custom_name + '.safetensors'
        else:
            filename = custom_name
        return os.path.join(LORA_CACHE_DIR, filename)

    # 2. 로컬 경로인 경우 (이미 파일명만 입력한 경우)
    if lora_source.startswith(LORA_CACHE_DIR):
        return lora_source
    if '/' not in lora_source and '\\' not in lora_source:
        # "my_lora.safetensors" 처럼 파일명만 온 경우
        if not lora_source.endswith('.safetensors'):
            lora_source += '.safetensors'
        return os.path.join(LORA_CACHE_DIR, lora_source)
    
    # 3. URL인 경우 (이름 지정 없으면 기존대로 해시 사용)
    if lora_source.startswith('http'):
        url_hash = hashlib.md5(lora_source.encode()).hexdigest()[:12]
        return os.path.join(LORA_CACHE_DIR, f"lora_{url_hash}.safetensors")
    
    # 4. 그 외 (HuggingFace 등)
    filename = lora_source.replace('/', '_').replace('\\', '_')
    if not filename.endswith('.safetensors'):
        filename += '.safetensors'
    return os.path.join(LORA_CACHE_DIR, filename)


def download_lora(lora_source, token=None, custom_name=None):
    """
    핸들러에서 호출하는 메인 함수 (custom_name 추가됨)
    """
    # [추가] Civitai URL 쿼리 파라미터제거 (예: ?type=Model 제거)
    if "civitai.com/api/download/models/" in lora_source and "?" in lora_source:
        lora_source = lora_source.split("?")[0]
        print(f"🧹 Cleaned Civitai URL: {lora_source}")

    # 경로 계산 시 custom_name 전달
    cache_path = get_lora_cache_path(lora_source, custom_name)
    
    if os.path.exists(cache_path):
        print(f"♻️ 캐시된 LoRA 사용: {cache_path}")
        return cache_path
    
    # URL이 아닌데 파일도 없다면? (재사용 시 파일명이 틀린 경우 등)
    if not lora_source.startswith("http") and not "/" in lora_source:
        print(f"❌ 오류: '{lora_source}' 파일을 찾을 수 없습니다. (URL이 아니므로 다운로드 불가)")
        return None

    # 인자로 받은 토큰 우선, 없으면 환경변수 확인
    final_token = token or os.environ.get("CIVITAI_API_TOKEN")
    
    # 다운로드 실행
    success = download_file(lora_source, cache_path, final_token)
    return cache_path if success else None

if __name__ == "__main__":
    print("🚀 download_weights.py 로컬 테스트 모드")
    
    # 테스트 URL
    TEST_URL = "https://civitai.com/api/download/models/1536582"
    
    # [중요] 여기에 본인의 새 토큰을 입력하세요
    TEST_TOKEN = "여기에_토큰을_입력하세요"
    
    # 로컬 테스트용 경로 설정
    LORA_CACHE_DIR = "./test_downloads"
    
    # [수정] 함수 호출 시 토큰을 전달하도록 변경
    download_lora(TEST_URL, token=TEST_TOKEN)