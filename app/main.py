from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, json, datetime
from PIL import Image
import boto3

from models_loader import MODELS
from s3_utils import upload_file_to_s3, upload_json_to_s3

# -------------------------------
# 기본 설정
# -------------------------------
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

S3_BUCKET = "vw1-telemedic"
REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
s3 = boto3.client("s3", region_name=REGION)

app = FastAPI(title="TeleMedic AI Analysis API")

# -------------------------------
# CORS 설정
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 헬스체크
# -------------------------------
@app.get("/")
def root():
    return {"message": "TeleMedic AI API is running."}


# =========================================================
# 1️⃣ 파일 업로드 및 분석 요청
# =========================================================
@app.post("/api/analysis")
async def analyze_image(
    userId: str = Form(...),
    modality: str = Form(...),
    file: UploadFile = File(...)
):
    modality = modality.lower().strip()
    if modality not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid modality. Choose from xray, ct, mri.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image files are allowed.")

    # 날짜(YYMMDD)
    today = datetime.datetime.now().strftime("%y%m%d")

    # 로컬 폴더 구조 생성
    local_img_dir = os.path.join(UPLOAD_DIR, userId, today, "img")
    local_json_dir = os.path.join(UPLOAD_DIR, userId, today, "json")
    os.makedirs(local_img_dir, exist_ok=True)
    os.makedirs(local_json_dir, exist_ok=True)

    # ✅ count 계산 (로컬 + S3)
    count_local = 0
    count_s3 = 0

    if os.path.exists(local_json_dir):
        local_jsons = [
            f for f in os.listdir(local_json_dir)
            if f.startswith(f"{userId}_{today}_{modality}") and f.endswith(".json")
        ]
        count_local = len(local_jsons)

    s3_prefix_json = f"api/{userId}/{today}/json/"
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix_json)
        if "Contents" in response:
            s3_jsons = [
                obj["Key"] for obj in response["Contents"]
                if f"_{modality}_" in obj["Key"] and obj["Key"].endswith(".json")
            ]
            count_s3 = len(s3_jsons)
    except Exception as e:
        print(f"⚠️ S3 count 조회 실패: {e}")

    count = max(count_local, count_s3) + 1

    # 파일명
    unique_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1] or ".png"
    filename_base = f"{userId}_{today}_{modality}_{count}"
    img_filename = f"{filename_base}{ext}"
    json_filename = f"{filename_base}.json"

    local_img_path = os.path.join(local_img_dir, img_filename)
    local_json_path = os.path.join(local_json_dir, json_filename)

    # 이미지 로컬 저장
    try:
        with open(local_img_path, "wb") as buffer:
            buffer.write(await file.read())
        Image.open(local_img_path).verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 모델 추론
    model = MODELS[modality]
    analysis_result = model.predict(local_img_path)
    if analysis_result.get("result") == "모델 로드 실패":
        raise HTTPException(status_code=500, detail="Model loading failed.")

    # ✅ S3 업로드 경로 (img / json 분리)
    s3_img_prefix = f"api/{userId}/{today}/img"
    s3_json_prefix = f"api/{userId}/{today}/json"

    image_url = upload_file_to_s3(local_img_path, s3_img_prefix, img_filename)
    if not image_url:
        raise HTTPException(status_code=500, detail="Image upload failed.")

    # JSON 데이터 구성
    result_data = {
        "id": unique_id,
        "userId": userId,
        "modality": modality,
        "imageUrl": image_url,
        "analysis": analysis_result,
        "date": today
    }

    # 로컬 JSON 저장
    with open(local_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # S3 JSON 업로드
    result_url = upload_json_to_s3(result_data, s3_json_prefix, json_filename)
    if not result_url:
        raise HTTPException(status_code=500, detail="Result upload failed.")

    return result_data


# =========================================================
# 2️⃣ 분석 결과 조회 (userId 전체 날짜)
# =========================================================
@app.get("/api/analysis")
def get_analysis_results(userId: str):
    """
    userId에 해당하는 모든 날짜(json 폴더)에서 결과 조회
    """
    if not userId:
        return {"error": "Invalid request. userId is required."}

    base_prefix = f"api/{userId}/"
    results = []

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=base_prefix):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue
                if "/json/" not in key:
                    continue  # ✅ json 폴더만 대상

                file_obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                content = file_obj["Body"].read().decode("utf-8")
                data = json.loads(content)
                results.append(data)

        # 최신 날짜 순 정렬
        results.sort(key=lambda x: x.get("date", ""), reverse=True)
        return results

    except Exception as e:
        print(f"⚠️ [GET /api/analysis] 조회 오류: {e}")
        return {"error": "Internal server error while retrieving analysis data."}

