import boto3
import os
import json
from botocore.exceptions import ClientError

S3_BUCKET = os.getenv("S3_BUCKET", "vw1-telemedic")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

s3 = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# =========================================================
# S3 업로드 (이미지)
# =========================================================
def upload_file_to_s3(local_path: str, prefix: str, filename: str):
    """
    이미지 파일을 S3에 업로드 (하위 timestamp 폴더 없이)
    prefix 예시: api/test0001/251112/
    """
    try:
        # ✅ 불필요한 슬래시 제거
        prefix = prefix.strip("/")
        s3_key = f"{prefix}/{filename}"

        s3.upload_file(local_path, S3_BUCKET, s3_key)
        file_url = f"https://{S3_BUCKET}.s3.{REGION}.amazonaws.com/{s3_key}"
        print(f"✅ [S3 업로드 완료] {file_url}")
        return file_url
    except ClientError as e:
        print(f"⚠️ S3 업로드 실패: {e}")
        return None


# =========================================================
# S3 업로드 (JSON 결과)
# =========================================================
def upload_json_to_s3(data: dict, prefix: str, json_name: str):
    """
    분석 결과(JSON)를 S3에 업로드 (timestamp 폴더 없이)
    """
    try:
        prefix = prefix.strip("/")
        s3_key = f"{prefix}/{json_name}"

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

        file_url = f"https://{S3_BUCKET}.s3.{REGION}.amazonaws.com/{s3_key}"
        print(f"✅ [JSON 업로드 완료] {file_url}")
        return file_url
    except ClientError as e:
        print(f"⚠️ JSON 업로드 실패: {e}")
        return None
