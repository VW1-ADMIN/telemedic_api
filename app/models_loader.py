import os
import boto3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ======================================================
# 1️⃣ AWS S3 설정
# ======================================================
S3_BUCKET = os.getenv("S3_BUCKET", "vw1-telemedic")
MODEL_PREFIX = "model/"
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

s3 = boto3.client(
    "s3",
    region_name=REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# ======================================================
# 2️⃣ 로컬 모델 저장 경로
# ======================================================
LOCAL_MODEL_DIR = Path("/app/model")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# ======================================================
# 3️⃣ ConvNeXt Tiny 모델 구조 정의
# ======================================================
class MedicalImagingConvNeXtTiny(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = models.convnext_tiny(weights=None)
        in_feats = self.base_model.classifier[-1].in_features
        self.base_model.classifier = nn.Identity()
        if not hasattr(self.base_model, "avgpool"):
            self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_feats),
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.base_model.features(x)
        x = self.base_model.avgpool(feats)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ======================================================
# 4️⃣ 클래스 라벨 (영문 → 한글)
# ======================================================
# CLASS_MAP = {
#     "ct": [
#         "선암(adenocarcinoma)",
#         "대세포암(large.cell.carcinoma)",
#         "정상(normal)",
#         "편평상피세포암(squamous.cell.carcinoma)"
#     ],
#     "mri": [
#         "성상세포종(Astrocytoma)", "암종(Carcinoma)", "상피종(Ependimoma)", "신경교종(Ganglioglioma)",
#         "배세포종(Germinoma)", "교모세포종(Glioblastoma)", "육아종(Granuloma)", "수모세포종(Meduloblastoma)",
#         "수막종(Meningioma)", "신경세포종(Neurocitoma)", "정상(NORMAL)", "희소돌기교종(Oligodendroglioma)",
#         "유두종(Papiloma)", "신경초종(Schwannoma)", "결핵종(Tuberculoma)"
#     ],
#     "xray": [
#         "세균성 폐렴(Bacterial)", "정상(Normal)", "바이러스성 폐렴(Viral)"
#     ]
# }

# Global 버전 CLASS 
CLASS_MAP = {
    "ct": [
        "adenocarcinoma",
        "large.cell.carcinoma",
        "normal",
        "squamous.cell.carcinoma"
    ],
    "mri": [
        "Astrocytoma", "Carcinoma", "Ependimoma", "Ganglioglioma",
        "Germinoma", "Glioblastoma", "Granuloma", "Meduloblastoma",
        "Meningioma", "Neurocitoma", "NORMAL", "Oligodendroglioma",
        "Papiloma", "Schwannoma", "Tuberculoma"
    ],
    "xray": [
        "Bacterial", "Normal", "Viral"
    ]
}

# ======================================================
# 5️⃣ 이미지 전처리 정의
# ======================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======================================================
# 6️⃣ 모델 다운로드 함수
# ======================================================
def download_model_from_s3(model_name: str) -> Path:
    local_path = LOCAL_MODEL_DIR / model_name
    s3_key = f"{MODEL_PREFIX}{model_name}"

    if local_path.exists():
        print(f"✅ [모델 캐시됨] {model_name} 이미 존재, 다운로드 생략")
        return local_path

    print(f"⬇️ [S3 다운로드 시작] s3://{S3_BUCKET}/{s3_key}")
    s3.download_file(S3_BUCKET, s3_key, str(local_path))
    print(f"✅ [다운로드 완료] {local_path}")
    return local_path


# ======================================================
# 7️⃣ 모델 래퍼 클래스
# ======================================================
class ModelWrapper:
    def __init__(self, name: str, s3_filename: str, modality: str):
        self.name = name
        self.modality = modality
        self.file_path = download_model_from_s3(s3_filename)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        try:
            print(f"🔹 [{self.name}] 모델 로드 중...")
            num_classes = len(CLASS_MAP[self.modality])
            model = MedicalImagingConvNeXtTiny(num_classes=num_classes).to(self.device)
            state_dict = torch.load(self.file_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            self.model = model
            print(f"✅ [{self.name}] 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ [{self.name}] 로드 실패: {e}")
            self.model = None

    def predict(self, image_path: str):
        if self.model is None:
            return {"result": "모델 로드 실패", "confidence": 0}

        try:
            img = Image.open(image_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = round(probs[pred_idx].item() * 100, 2)
                class_name = CLASS_MAP[self.modality][pred_idx]

            return {
                "result": class_name,
                "probability": confidence,
                "confidence": confidence,
                "findings": f"The {self.name} model diagnosed it as '{class_name}'."
            }

        except Exception as e:
            print(f"⚠️ [{self.name}] 추론 실패: {e}")
            return {"result": "에러 발생", "confidence": 0}

## {self.name} 모델이 '{class_name}'으로 진단했습니다.
# ======================================================
# 8️⃣ 전체 모델 로드
# ======================================================
def load_models():
    print("🔹 S3 모델 다운로드 및 로드 시작...")
    models = {
        "xray": ModelWrapper("X-Ray", "xray3_20251024_101104_final.pth", "xray"),
        "ct":   ModelWrapper("CT",    "CT1_20251016_175034_final.pth",   "ct"),
        "mri":  ModelWrapper("MRI",   "mri2_20251029_192936_final.pth",  "mri")
    }
    print("✅ 모든 모델 로드 완료")
    return models


# ======================================================
# 9️⃣ MODELS 전역 객체
# ======================================================
MODELS = load_models()
