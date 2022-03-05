TRAIN_CSV_PATH = "./dataset/train/*/*.csv"
TEST_CSV_PATH = "./dataset/test/*/*.csv"
NEW_TRAIN_CSV_PATH = "./dataset/train/*/new_*.csv"
NEW_TEST_CSV_PATH = "./dataset/test/*/new_*.csv"

TRAIN_JSON_PATH = "./dataset/train/*/*.json"

TRAIN_IMAGE_PATH = "./dataset/train/*/*.jpg"
TEST_IMAGE_PATH = "./dataset/test/*/*.jpg"

SUBMISSION_CSV_PATH = "./dataset/sample_submission.csv"

TRAIN_LABEL_CSV_PATH = "./dataset/train.csv"

SAVE_MODEL_NAME = "model.pt"
SELECT_COLUMNS = [
    "내부 온도 1 평균",
    "내부 온도 1 최고",
    "내부 온도 1 최저",
    "내부 습도 1 평균",
    "내부 습도 1 최고",
    "내부 습도 1 최저",
    "내부 이슬점 평균",
    "내부 이슬점 최고",
    "내부 이슬점 최저",
]

SELECT_NUMBER_OF_ROW = 512

crop_dict = {"1": "딸기", "2": "토마토", "3": "파프리카", "4": "오이", "5": "고추", "6": "시설포도"}
disease_dict = {
    "1": {
        "a1": "딸기잿빛곰팡이병",
        "a2": "딸기흰가루병",
        "b1": "냉해피해",
        "b6": "다량원소결핍 (N)",
        "b7": "다량원소결핍 (P)",
        "b8": "다량원소결핍 (K)",
    },
    "2": {
        "a5": "토마토흰가루병",
        "a6": "토마토잿빛곰팡이병",
        "b2": "열과",
        "b3": "칼슘결핍",
        "b6": "다량원소결핍 (N)",
        "b7": "다량원소결핍 (P)",
        "b8": "다량원소결핍 (K)",
    },
    "3": {
        "a9": "파프리카흰가루병",
        "a10": "파프리카잘록병",
        "b3": "칼슘결핍",
        "b6": "다량원소결핍 (N)",
        "b7": "다량원소결핍 (P)",
        "b8": "다량원소결핍 (K)",
    },
    "4": {
        "a3": "오이노균병",
        "a4": "오이흰가루병",
        "b1": "냉해피해",
        "b6": "다량원소결핍 (N)",
        "b7": "다량원소결핍 (P)",
        "b8": "다량원소결핍 (K)",
    },
    "5": {
        "a7": "고추탄저병",
        "a8": "고추흰가루병",
        "b3": "칼슘결핍",
        "b6": "다량원소결핍 (N)",
        "b7": "다량원소결핍 (P)",
        "b8": "다량원소결핍 (K)",
    },
    "6": {"a11": "시설포도탄저병", "a12": "시설포도노균병", "b4": "일소피해", "b5": "축과병"},
}
risk_dict = {0: "정상", 1: "초기", 2: "중기", 3: "말기"}

CROP_DISEASE_DICT = {}
c = 0
for crop_key in disease_dict:
    CROP_DISEASE_DICT[f"{crop_key}_00"] = c
    c += 1
    for disease_key in disease_dict[crop_key]:
        CROP_DISEASE_DICT[f"{crop_key}_{disease_key}"] = c
        c += 1

LABELS = [
    "1_00_0",
    "2_00_0",
    "2_a5_2",
    "3_00_0",
    "3_a9_1",
    "3_a9_2",
    "3_a9_3",
    "3_b3_1",
    "3_b6_1",
    "3_b7_1",
    "3_b8_1",
    "4_00_0",
    "5_00_0",
    "5_a7_2",
    "5_b6_1",
    "5_b7_1",
    "5_b8_1",
    "6_00_0",
    "6_a11_1",
    "6_a11_2",
    "6_a12_1",
    "6_a12_2",
    "6_b4_1",
    "6_b4_3",
    "6_b5_1",
]

LABEL_DICT = {key: value for key, value in zip(LABELS, range(len(LABELS)))}
# print(LABEL_DICT)
LABEL_DECODE_DICT = {value: key for key, value in zip(LABELS, range(len(LABELS)))}
# print(LABEL_DECODE_DICT)
