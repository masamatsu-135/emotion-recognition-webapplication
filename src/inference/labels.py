EMOTION_LABELS_EN = [
    "Angry",     # 0
    "Disgust",   # 1
    "Fear",      # 2
    "Happy",     # 3
    "Sad",       # 4
    "Surprise",  # 5
    "Neutral",   # 6
]

EMOTION_LABELS_JA = [
    "怒り",        # 0
    "嫌悪",        # 1
    "恐れ",        # 2
    "幸福",        # 3
    "悲しみ",      # 4
    "驚き",        # 5
    "ニュートラル"  # 6
]

def get_label_en(class_id):
    return EMOTION_LABELS_EN[class_id]

def get_label_ja(class_id):
    return EMOTION_LABELS_JA[class_id]
