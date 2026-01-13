from __future__ import annotations

from pathlib import Path
import sys
import streamlit as st
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

from src.inference.predictor import EmotionPredictor
from src.inference.face_detector import detect_and_crop_largest_face
from src.inference.labels import EMOTION_LABELS_EN, EMOTION_LABELS_JA


@st.cache_resource
def load_predictor(model_type):
    if model_type == "resnet":
        ckpt = CHECKPOINT_DIR / "best_resnet_fer2013.pth"
    else:
        pass

    predictor = EmotionPredictor(
        model_type=model_type,
        checkpoint_path=str(ckpt),
        device=None
    )
    return predictor


def pil_to_bgr(image):
    rgb = np.array(image)
    bgr = rgb[:, :, ::-1]
    return bgr


def draw_box_and_label(image_bgr, box, label_en, score):

    annotated = image_bgr.copy()
    x, y, w, h = box.as_tuple()

    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = f"{label_en} ({score:.2%})"
    cv2.putText(
        img=annotated,
        text=text,
        org=(x, y - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.1,
        color=(0, 255, 0),
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    return annotated


def render_sidebar():
    st.sidebar.header("設定")

    model_labels = {
        "resnet": "ResNet-18",
    }

    model_type = st.sidebar.selectbox(
        "モデルタイプ",
        options=list(model_labels.keys()),
        format_func=lambda k: model_labels[k],
    )

    return model_type


def get_input_image():
    input_mode = st.radio(
        "入力方法を選択してください",
        options=["画像アップロード", "Webカメラ"],
        horizontal=True,
    )

    if input_mode == "画像アップロード":
        file = st.file_uploader(
            "顔が写った画像をアップロードしてください",
            type=["jpg", "jpeg", "png"]
            )
        if file is None:
            return None
        img = Image.open(file).convert("RGB")
        st.image(img, caption="アップロード画像", use_container_width=True)
        return img

    camera_image = st.camera_input("Webカメラで撮影")
    if camera_image is None:
        return None
    img = Image.open(camera_image).convert("RGB")
    st.image(img, caption="撮影画像", use_container_width=True)
    return img


def run_inference_flow(uploaded_image, model_type):
    # loading model
    try:
        predictor = load_predictor(model_type)
    except FileNotFoundError as e:
        st.error(
            f"モデルファイルが見つかりませんでした。\n{e}\n"
            "train.py で学習を行い、.pth を"
            "models/checkpoints/ に配置してください。"
        )
        return

    # 2) detecting face
    bgr_image = pil_to_bgr(uploaded_image)
    box, face_img = detect_and_crop_largest_face(bgr_image, bgr=True)
    if face_img is None:
        st.error("顔が検出できませんでした。別の画像で試してみてください。")
        return

    # 3) inference
    result = predictor.predict_from_ndarray(face_img, bgr=True)
    class_id = result["class_id"]
    label_ja = result["label_ja"]
    label_en = result["label_en"]
    probs = result["probs"]
    confidence = probs[class_id]

    st.subheader("推論結果")
    st.markdown(
        f"**感情:** {label_ja}（{label_en}）  \n"
        f"**確信度:** {confidence:.2%}"
    )

    # 4) graph
    st.write("各感情クラスの確率:")
    prob_dict = {
        f"{EMOTION_LABELS_JA[i]} ({EMOTION_LABELS_EN[i]})": probs[i]
        for i in range(len(probs))
    }
    st.bar_chart(prob_dict)

    # 5) updating image
    st.write("検出された顔と推論結果:")
    annotated_bgr = draw_box_and_label(bgr_image, box, label_en, confidence)
    annotated_rgb = annotated_bgr[:, :, ::-1]
    st.image(
        annotated_rgb,
        use_container_width=True
        )


def main():
    st.set_page_config(
        page_title="Facial Expression Recognition",
        page_icon="🫠"
        )
    st.title("表情認識アプリ")
    st.write("アップロード画像 or Webカメラから顔の感情を推定します。\n"
             "画像や写真が保存される事はありません。")

    model_type = render_sidebar()
    uploaded_image = get_input_image()

    if st.button("表情を推定する"):
        if uploaded_image is None:
            st.warning("先に画像を用意してください。")
            return
        run_inference_flow(uploaded_image, model_type)



if __name__ == "__main__":
    main()
