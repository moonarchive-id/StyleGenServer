# import gradio as gr
# import cv2
# from PIL import Image
# import numpy as np
# import os
# from utils.face_shape_detector import detect_face_shape

# SAVE_PATH = "detected_faces"
# os.makedirs(SAVE_PATH, exist_ok=True)

# def analyze_and_save(img_pil):
#     if img_pil is None:
#         return None, "❌ Tidak ada gambar."

#     img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
#     shape, error, annotated = detect_face_shape(img_cv2)
#     result_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

#     if error:
#         return result_pil, error

#     # Simpan ke folder sesuai bentuk
#     folder = os.path.join(SAVE_PATH, shape.lower())
#     os.makedirs(folder, exist_ok=True)
#     filename = os.path.join(folder, f"{shape}_{len(os.listdir(folder)) + 1}.jpg")
#     cv2.imwrite(filename, img_cv2)

#     return result_pil, f"✅ Bentuk wajah: {shape}. Gambar disimpan di: {filename}"

# with gr.Blocks(theme=gr.themes.Soft()) as barbershop_ui:
#     gr.Markdown("## 💈 Barbershop AI - Deteksi Bentuk Wajah")
#     gr.Markdown("Upload atau capture wajah untuk mendeteksi bentuk wajah: **Oval**, **Round**, atau **Square**.")

#     with gr.Row():
#         input_img = gr.Image(label="📸 Wajah (Webcam/Upload)", type="pil")
#         output_img = gr.Image(label="🧠 Deteksi Landmark", type="pil", interactive=False)

#     analyze_btn = gr.Button("🔍 Deteksi Bentuk Wajah")
#     result_text = gr.Textbox(label="Hasil", interactive=False)

#     analyze_btn.click(fn=analyze_and_save, inputs=input_img, outputs=[output_img, result_text])

# if __name__ == "__main__":
#     barbershop_ui.launch(share=True, server_port=9000)
