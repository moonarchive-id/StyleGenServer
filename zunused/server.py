import os
import io
import asyncio
import socket
import traceback
import contextlib
import numpy as np
import cv2
import math

import mediapipe as mp

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from hair_swap import HairFast, get_parser

import uvicorn
import gradio as gr
# --- IMPORT BARU UNTUK ZEROCONF ---
from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, RedirectResponse

app = FastAPI(
    title="HairFast AI API & Web UI",
    description="API untuk Flutter & Web UI Gradio dengan logika yang konsisten.",
    version="Studio-V17-Zeroconf-Fixed",
)

hair_fast_model: HairFast = None 
zeroconf_instance_global: AsyncZeroconf = None
mp_face_mesh = mp.solutions.face_mesh

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global hair_fast_model, zeroconf_instance_global
    
    print("Memulai Lifespan Event: Startup...", flush=True)

    # Menentukan IP dan menyimpannya di app.state agar bisa diakses Zeroconf
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        true_ip = s.getsockname()[0]
    except Exception:
        true_ip = '127.0.0.1' 
    finally:
        s.close()
    app.state.true_ip = true_ip 
    print(f"IP Asli Server Terdeteksi: {app.state.true_ip}", flush=True)

    # Memuat model AI seperti biasa
    model_parser = get_parser(); model_args = model_parser.parse_args(args=['--device', 'cuda' if torch.cuda.is_available() else 'cpu', '--ckpt', 'pretrained_models/StyleGAN/ffhq.pt', '--blending_checkpoint', 'pretrained_models/Blending/checkpoint.pth', '--rotate_checkpoint', 'pretrained_models/Rotate/rotate_best.pth', '--pp_checkpoint', 'pretrained_models/PostProcess/pp_model.pth'])
    hair_fast_model = HairFast(model_args); print("Model HairFast (untuk Swap) berhasil dimuat!", flush=True)
    
    # Logika pendaftaran Zeroconf dari kodemu
    port = 5000
    try:
        service_info = ServiceInfo(
            "_http._tcp.local.",
            "My HairFast API._http._tcp.local.", 
            addresses=[socket.inet_aton(app.state.true_ip)],
            port=port,
            properties={'path': '/'},
            server=f"{socket.gethostname()}.local."
        )
        zeroconf_instance_global = AsyncZeroconf() 
        await zeroconf_instance_global.async_register_service(service_info)
        print(f"Pendaftaran layanan Zeroconf 'My HairFast API' berhasil!", flush=True)
    except Exception as e:
        print(f"ERROR: Gagal mendaftarkan layanan Zeroconf: {e}", flush=True)

    yield

    # Logika shutdown Zeroconf dari kodemu
    print("Memulai Lifespan Event: Shutdown...", flush=True)
    if zeroconf_instance_global:
        print("Membatalkan pendaftaran layanan Zeroconf...", flush=True)
        # ServiceInfo harus dibuat ulang untuk unregister
        service_info_to_unregister = ServiceInfo(
            "_http._tcp.local.",
            "My HairFast API._http._tcp.local.",
            addresses=[socket.inet_aton(app.state.true_ip)],
            port=5000
        )
        await zeroconf_instance_global.async_unregister_service(service_info_to_unregister)
        await zeroconf_instance_global.async_close()
        print("Layanan Zeroconf telah dihentikan.", flush=True)


app.router.lifespan_context = lifespan

def local_warp_effect(image, center, radius, strength):
    h, w, _ = image.shape
    map_x = np.zeros((h, w), dtype=np.float32); map_y = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dist_x, dist_y = x - center[0], y - center[1]; distance = math.sqrt(dist_x**2 + dist_y**2)
            if distance < radius:
                factor = ((radius - distance) / radius) ** 2
                new_x, new_y = x - strength * dist_x * factor, y - strength * dist_y * factor
                map_x[y, x], map_y[y, x] = new_x, new_y
            else: map_x[y, x], map_y[y, x] = float(x), float(y)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def edit_wajah_mediapipe(face_pil, p_skin, p_eyes, p_nose_bridge, p_nose_tip, p_lips, p_forehead_width, p_forehead_height, p_jaw):
    try:
        if face_pil is None: return None, "❌ Silakan unggah gambar wajah terlebih dahulu."
        original_cv = cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR); edited_cv = original_cv.copy()
        h, w, _ = original_cv.shape

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks: return face_pil, "⚠️ Wajah tidak terdeteksi di gambar."
            
            landmarks = results.multi_face_landmarks[0].landmark

            if p_skin != 0:
                face_oval_indices = mp_face_mesh.FACEMESH_FACE_OVAL
                face_points = np.array([(int(landmarks[i[0]].x * w), int(landmarks[i[0]].y * h)) for i in face_oval_indices])
                face_mask = np.zeros((h, w), dtype=np.uint8); cv2.fillConvexPoly(face_mask, face_points, 255);
                face_mask_blurred = cv2.GaussianBlur(face_mask, (21, 21), 0)
                if p_skin > 0:
                    low_pass = cv2.GaussianBlur(edited_cv, (21, 21), 0); high_pass = cv2.subtract(edited_cv, low_pass)
                    smoothed_low_pass = cv2.bilateralFilter(low_pass, int(p_skin*1.5)*2+1, 75, 75)
                    modified_face = cv2.add(smoothed_low_pass, high_pass); blend_ratio = p_skin / 25.0
                else:
                    amount = abs(p_skin) / 15.0; blurred = cv2.GaussianBlur(edited_cv, (0,0), 3)
                    modified_face = cv2.addWeighted(edited_cv, 1.0 + amount, blurred, -amount, 0); blend_ratio = amount * 0.75
                blended_face = cv2.addWeighted(edited_cv, 1 - blend_ratio, modified_face, blend_ratio, 0)
                edited_cv = np.where(face_mask_blurred[:, :, None].astype(bool), blended_face, edited_cv)

            if p_eyes != 0:
                strength = p_eyes * -0.045
                radius = int(w * 0.08)
                left_eye_center = (int(landmarks[473].x * w), int(landmarks[473].y * h)); edited_cv = local_warp_effect(edited_cv, left_eye_center, radius, strength)
                right_eye_center = (int(landmarks[468].x * w), int(landmarks[468].y * h)); edited_cv = local_warp_effect(edited_cv, right_eye_center, radius, strength)
                
            if p_nose_bridge != 0:
                strength = p_nose_bridge * 0.035; radius = int(h * 0.22)
                bridge_center = (int(landmarks[168].x * w), int(landmarks[168].y * h)); edited_cv = local_warp_effect(edited_cv, bridge_center, radius, strength)
            
            if p_nose_tip != 0:
                strength = p_nose_tip * 0.05; radius = int(h * 0.12)
                tip_center = (int(landmarks[4].x * w), int(landmarks[4].y * h)); edited_cv = local_warp_effect(edited_cv, tip_center, radius, strength)

            if p_lips != 0:
                strength = p_lips * -0.07
                radius = int(w * 0.1)
                upper_lip_center = (int(landmarks[0].x*w), int(landmarks[0].y*h))
                lower_lip_center = (int(landmarks[17].x*w), int(landmarks[17].y*h))
                edited_cv = local_warp_effect(edited_cv, upper_lip_center, radius, strength)
                edited_cv = local_warp_effect(edited_cv, lower_lip_center, radius, strength)

            if p_forehead_width != 0:
                strength = p_forehead_width * -0.07
                radius = int(h * 0.2)
                left_temple = (int(landmarks[103].x*w), int(landmarks[103].y*h))
                right_temple = (int(landmarks[332].x*w), int(landmarks[332].y*h))
                edited_cv = local_warp_effect(edited_cv, left_temple, radius, -strength)
                edited_cv = local_warp_effect(edited_cv, right_temple, radius, -strength)
                
            if p_forehead_height != 0:
                strength = p_forehead_height * -0.06
                radius = int(w * 0.25)
                forehead_top_center = (int(landmarks[10].x*w), int(landmarks[10].y*h))
                edited_cv = local_warp_effect(edited_cv, forehead_top_center, radius, strength)

            if p_jaw != 0:
                strength = p_jaw * 0.055; radius = int(h * 0.25)
                chin_center = (int(landmarks[152].x * w), int(landmarks[152].y * h)); edited_cv = local_warp_effect(edited_cv, chin_center, radius, strength)

            return Image.fromarray(cv2.cvtColor(edited_cv, cv2.COLOR_BGR2RGB)), "✅ Wajah berhasil diedit."
    except Exception as e:
        traceback.print_exc(); return None, f"❌ Terjadi kesalahan saat edit wajah: {e}"

@app.get("/", include_in_schema=False)
async def redirect_to_web(): return RedirectResponse(url="/web")

@app.post("/preview_face", tags=["API untuk Aplikasi"])
async def preview_face_api(face_image: UploadFile = File(...), p_skin: float = Form(0.0), p_eyes: float = Form(0.0), p_nose_bridge: float = Form(0.0), p_nose_tip: float = Form(0.0), p_lips: float = Form(0.0), p_forehead_width: float = Form(0.0), p_forehead_height: float = Form(0.0), p_jaw: float = Form(0.0)):
    face_pil = Image.open(io.BytesIO(await face_image.read())); result_pil, status_msg = edit_wajah_mediapipe(face_pil, p_skin, p_eyes, p_nose_bridge, p_nose_tip, p_lips, p_forehead_width, p_forehead_height, p_jaw)
    if result_pil is None: raise HTTPException(status_code=500, detail=status_msg)
    byte_io = io.BytesIO(); result_pil.save(byte_io, 'JPEG'); byte_io.seek(0); return StreamingResponse(byte_io, media_type="image/jpeg")

def unified_swap_logic(face_pil, shape_pil, color_pil, edit_kwargs):
    try:
        device = hair_fast_model.args.device; face_tensor = F.resize(F.to_tensor(face_pil), [1024, 1024]).to(device)
        shape_tensor = F.resize(F.to_tensor(shape_pil), [1024, 1024]).to(device); color_tensor = F.resize(F.to_tensor(color_pil), [1024, 1024]).to(device)
        with torch.no_grad(): result = hair_fast_model.swap(face_tensor, shape_tensor, color_tensor, align=True, **edit_kwargs)
        result_tensor = result[0] if isinstance(result, tuple) else result
        return transforms.ToPILImage()(result_tensor.squeeze(0).clamp(0, 1).cpu()), "✅ Swap rambut berhasil."
    except Exception as e: return None, f"❌ Gagal saat swap rambut: {e}"

@app.post("/swap_hair", tags=["API untuk Aplikasi"])
async def swap_hair_api(face_image: UploadFile = File(...), shape_image: UploadFile = File(...), color_image: UploadFile = File(...), gender: float = Form(0.0), age: float = Form(0.0)):
    face_pil = Image.open(io.BytesIO(await face_image.read())).convert("RGB")
    shape_pil = Image.open(io.BytesIO(await shape_image.read())).convert("RGB")
    color_pil = Image.open(io.BytesIO(await color_image.read())).convert("RGB")
    edit_kwargs = {'gender': gender, 'age': age}
    result_pil, status_msg = unified_swap_logic(face_pil, shape_pil, color_pil, edit_kwargs)
    if result_pil is None: raise HTTPException(status_code=500, detail=status_msg)
    byte_io = io.BytesIO(); result_pil.save(byte_io, 'JPEG'); byte_io.seek(0)
    return StreamingResponse(byte_io, media_type="image/jpeg")

gradio_ui = gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.blue))
with gradio_ui:
    gr.Markdown("# ✨ HairFast Studio Pro (The Final Masterpiece) ✨")
    with gr.Tabs() as tabs:
        with gr.Tab(label="🖌️ Editor Wajah Profesional", id=0):
            with gr.Row():
                edit_face_input = gr.Image(label="Gambar Asli", type="pil", sources=['upload', 'webcam', 'clipboard'])
                edit_face_output = gr.Image(label="Hasil Edit", type="pil", interactive=False)
            with gr.Blocks():
                edit_status_text = gr.Textbox(label="Status", interactive=False)
                with gr.Accordion("Kontrol Editor", open=True):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Bentuk Wajah")
                            eyes_slider = gr.Slider(-15, 15, step=1, label="Ukuran Mata", value=0, info="Kiri: kecil, Kanan: besar.")
                            lips_slider = gr.Slider(-15, 15, step=1, label="Ukuran Bibir", value=0, info="Kiri: tipis, Kanan: tebal.")
                            nose_bridge_slider = gr.Slider(-15, 15, step=1, label="Batang Hidung", value=0, info="Kiri: lebar, Kanan: ramping.")
                            nose_tip_slider = gr.Slider(-15, 15, step=1, label="Ujung Hidung", value=0, info="Kiri: lebar, Kanan: ramping.")
                        with gr.Column():
                            gr.Markdown("#### Kontur & Tekstur")
                            forehead_width_slider = gr.Slider(-15, 15, step=1, label="Lebar Dahi", value=0, info="Kiri: ramping, Kanan: lebar.")
                            forehead_height_slider = gr.Slider(-15, 15, step=1, label="Tinggi Dahi", value=0, info="Kiri: pendek, Kanan: tinggi.")
                            jaw_slider = gr.Slider(-15, 15, step=1, label="Bentuk Dagu", value=0, info="Kiri: bulat, Kanan: tirus.")
                            skin_slider = gr.Slider(-15, 15, step=1, label="Tekstur Kulit", value=0, info="Kiri: tajam, Kanan: halus.")
                with gr.Row():
                    edit_run_button = gr.Button("🎨 Edit Wajah Ini", variant="primary", scale=3)
                    send_to_swap_btn = gr.Button("➡️ Kirim ke Swap", variant="secondary", scale=1)
        with gr.Tab(label="✂️ Swap Rambut (Model AI)", id=1):
            with gr.Row():
                swap_face_input = gr.Image(label="Wajah (face)", type="pil", sources=['upload', 'webcam', 'clipboard'])
                swap_shape_input = gr.Image(label="Bentuk Rambut (shape)", type="pil", sources=['upload', 'webcam', 'clipboard'])
                swap_color_input = gr.Image(label="Warna Rambut (color)", type="pil", sources=['upload', 'webcam', 'clipboard'])
            with gr.Accordion("Kontrol Edit Tambahan (Opsional)", open=False):
                swap_gender_slider = gr.Slider(-10.0, 10.0, step=0.1, label="Gender", value=0)
                swap_age_slider = gr.Slider(-10.0, 10.0, step=0.1, label="Usia", value=0)
            swap_run_button = gr.Button("🎨 Proses Swap Rambut", variant="primary")
            swap_output_image = gr.Image(label="Hasil Akhir", type="pil", interactive=False)
            swap_status_text = gr.Textbox(label="Status", interactive=False)

    edit_sliders = [skin_slider, eyes_slider, nose_bridge_slider, nose_tip_slider, lips_slider, forehead_width_slider, forehead_height_slider, jaw_slider]
    edit_run_button.click(fn=edit_wajah_mediapipe, inputs=[edit_face_input] + edit_sliders, outputs=[edit_face_output, edit_status_text])
    def send_image_to_swap(image): return image, gr.Tab.update(selected=1)
    send_to_swap_btn.click(fn=send_image_to_swap, inputs=[edit_face_output], outputs=[swap_face_input, tabs])
    def swap_hair_gradio_wrapper(face_img, shape_img, color_img, gender, age):
        if not all([face_img, shape_img, color_img]): return None, "❌ Semua gambar wajib diunggah."
        edit_kwargs = {'gender': gender, 'age': age}
        return unified_swap_logic(face_img, shape_img, color_img, edit_kwargs)
    swap_run_button.click(fn=swap_hair_gradio_wrapper, inputs=[swap_face_input, swap_shape_input, swap_color_input, swap_gender_slider, swap_age_slider], outputs=[swap_output_image, swap_status_text])

app = gr.mount_gradio_app(app, gradio_ui, path="/web")

if __name__ == '__main__':
    host_ip = '0.0.0.0'; port = 5000
    # Dapatkan IP di sini khusus untuk dicetak.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 1)); true_ip_for_print = s.getsockname()[0]; s.close()
    except Exception: true_ip_for_print = "127.0.0.1" # Fallback
    print("\n" + "="*60); print("🚀 Server Studio Pro (Final Masterpiece + Zeroconf) Siap Dijalankan! 🚀")
    print("✨ Ini adalah versi terbaik kita, dengan semua perbaikan terakhir."); print(f"🌐 Antarmuka Web tersedia di: http://{host_ip}:{port}/web")
    print(f"📱 API untuk aplikasi Flutter/Dart (dengan Zeroconf) tersedia di: http://{true_ip_for_print}:{port}"); print("="*60 + "\n")
    uvicorn.run(app, host=host_ip, port=port, log_level="info", reload=False, workers=1)