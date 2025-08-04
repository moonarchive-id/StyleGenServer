import os
import io
import asyncio
import socket
import argparse 
import torch 
import torchvision.transforms.functional as F 
import contextlib 

from hair_swap import HairFast, get_parser 
from zeroconf import ServiceInfo 
from zeroconf.asyncio import AsyncZeroconf 

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles 
from starlette.templating import Jinja2Templates 
import uvicorn 
from PIL import Image

app = FastAPI(
    title="HairFast AI API",
    description="API for HairFast AI model to swap hairstyles and apply attributes.",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

hair_fast_model: HairFast = None 
zeroconf_instance_global: AsyncZeroconf = None 

app.state.true_ip = "127.0.0.1" 

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global hair_fast_model
    global zeroconf_instance_global

    print("Memulai Lifespan Event: Startup...", flush=True)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        true_ip = s.getsockname()[0]
    except Exception:
        true_ip = '127.0.0.1' 
    finally:
        s.close()
    
    app.state.true_ip = true_ip 
    print(f"IP Asli Server Terdeteksi (via Lifespan Startup): {app.state.true_ip}", flush=True)

    model_parser = get_parser()
    model_args = model_parser.parse_args(args=[
        '--device', 'cuda' if torch.cuda.is_available() else 'cpu',
        '--ckpt', 'pretrained_models/StyleGAN/ffhq.pt', 
        '--blending_checkpoint', 'pretrained_models/Blending/checkpoint.pth',
        '--rotate_checkpoint', 'pretrained_models/Rotate/rotate_best.pth',
        '--pp_checkpoint', 'pretrained_models/PostProcess/pp_model.pth',
        '--save_all_dir', 'output', 
        '--size', '1024',
        '--latent', '512',
        '--n_mlp', '8'
    ])
    hair_fast_model = HairFast(model_args)
    print("Model HairFast AI berhasil dimuat!", flush=True)

    port = 5000 
    service_info = ServiceInfo(
        "_http._tcp.local.",
        "My HairFast API._http._tcp.local.", 
        addresses=[socket.inet_aton(true_ip)],
        port=port,
        properties={'path': '/'},
        server=f"{socket.gethostname()}.local."
    )
    zeroconf_instance_global = AsyncZeroconf() 
    print(f"Mengumumkan layanan 'My HairFast API' di jaringan lokal...", flush=True) 
    await zeroconf_instance_global.async_register_service(service_info)

    yield 

    print("Memulai Lifespan Event: Shutdown...", flush=True)
    if zeroconf_instance_global:
        print("Membatalkan pengumuman layanan Zeroconf.", flush=True)
        await zeroconf_instance_global.async_close()
        print("Layanan Zeroconf telah dihentikan.", flush=True)

app.router.lifespan_context = lifespan 

@app.get("/", response_class=HTMLResponse)
async def get_test_page(request: Request): 
    server_ip_to_display = app.state.true_ip 
    server_port = 5000 
    return templates.TemplateResponse("index.html", {"request": request, "server_ip": server_ip_to_display, "server_port": server_port})

@app.post("/preview_face") 
async def preview_face_api(
    face_image: UploadFile = File(...), 
    gen: float = Form(0.0), 
    age: float = Form(0.0),
    hair: float = Form(0.0), 
    face: float = Form(0.0), 
    lighting: float = Form(0.0), 
):
    if hair_fast_model is None:
        raise HTTPException(status_code=503, detail="Model AI belum siap. Harap tunggu sebentar.")

    try:
        print(f"Menerima request preview wajah. Input slider: Gen={gen}, Hair={hair}, Face={face}, Age={age}, Lighting={lighting}", flush=True)

        face_pil = Image.open(io.BytesIO(await face_image.read())).convert("RGB")
        face_tensor = F.resize(F.to_tensor(face_pil), [1024, 1024]).to(hair_fast_model.args.device)

        images_to_name = {face_tensor: ['face']}
        name_to_embed = hair_fast_model.embed.embedding_images(images_to_name)
        latent = name_to_embed['face']

        latent_tensor_for_edit = None
        if isinstance(latent, dict) and 'W' in latent:
            latent_tensor_for_edit = latent['W'].detach().clone()
        elif isinstance(latent, torch.Tensor):
            latent_tensor_for_edit = latent.detach().clone()
        else:
            raise TypeError(f"Tipe latent tidak dikenali untuk preview: {type(latent)}")

        edited_latent_tensor = hair_fast_model.edit_latent(
            latent_tensor_for_edit, 
            smile=0.0, age=age, gender=gen, pose=0.0, glasses=0.0
        )

        with torch.no_grad():
            final_image_tensor, _ = hair_fast_model.gan([edited_latent_tensor], input_is_latent=True, randomize_noise=False)
            
        final_image_tensor = (final_image_tensor + 1.0) / 2.0 
        final_image_pil = F.to_pil_image(final_image_tensor[0].clamp(0, 1).cpu())

        byte_io = io.BytesIO()
        final_image_pil.save(byte_io, 'JPEG')
        byte_io.seek(0)

        print("Preview wajah berhasil. Mengirim respons...", flush=True)
        return StreamingResponse(byte_io, media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=face_preview.jpg"})

    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Gagal melakukan preview wajah: {e}")

@app.post("/swap_hair") 
async def swap_hair_api(
    face_image: UploadFile = File(...), 
    shape_image: UploadFile = File(...),
    color_image: UploadFile = File(...),
    gen: float = Form(0.0), 
    age: float = Form(0.0), 
    hair: float = Form(0.0), 
    face: float = Form(0.0), 
    lighting: float = Form(0.0), 
):
    if hair_fast_model is None:
        raise HTTPException(status_code=503, detail="Model AI belum siap. Harap tunggu sebentar.")

    try:
        print(f"Menerima request swap rambut. Memproses input...", flush=True)

        face_pil = Image.open(io.BytesIO(await face_image.read())).convert("RGB")
        shape_pil = Image.open(io.BytesIO(await shape_image.read())).convert("RGB")
        color_pil = Image.open(io.BytesIO(await color_image.read())).convert("RGB")

        face_tensor = F.resize(F.to_tensor(face_pil), [1024, 1024]).to(hair_fast_model.args.device)
        shape_tensor = F.resize(F.to_tensor(shape_pil), [1024, 1024]).to(hair_fast_model.args.device)
        color_tensor = F.resize(F.to_tensor(color_pil), [1024, 1024]).to(hair_fast_model.args.device)
        
        edit_kwargs = {
            'smile': 0.0,    
            'age': 0.0,      
            'gender': 0.0,   
            'pose': 0.0,     
            'glasses': 0.0   
        }
        
        with torch.no_grad(): 
            result = hair_fast_model.swap(
                face_tensor,
                shape_tensor,
                color_tensor,
                align=True, 
                **edit_kwargs
            )

        if isinstance(result, tuple):
            final_image_tensor = result[0]
        else:
            final_image_tensor = result

        final_image_tensor = (final_image_tensor + 1.0) / 2.0 
        final_image_pil = F.to_pil_image(final_image_tensor.clamp(0, 1).cpu())

        byte_io = io.BytesIO()
        final_image_pil.save(byte_io, 'JPEG')
        byte_io.seek(0)

        print("Swap rambut berhasil. Mengirim respons...", flush=True)
        return StreamingResponse(byte_io, media_type="image/jpeg", headers={"Content-Disposition": "attachment; filename=hair_swap_result.jpg"})

    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Gagal melakukan swap rambut: {e}")

if __name__ == '__main__':
    host_ip = '0.0.0.0'
    port = 5000
    
    zeroconf_instance_global: AsyncZeroconf = None 

    try:
        uvicorn.run(app, host=host_ip, port=port, log_level="info", reload=False, workers=1) 

    except KeyboardInterrupt:
        print("\nMenjalankan proses shutdown server...", flush=True)
    finally:
        pass 