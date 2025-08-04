import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as transforms
from hair_swap import HairFast, get_parser

model_args = get_parser().parse_args(args=[])
model_args.device = 'cpu'
hair_fast = HairFast(model_args)

resize_transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])

def edit_face_only(face_img, p_gen, p_hair, p_face, p_age, p_lighting):
    if face_img is None:
        return None, "❌ Gambar Wajah wajib diunggah."
    
    try:
        face_tensor = resize_transform(face_img.convert("RGB"))
        images_to_name = {face_tensor: ['face']}
        name_to_embed = hair_fast.embed.embedding_images(images_to_name)
        latent = name_to_embed['face']

        if isinstance(latent, dict):
            latent['W'] = latent['W'].detach().clone()
        elif isinstance(latent, torch.Tensor):
            latent = latent.detach().clone()

        edited_latent = hair_fast.edit_latent(
            latent, smile=p_gen, age=p_hair, gender=p_face, pose=p_age, glasses=p_lighting
        )
        
        if isinstance(edited_latent, dict):
            latent_tensor = edited_latent['W']
        else:
            latent_tensor = edited_latent
        
        result_tensor, _ = hair_fast.gan([latent_tensor], input_is_latent=True, randomize_noise=False)
        result_tensor = (result_tensor + 1.0) / 2.0
        result_pil = transforms.ToPILImage()(result_tensor[0].clamp(0, 1).cpu())
        
        return result_pil, "✅ Wajah berhasil diedit."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Gagal saat edit wajah: {e}"

def swap_hair_only(face_img, shape_img, color_img):
    if face_img is None or shape_img is None or color_img is None:
        return None, "❌ Semua gambar (Wajah, Bentuk, Warna) wajib diunggah."
        
    try:
        face_tensor = resize_transform(face_img.convert("RGB"))
        shape_tensor = resize_transform(shape_img.convert("RGB"))
        color_tensor = resize_transform(color_img.convert("RGB"))

        with torch.no_grad():
            result = hair_fast.swap(
                face_tensor, shape_tensor, color_tensor, align=True 
            )

        result_tensor = result[0] if isinstance(result, tuple) else result
        result_pil = transforms.ToPILImage()(result_tensor.clamp(0, 1).cpu())
        
        return result_pil, "✅ Swap rambut berhasil."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Gagal saat swap rambut: {e}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✨ HairFast Studio ✨")
    gr.Markdown("Pilih mode di bawah: **Edit Wajah** untuk mengubah fitur, atau **Swap Rambut** untuk mengganti gaya rambut.")

    shared_face_for_swap = gr.State()

    def set_face_for_swap(img_from_edit):
        return img_from_edit, "✅ Wajah hasil edit siap digunakan di tab Swap Rambut!"

    with gr.Tabs() as tabs:
        with gr.TabItem("🖌️ Edit Wajah", id=0):
            gr.Markdown("## Mode Edit Wajah")
            gr.Markdown("Upload wajah lalu mainkan slider untuk mengubah fitur wajah. Klik **'Gunakan Hasil Ini'** untuk memakainya di mode Swap Rambut.")
            with gr.Row():
                with gr.Column(scale=1):
                    edit_face_input = gr.Image(label="Upload Wajah", type="pil")
                    slider_min = -30.0
                    slider_max = 30.0
                    gen_slider = gr.Slider(slider_min, slider_max, step=0.1, label="Gen : maskuline-feminime", value=0.0)
                    hair_slider = gr.Slider(slider_min, slider_max, step=0.1, label="Hair : less-more", value=0.0)
                    face_slider = gr.Slider(slider_min, slider_max, step=0.1, label="Face : wide-slim", value=0.0)
                    age_slider = gr.Slider(slider_min, slider_max, step=0.1, label="Age : young-old", value=0.0)
                    lighting_slider = gr.Slider(slider_min, slider_max, step=0.1, label="Lighting : soft-hard", value=0.0)
                    
                    with gr.Row():
                        edit_run_button = gr.Button("🎨 Edit Wajah Ini")
                        use_edit_result_button = gr.Button("🚀 Gunakan Hasil Ini untuk Swap")

                with gr.Column(scale=1):
                    edit_face_output = gr.Image(label="Hasil Edit Wajah", type="pil", interactive=False)
                    edit_status_text = gr.Textbox(label="Status", interactive=False)

        with gr.TabItem("✂️ Swap Rambut", id=1):
            gr.Markdown("## Mode Swap Rambut")
            gr.Markdown("Input Wajah di sini akan otomatis terisi jika kamu menekan 'Gunakan Hasil Ini' dari tab Edit Wajah.")
            with gr.Row():
                swap_face_input = gr.Image(label="Wajah (face)", type="pil")
                swap_shape_input = gr.Image(label="Bentuk Rambut (shape)", type="pil")
                swap_color_input = gr.Image(label="Warna Rambut (color)", type="pil")
            
            swap_run_button = gr.Button("🎨 Proses Swap Rambut")
            swap_output_image = gr.Image(label="Hasil Akhir", type="pil", interactive=False)
            swap_status_text = gr.Textbox(label="Status", interactive=False)

    edit_sliders = [gen_slider, hair_slider, face_slider, age_slider, lighting_slider]
    edit_run_button.click(
        fn=edit_face_only,
        inputs=[edit_face_input] + edit_sliders,
        outputs=[edit_face_output, edit_status_text]
    )

    use_edit_result_button.click(
        fn=set_face_for_swap,
        inputs=[edit_face_output],
        outputs=[shared_face_for_swap, edit_status_text]
    ).then(
        lambda: gr.Tabs.update(selected=1), None, tabs
    ).then(
        lambda x: x, shared_face_for_swap, swap_face_input
    )

    swap_run_button.click(
        fn=swap_hair_only,
        inputs=[swap_face_input, swap_shape_input, swap_color_input],
        outputs=[swap_output_image, swap_status_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)