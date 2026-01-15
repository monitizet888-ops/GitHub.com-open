import streamlit as st
from TTS.api import TTS
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageClip, AudioFileClip
import torch

st.set_page_config(page_title="AI Text to Video", layout="centered")
st.title("🎬 AI Text to Video Generator (FREE)")

text = st.text_area("✍️ Apna text likho")

if st.button("Generate Video"):
    if text.strip() == "":
        st.warning("Pehle text likho")
    else:
        with st.spinner("AI video bana raha hai..."):

            # Text to Speech
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            tts.tts_to_file(text=text, file_path="voice.wav")

            # Text to Image
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32
            )
            pipe = pipe.to("cpu")
            image = pipe(text).images[0]
            image.save("image.png")

            # Image + Audio → Video
            clip = ImageClip("image.png").set_duration(8)
            audio = AudioFileClip("voice.wav")
            video = clip.set_audio(audio)
            video.write_videofile("final.mp4", fps=24)

            st.success("✅ Video ready!")
            st.video("final.mp4")
