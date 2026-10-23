import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

st.set_page_config(page_title="AeroGen AI", page_icon="✈️", layout="wide")
st.title("✈️ AeroGen AI")
st.caption("AI-powered aviation data and image generation laboratory")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architectures
class AircraftAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z)


@st.cache_resource
def load_autoencoder():
    model = AircraftAutoencoder().to(device)
    model.load_state_dict(torch.load(
        "aerogen_aircraft_autoencoder.pth",
        map_location=device,
        weights_only=True
    ))
    model.eval()
    return model


@st.cache_resource
def load_generator():
    model = Generator().to(device)
    model.load_state_dict(torch.load(
        "aerogen_generator_v2.pth",
        map_location=device,
        weights_only=True
    ))
    model.eval()
    return model


module = st.sidebar.selectbox("Choose a module", [
    "Flight Data Explorer",
    "Aircraft Image Lab",
    "GAN Image Generator",
    "Passenger Notification Generator",
    "Aviation Advertisement Generator"
])

if module == "Flight Data Explorer":
    st.header("Flight Delay Distribution")
    np.random.seed(42)
    normal = np.clip(np.random.normal(20, 15, 1000), 0, None)
    uniform = np.random.uniform(0, 60, 1000)

    fig, ax = plt.subplots()
    ax.hist(normal, bins=30, alpha=0.6, label="Normal")
    ax.hist(uniform, bins=30, alpha=0.6, label="Uniform")
    ax.set_xlabel("Delay (minutes)")
    ax.set_ylabel("Flights")
    ax.legend()
    st.pyplot(fig)
    st.metric("Normal distribution mean", f"{normal.mean():.2f} min")
    st.metric("Uniform distribution mean", f"{uniform.mean():.2f} min")

elif module == "Aircraft Image Lab":
    st.header("Aircraft Image Reconstruction")
    uploaded = st.file_uploader("Upload an aircraft image", type=["png", "jpg", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        tensor = transform(image).unsqueeze(0).to(device)

        try:
            model = load_autoencoder()
            with torch.no_grad():
                reconstructed = model(tensor).squeeze(0).cpu()

            left, right = st.columns(2)
            left.image(image, caption="Original image", use_container_width=True)
            right.image(
                reconstructed.permute(1, 2, 0).clamp(0, 1).numpy(),
                caption="Reconstructed image",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not load the Autoencoder: {e}")
    else:
        st.info("Upload an aircraft image to reconstruct it.")

elif module == "GAN Image Generator":
    st.header("GAN Aircraft Image Generator")
    count = st.slider("Number of images", 1, 8, 4)

    if st.button("Generate aircraft images"):
        try:
            model = load_generator()
            noise = torch.randn(count, 100, 1, 1, device=device)

            with torch.no_grad():
                images = model(noise).cpu()

            cols = st.columns(min(4, count))
            for i, generated in enumerate(images):
                image = ((generated.permute(1, 2, 0) + 1) / 2).clamp(0, 1)
                cols[i % len(cols)].image(
                    image.numpy(),
                    caption=f"Generated image {i+1}",
                    use_container_width=True
                )
            st.caption("The GAN was trained for a limited number of epochs, so images may be abstract.")
        except Exception as e:
            st.error(f"Could not load the GAN: {e}")

elif module == "Passenger Notification Generator":
    st.header("GPT-2 Passenger Notification Generator")
    flight = st.text_input("Flight number", "AG101")
    status = st.selectbox(
        "Flight status", ["Delayed", "Cancelled", "Boarding", "On time"]
    )
    minutes = st.number_input("Delay in minutes", min_value=0, value=30)

    if st.button("Generate notification"):
        prompts = {
            "Delayed": f"Write a polite airline notification for flight {flight}, delayed by {minutes} minutes. Do not add any other flight details.",
            "Cancelled": f"Write a polite cancellation notification for flight {flight}. Do not add any other flight details.",
            "Boarding": f"Write a short boarding notification for flight {flight}. Do not add any other flight details.",
            "On time": f"Write a short on-time notification for flight {flight}. Do not add any other flight details."
        }

        try:
            from transformers import pipeline

            @st.cache_resource
            def load_gpt2():
                import torch
                return pipeline(
                    "text-generation",
                    model="gpt2",
                    device=0 if torch.cuda.is_available() else -1
                )

            generator = load_gpt2()
            result = generator(
                prompts[status],
                max_new_tokens=35,
                do_sample=True,
                temperature=0.6,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=generator.tokenizer.eos_token_id
            )

            # Use only the user-provided facts in the final message.
            if status == "Delayed":
                message = f"Flight {flight} is delayed by approximately {minutes} minutes. We apologise for the inconvenience. Please check official airline updates for further information."
            elif status == "Cancelled":
                message = f"Flight {flight} has been cancelled. We apologise for the inconvenience. Please contact the airline through its official channels for assistance."
            elif status == "Boarding":
                message = f"Flight {flight} is now boarding. Please check official airport displays for your gate information."
            else:
                message = f"Flight {flight} is currently reported as on time. Please check official airline updates for the latest information."

            st.success(message)
            with st.expander("View GPT-2 generated draft"):
                st.write(result[0]["generated_text"])
            st.caption("The final notification uses only the flight details you entered. Verify live flight status with the airline.")
        except Exception as e:
            st.error(f"GPT-2 error: {e}")

elif module == "Aviation Advertisement Generator":
    st.header("AI-Generated Aviation Advertisement")
    if st.button("Show generated advertisement"):
        try:
            st.image(
                "aerogen_aviation_advertisement.png",
                caption="Generated aviation advertisement",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Could not display advertisement: {e}")

    if not __import__("os").path.exists("aerogen_aviation_advertisement.png"):
        st.warning("The advertisement image file is missing from the current runtime.")
