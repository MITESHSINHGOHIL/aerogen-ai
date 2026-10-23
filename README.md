# AeroGen AI — Aviation Intelligence Lab

## Project Overview
AeroGen AI is an AI mini-project combining synthetic data generation,
image reconstruction, GAN-based image generation, text generation,
text-to-image generation, and a Streamlit dashboard.

## Features

1. Synthetic Flight Data
   - Generates synthetic flight-delay data.
   - Compares normal and uniform distributions.

2. Aircraft Image Reconstruction
   - Uses a trained convolutional Autoencoder.
   - Compares an uploaded image with its reconstruction.

3. GAN Image Generation
   - Uses a trained DCGAN Generator to produce synthetic images.

4. Passenger Notification Generator
   - Uses GPT-2 to generate a draft.
   - Displays a controlled notification based on user inputs.

5. Aviation Advertisement
   - Displays an advertisement generated using a text-to-image model.

6. Streamlit Dashboard
   - Integrates the five features into one interface.

## Main Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Diffusers
- Streamlit
- NumPy and Matplotlib
- Pillow and Torchvision

## Model Files
- aerogen_aircraft_autoencoder.pth
- aerogen_generator.pth
- aerogen_discriminator.pth
- aerogen_aviation_advertisement.png

## Running the Dashboard

Install the required dependencies in your Python environment:

pip install streamlit torch torchvision transformers diffusers accelerate safetensors numpy matplotlib pillow

Place app.py and the required model/image files in the same directory, then run:

streamlit run app.py

## Notes
- The Autoencoder and GAN were trained for a limited number of epochs.
- Generated GAN images may be blurry or abstract.
- GPT-2 is a general-purpose language model. Its draft should be reviewed.
- Passenger notifications use controlled templates for the final displayed message.
- Flight data is synthetic and should not be treated as real operational data.
