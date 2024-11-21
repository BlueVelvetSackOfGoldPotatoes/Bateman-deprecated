# config.py

import os

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
OPENAI_MODEL = "gpt-4o"  # Example: "gpt-4o-mini-2024-07-18"

# Hugging Face Configuration
HUGGINGFACE_MODEL = "gpt2"  # Example: "gpt2"

# SERPAPI Configuration
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "your-serpapi-api-key")

# File Paths
OUTPUT_FOLDER = "../data/pdfs"
CSV_PATH = "../data/metadata.csv"
JSON_PATH = "../data/metadata.json"

# LLM Configuration
LLM_MODELS = {
    "oa": "gpt-4o-mini-2024-07-18",
    "hf": "gpt2",
    # Add more models as needed
}

# Product of Interest
PRODUCT_DESCRIPTION = """
Quantum Sensing Principles
Diamond Magnetometry

Quantum Nuova achieves quantum sensing by using the crystal lattice defect in the diamond nanoparticles. The diamond is a crystal lattice of carbon atoms, and the defect is a carbon atom being replaced with a Nitrogen atom and an adjacent vacancy (NV center) (A). These nanodiamonds have the ability to fluoresce forever without any bleaching. The fluorescence behavior is dependent on the magnetic field. Due to this relation, it is possible to translate the magnetic field to an optical signal, which is dramatically easier to detect. The sensitivity of the nanodiamonds to the magnetic fields is so high that it is even possible to detect the magnetic field of a single electron. The nanodiamonds with these properties are called fluorescent nanodiamonds (FNDs).

Using a strong green laser pulse, it is possible to excite the NV centers of the fluorescent nanodiamonds from the ground state into an excited state (B). The laser is switched off for a period of time (the dark time), and the NV centers stochastically decay back into their ground state (C). Upon decaying, a photon is released. This fluorescence is an optical signal that can be detected. The rate of the fluorescence intensity decay during the relaxation from the excited state back to the ground state depends on the magnetic noise (random fluctuations in a magnetic field). It means that if there are more magnetic field sources (for example free radicals) then the fluorescence intensity decays at higher rates. T1 relaxometry is one of the magnetometry methods using this principle and measures the fluorescence intensity at the beginning of each laser pulse after different dark times. These intensities follow a decreasing exponential governed by a specific time constant called T1 (D). 


Free Radical Concentration (nanomolar)
Magnetic noise in cells can arise from unpaired electrons of free radicals. The rate of the fluorescence emitted by fluorescent nanodiamonds placed in biological samples is a direct measurement of the magnetic noise, and therefore the free radical concentration, that is present in the surroundings of the NV center. 

Fluorescent nanodiamonds are very biocompatible and can be placed into a variety of biological samples, including cells, tissues and live organisms. Furthermore, their unique quantum properties, namely the fluorescence they emit when excited by green laser light, render them excellent candidates for quantum sensing applications. The colorful specs on the left image show the fluorescent nanodiamonds inside of cells.

LEARN MORE
Confocal Microscope
The laser-scanning confocal microscope employs a laser light source which navigates through an arranged pinhole and lens system, focusing sharply on targeted specimen points. The microscope scans the specimen in X and Y directions, constructing an image from the focal plane. After enhancing sharpness and stitching the 2D slices together, the result is a high-resolution, 3D image.

The confocal microscope in the Quantum Nuova is used to image the biological sample and localize the fluorescent nanodiamonds used for quantum sensing.
"""
