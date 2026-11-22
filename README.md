# Binocular

**A C++ research prototype for gastric histopathology image classification and exploratory model interpretation.**

Binocular is an interactive desktop application built with **C++**, **OpenCV**, and **AlphaCore**, a custom C++ neural network library. The application loads a trained neural network, accepts an image of gastric tissue, and displays a **Normal**, **Abnormal**, or **Inconclusive** result based on the model's output. It also includes an experimental quadrant-replacement analysis to explore how changes in different image regions affect predictions.

Binocular was developed as an independent project using imagery from the **GasHisSDB** gastric histopathology research dataset. It explores whether changes to selected image regions affect the output of a trained classifier.

> **Research prototype only.** Binocular is not a clinically validated medical device and must not be used for medical diagnosis, screening, or treatment decisions. Model accuracy and generalization have not been established by the source files in this repository.

## Research background

Binocular was informed by the GasHisSDB research paper:

> Weiming Hu et al., **“GasHisSDB: A New Gastric Histopathology Image Dataset for Computer Aided Diagnosis of Gastric Cancer”** (2021). [Read the paper on arXiv](https://arxiv.org/abs/2106.02473) · [PDF](https://arxiv.org/pdf/2106.02473)

The paper introduces a dataset of gastric histopathology sub-images labeled **normal** or **abnormal**, along with baseline classification experiments. Binocular uses this research dataset as the basis for a separate C++ classification prototype and adds an exploratory **quadrant-based image perturbation** technique: changing portions of a reference image and observing the effect on the model’s output. This is an investigation of **model sensitivity**, not evidence that the highlighted regions are clinically diagnostic. The work does not claim to reproduce the paper's published performance results.

## Features

- **Interactive image interface:** OpenCV-powered desktop UI with image selection, preview, and classification.
- **Custom neural network integration:** Uses AlphaCore for inference rather than a high-level Python ML framework.
- **Grayscale preprocessing:** Converts image pixels to a flattened vector of values normalized to `[0, 1]`.
- **Binary classification:** Maps the model's single output to normal/abnormal labels using a threshold of `0.5`.
- **Exploratory interpretability:** Replaces quadrants of a normal reference image with quadrants from the uploaded image, then compares model outputs.
- **Training implementation:** Includes a historical training loop in `main.cpp`, currently commented out, using normal and abnormal GasHisSDB image samples.

## How it works

1. The application loads saved model parameters using **AlphaCore**.
2. The user selects an image through the OpenCV interface.
3. The image is converted to grayscale and its pixel intensities are normalized to floats in `[0, 1]`.
4. AlphaCore performs a forward pass through the network.
5. The UI labels an output **below `0.5` as Abnormal**, **above `0.5` as Normal**, and **exactly `0.5` as Inconclusive**.
6. The application performs an experimental quadrant-replacement comparison against a fixed normal reference image and displays differences in model output.

### Neural network architecture

The loaded feedforward model is configured with the following layer sizes:

```text
6,400 → 2,032 → 1,012 → 1,012 → 2,032 → 1
```

- **Input:** 6,400 normalized grayscale pixel values, corresponding to an **80 × 80** image.
- **Hidden layers:** Four hidden layers.
- **Output:** One sigmoid-activated value used for binary classification.
- **Activations:** Sigmoid functions are configured for each non-input layer.
- **Additional architecture:** An elementwise skip connection is configured between layers `1` and `4` using AlphaCore's layer-indexing convention.

The implementation is written in C++ and uses AlphaCore's **Eigen-backed CPU** configuration. The code initializes a learning rate of `0.00005`, but the normal application path loads an existing model rather than retraining it.

## Quadrant-replacement experiment

To explore which image regions influence a prediction, Binocular uses a reference image from the normal class and performs the following comparison:

1. Evaluate the original normal reference image.
2. Replace a quadrant of the reference image with the corresponding quadrant from the uploaded image.
3. Re-evaluate the modified image using the same network.
4. Display the difference between the original reference score and the modified-image score.

**Important implementation limitation:** The current code modifies the same reference image across successive iterations. Consequently, the displayed values represent **cumulative quadrant replacements**, not four independent one-quadrant-at-a-time sensitivity scores. This is an exploratory perturbation technique, not a validated explanation of clinical features or the model's internal reasoning.

## Technology

| Component | Technology |
| --- | --- |
| Application | C++ |
| Interface and image processing | OpenCV |
| Neural network | AlphaCore (custom C++ library) |
| CPU matrix operations | Eigen (via AlphaCore) |
| Research dataset | GasHisSDB |

**AlphaCore** is a separate companion project that provides the feedforward neural network implementation, saved-model loading, and inference functions used by Binocular.

## Repository files

```text
Binocular/
├── main.cpp             # UI, model loading, inference, and experimental analysis
├── Tools.h              # Image processing and UI helper functions
├── README.md
├── AlphaCore.h          # Required dependency; obtain from AlphaCore repository
└── Resources/           # Required at runtime; include files when permitted
    ├── Sigmoid1.7/      # Trained-model parameter files
    ├── 1.png
    ├── 3.png
    ├── 4.png
    ├── tc.png
    ├── CTRI.png
    ├── uploadScreen.png
    └── Normal-42760.png
```

*The tree shows the files expected by the current source code, not a claim that every file is included in this repository.*

## Prebuilt application

The original packaged executable and supporting files were shared as **Binocular v1.0.1**:

[Download Binocular v1.0.1 (Dropbox ZIP)](https://www.dropbox.com/scl/fi/apoio145t7gazbbbvc84z/Binocular-v1.0.1.zip?rlkey=xxmhaywhqntsyjak52ls4zr0w&st=qyvu9zdu&dl=0)

**Compatibility note:** The original release has limited support across hardware architectures and operating systems. The archive's current availability and compatibility have not been independently verified. If the executable does not run on your system, see the source/build notes below.

## Build and run

### Prerequisites

- A C++ compiler compatible with the included AlphaCore header
- **OpenCV** development libraries (including the HighGUI modules used for windows and mouse events)
- **Eigen** headers
- **AlphaCore.h** from the separate AlphaCore project
- The application UI assets, normal reference image, and trained-model parameter files listed above

The original project uses a machine-specific setup; a portable build script is **not yet provided**. You will need to configure your compiler's include paths for AlphaCore, Eigen, and OpenCV and link against OpenCV for your platform.

To launch the original application, the executable also needs access to its `Resources` directory. Note that the existing code mixes paths relative to the executable with paths relative to the current working directory, so you may need to adjust resource paths before running it on another machine.

**Image requirement:** Inference expects an image that becomes **80 × 80 grayscale pixels** (6,400 inputs). The UI currently does not automatically resize the inference input to this size, so use a correctly sized input image until input validation and resizing are added.

**Model requirement:** The application loads saved parameters from `Resources/Sigmoid1.7`; without those files, it cannot reproduce the demonstrated inference behavior. Only redistribute model weights and dataset examples when their licenses and permissions allow it.

## Training notes

The source contains an earlier training routine, currently commented out, which:

- Draws samples from normal and abnormal folders of the GasHisSDB dataset.
- Converts grayscale images into normalized vectors.
- Uses `1` as the target for normal samples and `0` for abnormal samples.
- Calls AlphaCore's backpropagation routine to update the network.

The training block currently references absolute paths from the original development machine and is **not a ready-to-run training pipeline**. The provided source also does not establish held-out test accuracy, sensitivity, specificity, or external validation results; none are claimed here.

## Known limitations and possible improvements

- Make model and UI resource paths portable across machines.
- Add explicit image-size validation/resizing and error handling for missing files.
- Make each quadrant analysis independent by starting from a fresh copy of the reference image.
- Revisit OpenCV mouse callback state management to ensure callback data remains valid.
- Provide build configuration, sample inputs, and a reproducible evaluation pipeline.
- Evaluate classification performance using a documented held-out dataset and appropriate metrics.

## Research attribution, license, and data

The **GasHisSDB** dataset and its associated publication are the work of Hu et al. (2021); they are not original to Binocular. The quadrant-based perturbation experiment and C++ application described above are part of this independent project.

Binocular's source code, its trained weights, its UI assets, the GasHisSDB dataset, and AlphaCore may have different licensing or redistribution requirements. Check permissions before publishing each component. Add a `LICENSE` file that accurately reflects the terms you intend to apply to **your own code**.

## Author

Developed by **Ethan Henry** (2024).
