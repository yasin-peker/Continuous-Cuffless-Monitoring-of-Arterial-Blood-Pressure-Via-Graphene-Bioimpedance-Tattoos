# Continuous Cuffless Monitoring of Arterial Blood Pressure Via Graphene Bioimpedance Tattoos

This repository contains the data, code, and documentation for the project **"Continuous Cuffless Monitoring of Arterial Blood Pressure Via Graphene Bioimpedance Tattoos."** The research aims to revolutionize blood pressure monitoring using graphene electronic tattoos (GETs) with advanced bioimpedance technology.

### Cuffless Monitoring of Arterial Blood Pressure Via Graphene Bioimpedance Tattoos
![Continuous Cuffless Monitoring of Arterial Blood Pressure Via Graphene Bioimpedance Tattoos](figures/41565_2022_1145_Fig1_HTML.png)

### Illustration of the peripheral arterial BP pulse waveform (red) and correlated arterial volume
![Illustration of the peripheral arterial BP pulse waveform (red) and correlated arterial volume](figures/41565_2022_1145_Fig2_HTML.png)

## Project Overview
Blood pressure monitoring is critical for preventing cardiovascular diseases and managing conditions like hypertension. Traditional methods rely on bulky equipment and are unsuitable for continuous use. This project addresses these limitations by leveraging graphene-based GETs for non-invasive, continuous blood pressure monitoring. 

### Key Features
- **Wearable Technology:** Self-adhesive, low-impedance graphene tattoos placed on the wrist for high-fidelity measurements.
- **Bioimpedance Measurement:** Utilizes arterial bioimpedance to estimate blood pressure metrics such as SBP, DBP, and MAP.
- **Robust Against Motion Artifacts:** GETs are resilient to movement, ensuring consistent data collection during various activities.
- **Machine Learning Integration:** AdaBoost and Support Vector Machines (SVM) for real-time blood pressure prediction with Grade A accuracy.

## Repository Structure
```Graphene-Bioimpedance-Tattoos/
├── README.md
├── data/
│   ├── raw/
│   │   ├── BioZ/
│   │   ├── PPG/
│   │   ├── BP/
│   └── processed/
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── src/
│   ├── preprocessing/
│   │   ├── detrending.py
│   │   ├── filtering.py
│   │   ├── normalization.py
│   └── feature_extraction/
│       ├── statistical_features.py
│       ├── spectral_features.py
│       ├── temporal_features.py
│       ├── PTT_extraction.py
│   └── models/
│       ├── adaboost.py
│       ├── svm.py
├── reports/
│   ├── final_report.pdf
│   ├── figures/
│   │   ├── BioZ_signals.png
│   │   ├── BP_correlations.png
│   │   ├── FFT_spectra.png
│   │   └── model_comparisons.png
├── requirements.txt
└── LICENSE
```

## Experimental Design
Experiments were conducted on seven healthy participants under various conditions:
- **Hand Grip Cold Pressor (HGCP):** Gradual BP elevation through exercise and cold exposure.
- **Cycling:** Stationary cycling for increased BP followed by rest.
- **Valsalva Maneuver:** BP modulation through controlled breathing exercises.
- **Baseline and Rest:** Measurements in a neutral state for control data.

### Dataset Description
The dataset includes:
- **Bioimpedance (BioZ):** Signals recorded at 1250 Hz sampling rate.
- **Blood Pressure (BP):** Continuous measurements using Finapres NOVA at 200 Hz.
- **Photoplethysmography (PPG):** Finger-cuff and fingertip measurements at 75 Hz and 1250 Hz, respectively.

## Methodology
### 1. Data Preprocessing
- **Filtering:** Bandpass and low-pass filters to remove noise and preserve signal integrity.
- **Normalization:** Standardized signals to a range of [0, 1].
- **Envelope Extraction:** Hilbert transform for signal dynamics.
- **FFT Analysis:** Frequency-domain representations for spectral characteristics.

### 2. Feature Extraction
- **Statistical Features:** Mean, standard deviation, skewness, and kurtosis.
- **Temporal Features:** Zero-crossing rate, energy, and signal length.
- **Spectral Features:** Spread, centroid, entropy, and MFCCs.
- **Pulse Transit Time (PTT):** Time delay between signals from paired GETs.
- **Mean Slope:** Reflects dynamic arterial behavior.

### 3. Machine Learning
- **Models Used:**
  - **AdaBoost:** Superior performance with Grade A accuracy for BP prediction.
  - **SVM Regression:** Benchmark model for comparison.
- **Training Metrics:**
  - DBP: Mean error of 0.2 ± 4.6 mmHg.
  - SBP: Mean error of 0.2 ± 5.8 mmHg.
  - MAP: Derived from SBP and DBP values.

## Results
- **High Accuracy:** GETs outperformed traditional Ag electrodes with comparable Grade A accuracy.
- **Long-Term Stability:** Demonstrated no degradation in signal quality during extended use, even under challenging conditions like sweating and movement.
- **Versatility:** Capable of monitoring BP during diverse activities, including workouts and rest.

###  Graphene Z-BP measurement results from the HGCP routine
![ Graphene Z-BP measurement results from the HGCP routine](figures/41565_2022_1145_Fig3_HTML.png)

### Graphene Z-BP model training and performance evaluation
![Graphene Z-BP model training and performance evaluation](figures/41565_2022_1145_Fig4_HTML.png)

### Comparison of BioZ Signals
![Comparison of BioZ Signals](./figures/bioz1_bioz2_pairwise.png)

### Raw and Filtered BioZ Signal
![Raw and Filtered BioZ Signal](./figures/raw_and_filtered_bioz1-2-3-4_signals.png)

### Raw and Enveloped BioZ Signal
![raw_and_enveloped_bioz1](./figures/raw_and_enveloped_bioz1.png)

### Fourier Transform of BioZ Signal
![Fourier Transform of BioZ Signal](./figures/FFT-Bioz1.png)

### Adaboost Model Results
![Adaboost Model Results](./figures/ada2.png)

### Support Regression Model Results
![Support Regression Model Results](./figures/srv.png)

## Future Work
- Integration with smartwatches for seamless daily monitoring.
- Development of wireless data transmission and storage solutions.
- Expansion to larger datasets and clinical trials.

## Usage
The main file to run is called BP_BioZ_v0.py

## Installation of Outside Packages

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages.

```bash
pip install numpy 
pip install pandas
pip install hjson
pip install scikit_learn
pip install tensorflow
pip install xgboost
```

## Requirements

Extracted through [pypreqs](https://pypi.org/project/pipreqs/)

```python
xgboost==1.4.2
numpy==1.20.3
pandas==1.2.4
hjson==3.0.2
tensorflow==2.3.0
scikit_learn==0.24.2
```

## Configuration
There are several parameters to be configured before running the BP estimation from Bio-Z. If you want to regenerate the original results you can keep the training-related parameters untouched.

To configure the parameters run:
```bash
python configure.py
```
This will walk you through setting the parameters. To use default values for each parameter, showed in the brackets, press "Enter" without entering any character.

example:
```bash
Use average features instead of beat-to-beat? (true/false)
features_mean [true]:
```

## Usage
To run BP estimation from Bio-Z:
```bash
python BP_BioZ_v0.py
```
The results including the BP estimation for each subject and the overal error metrics will be stored in folder: _'./Data/predictions/'_


## Acknowledgments

This project was inspired by the groundbreaking research conducted by Dmitry Kireev et al., titled **"Continuous cuffless monitoring of arterial blood pressure via graphene bioimpedance tattoos"**, published in *Nature Nanotechnology* (2022). The study introduced the potential of graphene electronic tattoos (GETs) as a platform for non-invasive and continuous health monitoring, which provided the foundation for our exploration of bioimpedance-based blood pressure monitoring.

You can access the original research paper here: [Continuous cuffless monitoring of arterial blood pressure via graphene bioimpedance tattoos](https://www.nature.com/articles/s41565-022-01145-w).

The dataset used in this project was obtained from [PhysioNet](https://physionet.org/), specifically the **"Continuous Cuffless Monitoring of Arterial Blood Pressure via Graphene Bioimpedance Tattoos"**, version 1.0.0. This dataset includes synchronized bioimpedance, photoplethysmography, and blood pressure recordings, which were instrumental in training and validating our machine learning models.  
You can access the dataset here: [Graphene Bioimpedance Arterial Blood Pressure Dataset](https://physionet.org/content/bp-graphene-bioimpedance/1.0.0/).

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
