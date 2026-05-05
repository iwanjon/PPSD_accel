\# PPSD Accel



⚠️ \*\*Disclaimer:\*\* \*This package is for personal usage only. The custom accelerometer noise models included in this package are unverified.\*



\*\*PPSD Accel\*\* is a custom Python module built on top of \[ObsPy](https://github.com/obspy/obspy). It extends ObsPy's standard PPSD (Probabilistic Power Spectral Density) capabilities by adding specialized handling for \*\*accelerometer\*\* data using custom Low Noise and High Noise models. 



It also includes an example automated processing pipeline to download waveform data, correct instrument responses, and generate raw, corrected, and PPSD plots based on a simple CSV input.



\## Installation



\### Requirements

\* Python >= 3.13

\* ObsPy == 1.4.2

\* NumPy == 2.3.0

\* Matplotlib == 3.10.0



\### Install Locally

If you have downloaded the source code, navigate to the root directory (where `pyproject.toml` is located) and run:

```bash

pip install .

