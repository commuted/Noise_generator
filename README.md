# Noise_generator

Generates a noise signal with a custom power spectral density (PSD) of 1/f^alpha.

    generate_custom_noise(frequency=44100, duration=3, alpha=1, output_file='custom_noise.wav')

Generate noise with a customizable power spectral density (PSD), where an **`alpha`** parameter in the argument determines the exponent of the frequency in the PSD, allowing you to create various types of noise such as **white**, **pink**, **brown**, 
**blue**, or **purple** noise.

    Parameters:
    - frequency (int): Sampling frequency in Hz (default: 44100)
    - duration (float): Duration of the signal in seconds (default: 3)
    - alpha (float): Exponent for the power spectral density (default: 1 for pink noise)
    - output_file (str): Name of the output WAV file (default: 'custom_noise.wav')

  ### Algorithm
  
  - Generate Gaussian white noise
  - Compute Fourier Transform
  - Compute frequencies
  - Compute scaling factors for the frequency domain
  - Apply scaling
  - Compute inverse Fourier Transform
  - Normalize the signal to prevent clipping
  - Convert to 16-bit integers 
  - Save as WAV file

Uncomment the graph if you want to plot the power spectral density (PSD) on a log-log plot;
  
