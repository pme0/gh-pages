+++

author = "pme0"
title = "Spoken Word Recognition"
date = "2023-01-01"
description = ""
tags = [
    "Audio",
    "Classification",
    "CNN", 
]
math = true

+++


{{< figure src="/images/spoken-word-recognition/thumbnail_spectrogram_grid10.png" width="25%" >}}


## Introduction 

Sound waves are represented digitally as audio data, most commonly in waveform. The waveform depicts the amplitude of the sound wave throughout a recording. Another possible representation is the audio spectrogram, which represents the frequencies of sound waves. Examples of these two representations are show below.

In this example we will train a Machine Learning model to recognize spoken words. We will transform waveform sound data into spectrogram image data and so recast the original Audio Classification problem as an Image Classification one.


## features

Given an audio signal $x(t)$ its Short-Time Fourier Transform (STFT)......
$$
\text{STFT}\\{x(t)\\}(\tau, \omega) = X(\tau, \omega) = \int_{-\infty}^{\infty} x(t) w(t-\tau) e^{-i \omega t} \text{d}t
$$
where $x(t)$ is the audio signal and $w(\tau)$ is a window function (e.g. Gaussian).
The STFT $X(\tau, \omega)$ then represents the signal over time ($\tau$) and frequency ($\omega$) domains, and is computed as the Fourier transform of the windowed function $x(t) w(t-\tau)$.



{{< figure src="/images/spoken-word-recognition/waveform_bw.png" width="50%" >}}


{{< figure src="/images/spoken-word-recognition/spectrogram_bw.png" width="45%" >}}


{{< figure src="/images/spoken-word-recognition/digits_spectrograms.png" width="70%" >}}


{{< figure src="/images/spoken-word-recognition/digits_spectrograms_grid.png" width="70%" >}}


{{< figure src="/images/spoken-word-recognition/mfcc.png" width="45%" >}}


{{< figure src="/images/spoken-word-recognition/audioclassif_fsd_cnn.png" width="80%" >}}


{{< figure src="/images/spoken-word-recognition/audioclassif_fsd_cnn__confusion.png" width="50%" >}}
