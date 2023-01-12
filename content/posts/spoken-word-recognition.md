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


In this example we will train a Machine Learning model to recognize spoken words. We will process audio data in a way that 
into spectrogram image data and so recast the original Audio Classification problem as an Image Classification one.


## Dataset: spoken words

*Free Spoken Digits* (FSDD) is a dataset **Spoken Word Recognition**. It consists of x audio samples



## Audio Processing

Sound are represented digitally as audio data, most commonly in waveform, which represents the amplitude of a sound wave. [explain: this is what microphones pick up, how is that signal translated to waveform]. Another possible representation is the audio spectrogram, which represents the frequencies of sound waves. Examples of these two representations are show below.


{{< figure src="/images/spoken-word-recognition/waveform_bw.png" width="50%" >}}

{{< figure src="/images/spoken-word-recognition/spectrogram_bw.png" width="45%" >}}


**From waveform to spectrogram.**
Given an audio signal $x(t)$ its Short-Time Fourier Transform is
$$
\text{STFT}\\{x(t)\\}(\tau, \omega) = X(\tau, \omega) = \int_{-\infty}^{\infty} x(t) w(t-\tau) e^{-i \omega t} \text{d}t
$$
where $x(t)$ is the waveform signal and $w(\tau)$ is a window function (e.g. Gaussian).
The STFT $X(\tau, \omega)$ then represents the signal over time ($\tau$) and frequency ($\omega$) domains, and is computed as the Fourier transform of the windowed function $x(t) w(t-\tau)$.

[explain: the spectrogram provides richer information as it disentangles the intensity of different frequencies of sound]


{{< figure src="/images/spoken-word-recognition/digits_spectrograms.png" width="70%" >}}


{{< figure src="/images/spoken-word-recognition/digits_spectrograms_grid.png" width="70%" >}}


{{< figure src="/images/spoken-word-recognition/mfcc.png" width="45%" >}}


## Model



## Results



{{< figure src="/images/spoken-word-recognition/audioclassif_fsd_cnn.png" width="80%" >}}


The confusion matrix reveals something interesting: 10% of utterances of the digit '8' are misclassified by the model as '6'. This would be expected from a model on image data, as the two have strong visual similarity. However, is seems that this similarity extends to the audio domain as well.

At this stage we can't rule out that this is due to bad training as the reverse is not true: ...



{{< figure src="/images/spoken-word-recognition/audioclassif_fsd_cnn__confusion.png" width="50%" >}}
