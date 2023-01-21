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


# Introduction


In this example we will train a Machine Learning model to recognize spoken words. We will process audio data in a way that 
into spectrogram image data and so recast the original Audio Classification problem as an Image Classification one.


# Audio Processing

Sound are represented digitally as audio data, most commonly in waveform, which represents the amplitude of a sound wave. [explain: this is what microphones pick up, how is that signal translated to waveform]. Another possible representation is the audio spectrogram, which represents the frequencies of sound waves. Examples of these two representations are show below.


{{< figure src="/images/spoken-word-recognition/waveform_bw.png" width="50%" >}}

{{< figure src="/images/spoken-word-recognition/spectrogram_bw.png" width="45%" >}}

Other types of features could be used, a common one being the Mel Frequency Cepstral Coefficients:

{{< figure src="/images/spoken-word-recognition/mfcc.png" width="45%" >}}


##### From waveform to spectrogram

Given an audio signal $x(t)$ its Short-Time Fourier Transform (STFT) is
$$
\text{STFT}\\{x(t)\\}(\tau, \omega) = X(\tau, \omega) = \int_{-\infty}^{\infty} x(t) w(t-\tau) e^{-i \omega t} \text{d}t
$$
where $x(t)$ is the waveform signal and $w(\tau)$ is a window function (e.g. Gaussian).
The STFT $X(\tau, \omega)$ then represents the signal over time ($\tau$) and frequency ($\omega$) domains, and is computed as the Fourier transform of the windowed function $x(t) w(t-\tau)$.

!!!!!!!!!!!!!!!!!!!!![explain: the spectrogram provides richer information as it disentangles the intensity of different frequencies of sound]


##### From audio to image

...

# Dataset

The [*Free Spoken Digits Dataset*](https://github.com/Jakobovski/free-spoken-digit-dataset) (FSDD) is a dataset for **Spoken Word Recognition**. It consists of 3,000 audio samples from 6 speakers of the digits 0-9, i.e. 50 samples of each digit per speaker.
This is a subset of 10% of the [*AudioMNIST dataset*](https://github.com/soerenab/AudioMNIST).

The recordings are pre-processed to extract the spectrograms. The necessary conditon for spectrograms to be successfully used as the basis for classification is that they show
1. consistency with class, i.e. similarity in the spectrograms of the same digit across different samples and speakers;
2. variability between classes, i.e. differences in the spectral signatures across different digits;

Below we show an example of the STFT spectrograms for two different digits, where we can see some differences in spectral signatures:

{{< figure src="/images/spoken-word-recognition/digits_spectrograms.png" width="70%" >}}

Similarly, we can see identify some consistency within samples of the same digit:

{{< figure src="/images/spoken-word-recognition/digits_spectrograms_grid.png" width="70%" >}}




## Model



## Results



{{< figure src="/images/spoken-word-recognition/audioclassif_fsd_cnn.png" width="80%" >}}


The confusion matrix reveals something interesting: 10% of utterances of the digit '8' are misclassified by the model as '6'. This would be expected from a model on image data, as the two have strong visual similarity. However, is seems that this similarity extends to the audio domain as well - given the model trained.
Note that the reverse is not true: utterances of '6' are never misclassified as '8'.


{{< figure src="/images/spoken-word-recognition/audioclassif_fsd_cnn__confusion.png" width="50%" >}}
