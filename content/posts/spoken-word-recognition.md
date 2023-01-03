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

## Introduction 

Sound waves are represented digitally as audio data, most commonly in waveform. The waveform depicts the amplitude of the sound wave. Another possible representation is the audio spectrogram, which represents the frequencies of sound waves. Examples of these two representations are show below.

In this example we will train a Machine Learning model to recognize spoken words. We will transform waveform sound data into spectrogram image data and so recast the original Audio Classification problem as an Image Classification one.