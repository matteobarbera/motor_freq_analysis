# motor_freq_analysis

This repository contains the code used to determine the actuator dynamics of a brushless DC motor. The motor is driven with a chirp signal, a sinusoidal signal of linearly increasing frequency over time, while the resulting sound of the propeller is recorded with an iPhone. The sound of the propeller is uniquely linked to its RPM, so by listening to the frequency of the sound and knowing the frequency of the chirp input a Bode plot of the actuator frequency response can be constructed.

Here is the spectrogram of one of these experiments:

<img src="https://user-images.githubusercontent.com/22910604/179405248-7540a5d4-8065-46b2-82f1-69ba78e024d4.png" width=1000 />

The highest power frequencies of each time segment of the spectrogram was extracted. To construct the Bode plot, it was necessary to compute how the highest and lowest frequencies achieved by the motor evolved over time. So the highest frequencies (in red) and lowest (in green) were fitted with a 3rd degree polynomial.

<img src="https://user-images.githubusercontent.com/22910604/179405314-bdfb83da-a584-4b75-9f54-58660b1a75b7.png" width=1000 />

The difference between the two polyomials was computed over time and scaled by their largest difference to construct the Bode plot that shows the actuator frequency response.

<img src="https://user-images.githubusercontent.com/22910604/179405242-209037be-3390-4567-8f49-ff0b5301b28c.png" width=1000 />
