# LSTM_Biosignals_Classification

In LSTMN.py a Long-Short Time Memory (LSTM) Network able to classify hand movements can be found. The EMG records have been taken from [1]. In this case, the movements classified are the one corresponding to hand closed, hand opened and supination. The data used is the one recorded from the only two trained subjects. Finally, an accuracy equal to 64 % has been achieved.
On the other hand, the file LSTMN_EEG.py contains a LSTM Network able to differentiate between visual and audio stimuli. The Network is based on the one implemented in the work that can be found in [2]. Moreover, the data used is the same as in [2] and can be found in [3]. Finally, it is worth noticing that instead of using Dropout layers, Lasso has been used for regularization, which has helped to increase the accuracy until a 92%, which supposes an improvement of a 10% with respect to the results obtained in [2].


References:

[1] https://github.com/biopatrec/biopatrec/wiki/Data_Repository.md
[2] https://github.com/Cerebro409/EEG-Classification-Using-Recurrent-Neural-Network
[3] https://ucmerced.box.com/s/m8tpshib6zztio9q34h9fh9ikhhl4a3m
