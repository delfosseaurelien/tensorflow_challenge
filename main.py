import os
import sys
import time
import numpy as np
from keras.callbacks import Callback
from scipy.io.wavfile import read, write
from keras.models import Model, Sequential
from keras.layers.merge import concatenate, multiply, add
from keras.layers import Conv1D, Flatten, Dense, \
    Input, Lambda, Activation


def wavenetBlock(nb_filters,
                 kernel_size,
                 dilatation_rate,
                 padding='causal'):
    def f(input_):
        residual = input_

        tanh_out = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          strides=1,
                          padding=padding,
                          dilation_rate=dilatation_rate,
                          activation='tanh')

        sigmoid = Conv1D(filters=nb_filters,
                          kernel_size=kernel_size,
                          strides=1,
                          padding=padding,
                          dilation_rate=dilatation_rate,
                          activation='sigmoid')

        merged = multiply([tanh_out,sigmoid])
        skip_out = Conv1D(1, 1, activation='relu', border_mode='same')(merged)
        out = add([skip_out, residual])
        return out, skip_out

    return f

input_size = 16000
input_ = Input(shape=(input_size, 1))
A, B = wavenetBlock(64, 2, 2)(input_)
skip_connections = [B]
for i in range(20):
    A, B = wavenetBlock(64, 2, 2**((i+2)%9))(A)
    skip_connections.append(B)
net = add(skip_connections, mode='sum')
net = Activation('relu')(net)
net = Conv1D(1, 1, activation='relu')(net)
net = Conv1D(1, 1)(net)
net = Flatten()(net)
net = Dense(256, activation='softmax')(net)
model = Model(input=input_, output=net)
model.compile(loss='categorical_crossentropy', optimizer='sgd',
              metrics=['accuracy'])
model.summary()

def get_basic_generative_model(input_size):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(64, 2, 2)(input_)
    skip_connections = [B]
    for i in range(20):
        A, B = wavenetBlock(64, 2, 2**((i+2)%9))(A)
        skip_connections.append(B)
    net = add(skip_connections, mode='sum')
    net = Activation('relu')(net)
    net = Conv1D(1, 1, activation='relu')(net)
    net = Conv1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(256, activation='softmax')(net)
    model = Model(input=input_, output=net)
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    model.summary()
    return model


def get_audio(filename):
    sr, audio = read(filename)
    audio = audio.astype(float)
    audio = audio - audio.min()
    audio = audio / (audio.max() - audio.min())
    audio = (audio - 0.5) * 2
    return sr, audio


def frame_generator(sr, audio, frame_size, frame_shift, minibatch_size=20):
    audio_len = len(audio)
    X = []
    y = []
    while 1:
        for i in range(0, audio_len - frame_size - 1, frame_shift):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= audio_len:
                break
            temp = audio[i + frame_size]
            target_val = int((np.sign(temp) * (np.log(1 + 256*abs(temp)) / (
                np.log(1+256))) + 1)/2.0 * 255)
            X.append(frame.reshape(frame_size, 1))
            y.append((np.eye(256)[target_val]))
            if len(X) == minibatch_size:
                yield np.array(X), np.array(y)
                X = []
                y = []


def get_audio_from_model(model, sr, duration, seed_audio):
    print('Generating audio...')
    new_audio = np.zeros((sr * duration))
    curr_sample_idx = 0
    while curr_sample_idx < new_audio.shape[0]:
        distribution = np.array(model.predict(seed_audio.reshape(1,
                                                                 frame_size, 1)
                                             ), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
        ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(
            ampl_val_8) - 1)) * 2**15
        new_audio[curr_sample_idx] = ampl_val_16
        seed_audio[-1] = ampl_val_16
        seed_audio[:-1] = seed_audio[1:]
        pc_str = str(round(100*curr_sample_idx/float(new_audio.shape[0]), 2))
        sys.stdout.write('Percent complete: ' + pc_str + '\r')
        sys.stdout.flush()
        curr_sample_idx += 1
    print('Audio generated.')
    return new_audio.astype(np.int16)


class SaveAudioCallback(Callback):
    def __init__(self, ckpt_freq, sr, seed_audio):
        super(SaveAudioCallback, self).__init__()
        self.ckpt_freq = ckpt_freq
        self.sr = sr
        self.seed_audio = seed_audio

    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%self.ckpt_freq==0:
            ts = str(int(time.time()))
            filepath = os.path.join('output/', 'ckpt_'+ts+'.wav')
            audio = get_audio_from_model(self.model, self.sr, 0.5, self.seed_audio)
            write(filepath, self.sr, audio)
