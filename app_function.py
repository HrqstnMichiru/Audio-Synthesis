# -*- coding: utf-8 -*-
import sys
from PyQt5.QtCore import pyqtSignal, QThread
import time
import pygame
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from model import audioModel
import json
import os
import pickle
from audioModel import SpeakerEncoder
import librosa
import numpy as np
import pyaudio

model_params_config = {
    "SpeakerEncoder": {
        "c_h": 128,
        "c_out": 128,
        "kernel_size": 5,
        "bank_size": 8,
        "bank_scale": 1,
        "c_bank": 128,
        "n_conv_blocks": 6,
        "n_dense_blocks": 3,
        "subsample": [1, 2, 1, 2, 1, 2],
        "dropout_rate": 0.5,
        "num_class": 31,
    }
}

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
LEN = 1

with open("./config/idx_2_cls0.pkl", "rb") as f:
    idx_2_cls = pickle.load(f)


def increment_path(path):
    for n in range(1, 9999):
        p = f"{path}{n}.wav"
        if not os.path.exists(p):
            break
    return p


def normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / (std + 1e-5)
    return data


def pad(x, maxlen, mode="wrap"):
    pad_len = maxlen - x.shape[1]
    y = np.pad(x, ((0, 0), (0, pad_len)), mode=mode)
    return y


def randntrunc(x, maxlen):
    r = np.random.randint(x.shape[1] - maxlen)
    y = x[:, r : r + maxlen]
    return y


def segment2d(x, maxlen):
    ## 该函数将melspec [80,len] ，padding到固定长度 seglen
    if x.shape[1] < maxlen:
        y = pad(x, maxlen)
    elif x.shape[1] == maxlen:
        y = x
    else:
        y = randntrunc(x, maxlen)
    return y


def wav_to_mel(wavpath=None, data=None, sr=None):  # 计算log FBank特征
    # 计算log FBank特征
    if wavpath:
        y, fs = librosa.load(wavpath, sr=16000)
    else:
        y = data
        fs = sr
    fbank = librosa.feature.melspectrogram(
        y, sr=fs, n_fft=1024, win_length=1024, hop_length=256, n_mels=128
    )
    fbank_db = librosa.power_to_db(fbank, ref=np.max)
    fbank_db = segment2d(fbank_db, maxlen=64)

    fbank_db = normalization(fbank_db)
    fbank_db = fbank_db.T
    return fbank_db


class audioRecognition(QThread):
    audiosignal = pyqtSignal(int)
    audioplaysignal=pyqtSignal(str)
    def __init__(self, audioModel, mode, parent=None, state=None,path=None):
        super().__init__(parent)
        self.path = path
        self.audioModel = audioModel
        self.parent = parent
        self.mode = mode
        self.state=state

    def run(self):
        if self.mode == 1:
            fbank = wav_to_mel(self.path).reshape(1, 64, 128)
            output = self.audioModel.predict(fbank).argmax(-1).item()
            self.parent.lineEdit_2.setText(idx_2_cls[output])
            self.audiosignal.emit(output)
        elif self.mode == 2:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
            print("start recording......")
            while True:
                if stop_status == 1:
                    frames = []
                    for _ in range(0, int(RATE / CHUNK * LEN)):
                        data = stream.read(CHUNK)
                        frames.append(data)
                    audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
                    audio_data = audio_data / np.max(audio_data)
                    audio_data = audio_data * 1.0
                    fbank = wav_to_mel(data=audio_data, sr=RATE).reshape(1, 64, 128)
                    prediction = self.audioModel.predict(fbank)
                    confidence = prediction.max(-1).item()
                    print(confidence)
                    if confidence > 0.8:
                        output = prediction.argmax(-1).item()
                        result = idx_2_cls[output]
                        self.parent.lineEdit_2.setText(result)
                        print(result)
                        if self.state:
                            if result=='left':
                                self.audioplaysignal.emit('left')
                            elif result=='right':
                                self.audioplaysignal.emit('right')
                            elif result=='stop':
                                self.audioplaysignal.emit("stop")
                            elif result=='go':
                                self.audioplaysignal.emit('go')
                            elif result=='off':
                                self.audioplaysignal.emit("off")
                elif stop_status == 0:
                    print("stop recording......")
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    break


class makingAudio(QThread):
    audiosignal = pyqtSignal(str)
    printsignal = pyqtSignal(str)

    def __init__(self, model=None, parent=None, savefile="./audio/audio.wav"):
        super().__init__(parent)
        self.model = model
        self.parent = parent
        self.savefile = savefile

    def run(self):
        self.parent.pushButton.setEnabled(False)
        self.model.generateAudio(self.savefile)
        self.parent.listWidget.addItem(self.savefile)
        self.parent.statusBar.showMessage("您刚刚合成了一条语音!")
        # os.system("cls")
        # info = self.model.getAudioInformation(self.savefile)
        self.audiosignal.emit(self.savefile)
        # self.printsignal.emit(json.dumps(info, ensure_ascii=False))


class AudioPlay(QThread):
    endsignal = pyqtSignal()

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.path = path
        self.parent.pushButton_5.setEnabled(True)
        self.parent.pushButton_6.setEnabled(True)

    def pause(self):
        pygame.mixer.music.pause()

    def resume(self):
        pygame.mixer.music.unpause()

    def run(self):
        pygame.mixer.init()
        pygame.mixer.music.load(self.path)
        pygame.mixer.music.play()
        while True:
            t = pygame.mixer.music.get_pos()
            if t == -1:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                break
        self.parent.pushButton_5.setEnabled(False)
        self.parent.pushButton_6.setEnabled(False)
        self.parent.pushButton.setEnabled(True)
        self.endsignal.emit()


class emitStr(QThread):
    def __init__(self, parent=None, target=None, args=None):
        super().__init__(parent)
        self.target = target
        self.args = args

    def run(self):
        if self.args == None:
            self.target()
        else:
            self.target(self.args)


class callBack:
    def __init__(self, obj):
        self.obj = obj
        self.model = audioModel()
        self.audioModel = SpeakerEncoder(**model_params_config["SpeakerEncoder"])
        self.audioModel.build(input_shape=(None, 64, 128))
        self.audioModel.load_weights("./config/model_16_0.92.h5", by_name=True)
        self.savepath = None
        self.COUNT = -1
        self.out=sys.stdout.write

    def onComboBoxIndexChanged(self):
        index = self.obj.comboBox.currentIndex()
        text = self.obj.comboBox.currentText()
        self.obj.comboBox_2.clear()
        self.obj.comboBox_3.clear()
        self.obj.comboBox_2.addItems([i[0] for i in self.obj.items[text].items()])
        if text in ["Bert-Vits", "GPT-Vits"]:
            self.obj.comboBox_3.addItems(
                [i[0] for i in self.obj.items["Language"][text].items()]
            )
        else:
            self.obj.comboBox_3.addItems(self.obj.items["Language"][text])
        self.obj.stackedLayout.setCurrentIndex(index)

    def onComboBox_3IndexChanged(self):
        text = self.obj.comboBox.currentText()
        index = self.obj.comboBox_3.currentIndex()
        if text == "Edge-tts":
            self.obj.comboBox_2.clear()
            speaker = self.obj.items[text]
            if index == 0:
                speaker = filter(lambda x: "liaoning" in x, speaker)
            elif index == 1:
                speaker = filter(lambda x: "TW" in x, speaker)
            elif index == 2:
                speaker = filter(lambda x: "HK" in x, speaker)
            elif index == 3:
                speaker = filter(lambda x: "CN-Y" in x or "CN-X" in x, speaker)
            elif index == 4:
                speaker = filter(lambda x: "shaanxi" in x, speaker)
            elif index == 5:
                speaker = filter(lambda x: "JP" in x, speaker)
            elif index == 6:
                speaker = filter(lambda x: "US" in x, speaker)
            self.obj.comboBox_2.addItems(speaker)

    def printToTextBrowser_2(self, message):
        self.obj.textBrowser_2.moveCursor(self.obj.textBrowser_2.textCursor().End)
        self.obj.textBrowser_2.insertPlainText(message)
        time.sleep(0.1)

    def printItems(self):
        self.obj.pushButton_3.setEnabled(False)
        items = self.obj.items
        print("语音源:")
        [print(i, end=",") for i in items["Engine"]]
        print("", end="\n")
        print("------------------------------------")
        print("百度智能云在线语音合成:")
        print("支持的语言:", items["Language"]["百度"][0])
        print("支持的角色:")
        [print(i[0]) for i in items["百度"].items()]
        print("------------------------------------")
        print("Bert-Vits在线语音合成:")
        print("支持的语言:")
        [print(i, end=",") for i in items["Language"]["Bert-Vits"]]
        print("", end="\n")
        print("支持的角色:")
        [print(i[0], end=",") for i in items["Bert-Vits"].items()]
        print("", end="\n")
        print("------------------------------------")
        print("GPT-Vits在线语音合成:")
        print("支持的语言:")
        [print(i, end=",") for i in items["Language"]["GPT-Vits"]]
        print("", end="\n")
        print("支持的角色:")
        [print(i[0], end=",") for i in items["GPT-Vits"].items()]
        print("", end="\n")
        print("------------------------------------")
        print("Edge-tts在线语音合成:")
        print("支持的语言:")
        [print(i, end=",") for i in items["Language"]["Edge-tts"]]
        print("", end="\n")
        print("支持的角色:")
        [print(i[0]) for i in items["Edge-tts"].items()]
        sys.stdout.write=self.out
        self.obj.pushButton_3.setEnabled(True)

    def printToTextBrowser(self, message):
        self.obj.textBrowser.moveCursor(self.obj.textBrowser.textCursor().End)
        self.obj.textBrowser.insertPlainText(message)
        time.sleep(0.1)

    def printAudioInfo(self, info):
        print("总采样点数：", info["frame_count"])
        print("采样率：", str(info["frame_rate"]) + "Hz")
        print("帧采样宽度：", str(info["frame_width"]) + "Byte")
        print("音频时长：", f'{info["duration_seconds"]:.3f}s')
        print("采样宽度：", str(info["sample_width"]) + "Byte")
        print("声道数：", info["channel"])
        print("分贝刻度：", f'{info["dBFS"]:.3f}dB')
        print("振幅均方根值：", info["rms"])
        print("音量：", f'{info["dB"]:.3f}dB')
        print("最大振幅：", info["max_value"])
        print()
        sys.stdout.write=self.out

    def onPushButton_2Clicked(self):
        sys.stdout.write = self.printToTextBrowser_2
        thread1 = emitStr(self.obj, target=self.printItems)
        thread1.start()

    def clearBrowser_2(self):
        self.obj.textBrowser_2.clear()

    def clearBrowser(self):
        self.obj.textBrowser.clear()

    def onPushButtonClicked(self,flag):
        if flag:
            text = self.obj.textEdit.toPlainText()
        else:
            text=self.obj.lineEdit_2.text()
        print(text)
        if not text:
            QMessageBox.warning(
                self.obj,
                "警告",
                "杂鱼！你忘记输入文本啦！才不是因为我写的程序有问题",
                QMessageBox.Ok | QMessageBox.Discard,
                QMessageBox.Ok,
            )
            return
        engine = self.obj.comboBox.currentText()
        self.engineIndex = self.obj.comboBox.currentIndex()
        speaker = self.obj.comboBox_2.currentText()
        voice = self.obj.items[engine][speaker]
        language = self.obj.comboBox_3.currentText()
        if self.obj.checkBox_3.isChecked():
            savedir = "./audio/audio"
            savefile = increment_path(savedir)
            self.obj.lineEdit.setText(savefile)
        if self.savepath != None:
            savefile = self.savepath
            self.obj.lineEdit.setText(savefile)
        if self.engineIndex == 0:
            volume = self.obj.HSlider_2.value()
            pitch = self.obj.HSlider_3.value()
            rate = self.obj.HSlider_4.value()
            self.model.setConfig(text, engine, voice, volume, rate, pitch)
        elif self.engineIndex == 1:
            volume = self.obj.HSlider.value() / 50
            rate = self.obj.doubleSpinBox.value()
            language = self.obj.items["Language"][engine][language]
            self.model.setConfig(text, engine, voice, volume, rate, language=language)
        elif self.engineIndex == 2:
            volume = self.obj.HSlider_5.value() / 50
            rate = self.obj.doubleSpinBox_2.value()
            language = self.obj.items["Language"][engine][language]
            self.model.setConfig(text, engine, voice, volume, rate, language=language)
        elif self.engineIndex == 3:
            rate = self.obj.spinBox.value()
            if rate >= 0:
                rate = "+" + str(rate) + "%"
            else:
                rate = str(rate) + "%"
            volume = self.obj.spinBox_2.value()
            if volume >= 0:
                volume = "+" + str(volume) + "%"
            else:
                volume = str(volume) + "%"
            pitch = self.obj.spinBox_3.value()
            if pitch >= 0:
                pitch = "+" + str(pitch) + "Hz"
            else:
                pitch = str(pitch) + "Hz"
            self.model.setConfig(text, engine, voice, volume, rate, pitch)
        thread2 = makingAudio(self.model, self.obj, savefile)
        thread2.audiosignal.connect(self.audioPlay)
        thread2.printsignal.connect(self.audioPrint)
        thread2.start()

    def audioPlay(self, savefile):
        if self.obj.checkBox_2.isChecked():
            self.thread3 = AudioPlay(savefile, self.obj)
            self.thread3.endsignal.connect(self.end)
            self.thread3.start()
            self.obj.statusBar.showMessage("正在播放语音!", msecs=2000)
        else:
            self.obj.pushButton.setEnabled(True)
        if self.obj.checkBox.isChecked():
            _, _ = QFileDialog.getOpenFileName(
                self.obj, "已合成的音乐文件", "./audio", "音频文件(*.mp3 *.wav)"
            )

    def onPushButton_5Clicked(self):
        self.thread3.pause()

    def onPushButton_6Clicked(self):
        self.thread3.resume()

    def audioPrint(self, info):
        sys.stdout.write = self.printToTextBrowser
        info = json.loads(info)
        thread1 = emitStr(self.obj, target=self.printAudioInfo, args=info)
        thread1.start()

    def onPushButton_8Clicked(self):
        path, _ = QFileDialog.getSaveFileName(
            self.obj, "音乐文件保存地址", os.getcwd(), "音频文件(*.mp3 *.wav)"
        )
        self.savepath = path

    def onPushButton_7Clicked(self):
        item = self.obj.listWidget.currentItem()
        if item:
            path = item.text()
            self.thread3 = AudioPlay(path, self.obj)
            self.thread3.endsignal.connect(self.end)
            self.thread3.start()
            self.obj.statusBar.showMessage("正在播放语音!", msecs=2000)

    def end(self):
        self.obj.statusBar.showMessage("语音播放结束", 2000)

    def onPushButton_9Clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self.obj, "本地音频文件", "E:/New_PythonProjects/keras", "音频文件(*.mp3 *.wav)"
        )
        if path:
            thread4 = audioRecognition(self.audioModel, 1, parent=self.obj, path=path)
            thread4.audiosignal.connect(self.audio_synthesis)
            thread4.start()
            self.obj.statusBar.showMessage("语音识别开始", 2000)
        else:
            self.obj.statusBar.showMessage("未选择文件", 2000)

    def audio_synthesis(self,msg):
        print(msg)
        if self.obj.checkBox_5.isChecked():
            self.onPushButtonClicked(0)
        else:
            return

    def onPushButton_11Clicked(self):
        global stop_status
        stop_status = 1
        print("开始语音识别")
        state = self.obj.checkBox_4.isChecked()
        thread5 = audioRecognition(self.audioModel, 2, parent=self.obj,state=state)
        thread5.audioplaysignal.connect(self.decisionTree)
        thread5.start()
        self.obj.statusBar.showMessage("语音识别开始", 2000)

    def onpushButton_10Clicked(self):
        global stop_status
        stop_status = 0
        print('停止语音识别')
        self.obj.statusBar.showMessage("语音识别结束", 2000)

    def decisionTree(self,msg):
        print(msg)
        maxCount=self.obj.listWidget.count()
        if msg in ['right','left']:
            if msg=='right':
                if self.COUNT<maxCount:
                    self.COUNT+=1
            if msg=='left':
                if self.COUNT>-1:
                    self.COUNT -= 1
            print(self.COUNT)
            self.obj.listWidget.setCurrentRow(self.COUNT) 
            item=self.obj.listWidget.currentItem()
            path = item.text()
            self.thread3 = AudioPlay(path, self.obj)
            self.thread3.endsignal.connect(self.end)
            self.thread3.start()
            self.obj.statusBar.showMessage("正在播放语音!", msecs=2000)
        if msg=='stop':
            self.onPushButton_5Clicked()
        if msg=='off':
            self.onpushButton_10Clicked()
        if msg=='go':
            self.onPushButton_6Clicked()
