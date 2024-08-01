# -*- coding: utf-8 -*-
import asyncio
import json
import math
import os

import edge_tts
import requests
import websocket
import yaml
from aip import AipSpeech
from pydub import AudioSegment
from pydub.playback import play


class audioModel:
    def __init__(self):
        self.text = None
        self.model = None
        with open("./config/voice.yaml", "r", encoding="utf-8") as f:
            self.data = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.header = self.data["header"]
        self.APP_ID = self.data["Baidu"]["APP_ID"]
        self.API_KEY = self.data["Baidu"]["API_KEY"]
        self.SECRET_KEY = self.data["Baidu"]["SECRET_KEY"]

    def setConfig(
        self,
        text=None,
        engine=None,
        voice=None,
        volume=None,
        rate=None,
        pitch=None,
        language=None,
    ):
        assert 0 <= len(text) < 200, "合成语音的文本长度应当大于0且小于200"
        self.text = text
        self.engine = engine
        self.voice = voice
        self.volume = volume
        self.rate = rate
        self.pitch = pitch
        self.language = language
        if self.engine == "Bert-Vits":
            self.model = self.Bert_VitsVoice
        elif self.engine == "GPT-Vits":
            self.model = self.GPT_VitsVoice
        elif self.engine == "百度":
            self.model = self.BaiduVoice
        elif self.engine == "Edge-tts":
            self.model = self.Edge_ttsVoice

    def Bert_VitsVoice(self, path):
        speaker = self.data["Bert-Vits"][self.voice]
        url1 = speaker["url1"]
        url2 = speaker["url2"]
        data = speaker["data"]
        data["data"][0] = self.text
        data["data"][6] = self.language
        response1 = requests.post(url=url1, headers=self.header, data=json.dumps(data))
        url2 = url2 + response1.json()["data"][1]["name"]
        response2 = requests.get(url=url2, headers=self.header)
        filename = path
        filename2 = "./audio/audio.wav"
        with open(filename, "wb") as f:
            f.write(response2.content)
        cmd = f'ffmpeg -i {filename} -af "volume={self.volume},atempo={self.rate}" {filename2}'
        os.system(cmd)
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(filename2, filename)

    def BaiduVoice(self, path):
        client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)
        result = client.synthesis(
            self.text,
            "zh",
            1,
            {
                "per": self.voice,
                "vol": self.volume,
                "pit": self.pitch,
                "spd": self.rate,
                "aue": 6,
                "cuid": "123456PYTHON",
            },
        )
        print(result)
        if not isinstance(result, dict):
            with open(path, "wb") as f:
                f.write(result)

    def GPT_VitsVoice(self, path):
        speaker = self.data["GPT-Vits"][self.voice]
        msg1 = speaker["msg1"]
        msg2 = speaker["msg2"]
        msg2["data"][3] = self.text
        msg2["data"][2] = self.language
        url1 = speaker["url1"]
        url2 = speaker["url2"]
        ws = websocket.create_connection(url1, timeout=10)
        ws.send(json.dumps(msg1))
        ws.send(json.dumps(msg2))
        count = 0
        while True:
            result = ws.recv()
            count += 1
            if count == 5:
                break
        ws.close()
        response1 = json.loads(result)["output"]["data"][0]["name"]
        url2 = url2 + response1
        response2 = requests.get(url=url2, headers=self.header)
        filename = path
        filename2 = "./audio/audio.wav"
        with open(filename, "wb") as f:
            f.write(response2.content)
        cmd = f'ffmpeg -i {filename} -af "volume={self.volume},atempo={self.rate}" {filename2}'
        os.system(cmd)
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(filename2, filename)

    async def Edge_ttsVoice(self, path):
        communicate = edge_tts.Communicate(
            text=self.text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )
        await communicate.save("./audio/temp.mp3")
        audio = AudioSegment.from_mp3("./audio/temp.mp3")
        audio.export(path, format="wav")
        os.remove("./audio/temp.mp3")

    def generateAudio(self, path="./audio/audio.wav"):
        if self.engine == "Edge-tts":
            asyncio.run(self.model(path))
        else:
            self.model(path)

    def getAudioInformation(self, path):
        audio = AudioSegment.from_wav(path)
        info = dict()
        info["frame_count"] = audio.frame_count()
        info["frame_rate"] = audio.frame_rate
        info["frame_width"] = audio.frame_width
        info["duration_seconds"] = audio.duration_seconds
        info["sample_width"] = audio.sample_width
        info["channel"] = audio.channels
        info["dBFS"] = audio.dBFS
        info["rms"] = audio.rms
        info["dB"] = 20 * math.log10(info["rms"])
        info["max_value"] = audio.max
        return info


if __name__ == "__main__":
    # Speaker = audioModel()
    # Speaker.setConfig("关注七域动漫协会谢谢喵\n", "taffy")
    # Speaker.generateAudio()
    # filename = "./audio/audio.wav"
    # try:
    #     audio = AudioSegment.from_wav("./audio/audio.wav")
    #     print("---------------------------------")
    #     print("总采样点数:", audio.frame_count())  # ?不传入数字的话默认一个采样点为一帧
    #     print("采样率:", audio.frame_rate)
    #     print("单点位数:", audio.frame_width)
    #     print("音频时长:", audio.duration_seconds)
    #     print("采样位数:", audio.sample_width)
    #     print("声道数:", audio.channels)
    #     print("全分贝刻度:", audio.dBFS)  # ?该信号的功率级别相对于满幅信号降低的dB数
    #     print("最大分贝刻度:", audio.max_dBFS)
    #     print("振幅均方根值:", audio.rms)
    #     print("最大振幅:", audio.max)
    #     print(len(audio.raw_data))
    #     play(audio)
    # except wave.Error as e:
    #     print(e.args)
    #     print("杂鱼！出错啦！才不是因为我写的程序有问题")
    # else:
    #     print("恭喜你！おめでとうございます！")
    # finally:
    #     print("您成功进行了一次语音合成")
    # os.remove(filename)

    # loop = asyncio.get_event_loop_policy().get_event_loop()
    # try:
    #     loop.run_until_complete(Speaker.Edge_ttsVoice())
    # finally:
    #     loop.close()

    Speaker = audioModel()
    Speaker.setConfig(
        "关注七域动漫协会谢谢喵",
        "Edge-tts",
        "zh-CN-shaanxi-XiaoniNeural",
        "+50%",
        "+50%",
        "+50Hz",
    )
    Speaker.generateAudio()
    audio = AudioSegment.from_wav("./audio/audio.wav")
    play(audio)
