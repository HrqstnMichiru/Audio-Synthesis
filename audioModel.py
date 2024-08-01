import warnings

import keras
import tensorflow as tf
from keras.layers import (
    AveragePooling1D,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)

warnings.filterwarnings("ignore", category=UserWarning)


def pad_layer(inp, layer, pad_type="REFLECT"):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = [
            [0, 0],
            [kernel_size // 2, kernel_size // 2 - 1],
            [0, 0],
        ]
    else:
        pad = [
            [0, 0],
            [kernel_size // 2, kernel_size // 2],
            [0, 0],
        ]
    # padding
    inp = tf.pad(inp, paddings=pad, mode=pad_type)
    # print('inp',inp.shape)
    out = layer(inp)
    return out


def conv_bank(x, act, module_list, pad_type="reflect"):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = tf.concat(outs + [x], axis=2)
    # print("conv_bank out shape",out.shape)
    return out


class SpeakerEncoder(keras.Model):
    def __init__(
        self,
        c_h,
        c_out,
        kernel_size,
        bank_size,
        bank_scale,
        c_bank,
        n_conv_blocks,
        n_dense_blocks,
        subsample,
        dropout_rate,
        num_class,
    ):
        super().__init__()
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.act = tf.nn.relu
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.conv_bank = [
            Conv1D(c_bank, kernel_size=k)
            for k in range(bank_scale, bank_size + 1, bank_scale)
        ]
        self.in_conv_layer = Conv1D(c_h, kernel_size=1)
        self.first_conv_layers = [
            Conv1D(c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)
        ]
        self.second_conv_layers = [
            Conv1D(c_h, kernel_size=kernel_size, strides=sub)
            for sub, _ in zip(subsample, range(n_conv_blocks))
        ]
        self.pooling_layer = GlobalAveragePooling1D("channels_last")
        self.first_dense_layers = [Dense(c_h) for _ in range(n_dense_blocks)]
        self.second_dense_layers = [Dense(c_h) for _ in range(n_dense_blocks)]
        self.output_layer = Dense(c_out)
        self.batchNorm = BatchNormalization()
        self.drop_layer = Dropout(dropout_rate)
        self.cls_layer = Dense(num_class, activation="softmax")

    def conv_blocks(self, inp):
        out = inp
        # convolution blocks
        for i in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[i])
            y = self.batchNorm(y)
            # y = self.drop_layer(y)
            y = self.act(y)
            y = pad_layer(y, self.second_conv_layers[i])
            y = self.batchNorm(y)
            # y = self.drop_layer(y)
            y = self.act(y)
            if self.subsample[i] > 1:
                out = AveragePooling1D(pool_size=self.subsample[i], padding="same")(out)
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        # dense layers
        for i in range(self.n_dense_blocks):
            y = self.first_dense_layers[i](out)
            y = self.act(y)
            y = self.drop_layer(y)
            y = self.second_dense_layers[i](y)
            y = self.act(y)
            y = self.drop_layer(y)
            out = y + out
        return out

    def call(self, x):
        # print("--- spk encoder ---")
        out = conv_bank(x, self.act, self.conv_bank)
        out = self.act(out)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)
        out = self.act(out)

        # avg pooling
        out = self.pooling_layer(out)

        # print(out.shape)
        # dense blocks
        out = self.dense_blocks(out)
        out = self.act(out)

        out = self.output_layer(out)
        out = self.act(out)
        # print("---")
        out = self.cls_layer(out)
        return out

    def look_shape_forward(self, x):
        print("--- spk encoder ---")
        print("输入数据维度:x", x.shape)
        out = conv_bank(x, self.act, self.conv_bank)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        print("经过conv bank", out.shape)
        # conv blocks
        out = self.conv_blocks(out)
        print("经过多个卷积层：", out.shape)
        # avg pooling
        print(out.shape)
        out = self.pooling_layer(out)
        print(out.shape)
        print("经过一个时间维度的池化层：", out.shape)

        print(out.shape)
        # dense blocks
        out = self.dense_blocks(out)
        print("经过多个线性层：", out.shape)
        out = self.output_layer(out)
        print("经过输出层：", out.shape)
        # print("---")
        out = self.cls_layer(out)
        print("经过分类层：", out.shape)
        return out


def testmodel():
    """
    测试一下写的模型 能不能正常输入输出
    """
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
            "subsample": [1, 2, 1, 2, 1, 2],  ## 下采样的主要功能：缩小时间帧
            "dropout_rate": 0.5,
            "num_class": 10,  ##这里改类别数
        }
    }
    speaker_clsmodel = SpeakerEncoder(
        **model_params_config["SpeakerEncoder"]
    )  ## 模型的定义
    speaker_clsmodel.build(input_shape=(None, 64, 128))
    speaker_clsmodel.summary()
    a = tf.random.normal(shape=(3, 64, 128))  ## 输入 频谱特征

    # out = speaker_clsmodel(a)  ## 输入给模型
    out = speaker_clsmodel.look_shape_forward(a)
    print(out.shape)


if __name__ == "__main__":
    testmodel()
