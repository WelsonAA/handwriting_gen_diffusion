import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Dense, Conv1D, Embedding, UpSampling1D, AveragePooling1D,
                                     AveragePooling2D, GlobalAveragePooling2D, Activation,
                                     LayerNormalization, Dropout, Layer)


def create_padding_mask(seq, repeats=1):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = tf.repeat(seq, repeats=repeats, axis=-1)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    return mask


def ff_network(C, dff=768, act_before=True):
    ff_layers = [
        Dense(dff, activation='swish'),
        Dense(C)
    ]
    if act_before: ff_layers.insert(0, Activation('swish'))
    return Sequential(ff_layers)


class AffineTransformLayer(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.gamma_dense = Dense(filters, bias_initializer='ones')
        self.beta_dense = Dense(filters)

    def call(self, x, sigma):
        gammas = self.gamma_dense(sigma)
        betas = self.beta_dense(sigma)

        # Reshape `gammas` and `betas` to match the shape of `x` for broadcasting
        gammas = tf.reshape(gammas, [-1, 1, 1, 1, tf.shape(gammas)[-1]])
        betas = tf.reshape(betas, [-1, 1, 1, 1, tf.shape(betas)[-1]])

        return x * gammas + betas


class MultiHeadAttention(Layer):
    def __init__(self, C, num_heads):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.wq = Dense(C)
        self.wk = Dense(C)
        self.wv = Dense(C)
        self.dense = Dense(C)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.C // self.num_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q, k, v = self.wq(q), self.wk(k), self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        attention_weights = tf.nn.softmax(
            tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.C // self.num_heads, tf.float32)), axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.C))
        output = self.dense(concat_attention)
        return output, attention_weights


class StyleExtractor(Model):
    def __init__(self):
        super().__init__()
        self.mobilenet = MobileNetV2(include_top=False, pooling=None, weights='imagenet', input_shape=(96, 96, 3))
        self.local_pool = AveragePooling2D((3, 3))
        self.global_avg_pool = GlobalAveragePooling2D()
        self.freeze_all_layers()

    def freeze_all_layers(self):
        for l in self.mobilenet.layers:
            l.trainable = False

    def call(self, im, training=False):
        x = tf.cast(im, tf.float32)
        x = tf.image.resize(x, (96, 96))
        if x.shape[-1] == 1:
            x = tf.image.grayscale_to_rgb(x)
        x = (x / 127.5) - 1
        x = self.mobilenet(x, training=training)
        x = self.local_pool(x)
        return tf.squeeze(x, axis=1)


class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, drop_rate=0.1, pos_factor=1):
        super().__init__()
        self.text_pe = tf.Variable(tf.random.uniform([1, 2000, d_model]), trainable=False)
        self.drop = Dropout(drop_rate)
        self.lnorm = LayerNormalization(epsilon=1e-6, trainable=False)
        self.text_dense = Dense(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = ff_network(d_model, d_model * 2)
        self.affine_layers = [AffineTransformLayer(d_model) for _ in range(4)]

    def call(self, x, text, sigma, text_mask):
        text = self.text_dense(tf.nn.swish(text))
        text = self.affine_layers[0](self.lnorm(text), sigma)
        text_pe = text + self.text_pe[:, :tf.shape(text)[1]]

        x_pe = x
        x2, att = self.mha(x_pe, text_pe, text, text_mask)
        x2 = self.lnorm(self.drop(x2))
        x2 = self.affine_layers[1](x2, sigma) + x

        x3, _ = self.mha(x2, x2, x2)
        x3 = self.lnorm(x2 + self.drop(x3))
        x3 = self.affine_layers[2](x3, sigma)

        x4 = self.ffn(x3)
        x4 = self.drop(x4) + x3
        return self.affine_layers[3](self.lnorm(x4), sigma), att


class Text_Style_Encoder(Model):
    def __init__(self, d_model, d_ff=512):
        super().__init__()
        self.emb = Embedding(73, d_model)
        self.text_conv = Conv1D(d_model, 3, padding='same')
        self.style_ffn = ff_network(d_model, d_ff)
        self.mha = MultiHeadAttention(d_model, 8)
        self.layernorm = LayerNormalization(epsilon=1e-6, trainable=False)
        self.dropout = Dropout(0.3)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
        self.affine4 = AffineTransformLayer(d_model)
        self.text_ffn = ff_network(d_model, d_model * 2)

    def call(self, text, style, sigma):
        # Ensure `style` matches the shape required for attention
        style = tf.reshape(style, [tf.shape(style)[0], 1, -1])  # Shape: [batch_size, 1, 1280]
        style = tf.broadcast_to(style, [tf.shape(text)[0], tf.shape(text)[1],
                                        tf.shape(style)[-1]])  # Shape: [batch_size, seq_length, 1280]

        # Apply affine transformations to style and text
        style = self.affine1(self.layernorm(self.style_ffn(style)), sigma)
        text = self.emb(text)
        text = self.affine2(self.layernorm(text), sigma)

        # Apply multi-head attention
        mha_out, _ = self.mha(text, style, style)

        # Ensure shapes are compatible for addition
        text_shape = tf.shape(text)
        mha_out = tf.reshape(mha_out, text_shape)  # Reshape mha_out to match text

        # Add and normalize
        text = self.affine3(self.layernorm(text + mha_out), sigma)

        # Output with final affine transformation
        text_out = self.affine4(self.layernorm(self.text_ffn(text)), sigma)
        return text_out


class ConvSubLayer(Model):
    def __init__(self, filters, dils=[1,1], activation='swish', drop_rate=0.0):
        super().__init__()
        self.act = Activation(activation)
        self.affine1 = AffineTransformLayer(filters//2)
        self.affine2 = AffineTransformLayer(filters)
        self.affine3 = AffineTransformLayer(filters)
        self.conv_skip = Conv1D(filters, 3, padding='same')
        self.conv1 = Conv1D(filters//2, 3, dilation_rate=dils[0], padding='same')
        self.conv2 = Conv1D(filters, 3, dilation_rate=dils[0], padding='same')
        self.fc = Dense(filters)
        self.drop = Dropout(drop_rate)

    def call(self, x, alpha):
        x_skip = self.conv_skip(x)
        x = self.conv1(self.act(x))
        x = self.drop(self.affine1(x, alpha))
        x = self.conv2(self.act(x))
        x = self.drop(self.affine2(x, alpha))
        x = self.fc(self.act(x))
        x = self.drop(self.affine3(x, alpha))
        x += x_skip
        return x


class DiffusionWriter(Model):
    def __init__(self, num_layers=4, c1=128, c2=192, c3=256, drop_rate=0.1, num_heads=8):
        super().__init__()
        self.input_dense = Dense(c1)  # Initial dense layer for image data

        # Sigma transformation layers
        self.sigma_ffn = ff_network(c1 // 4, 2048)

        # Encoding and decoding layers
        self.enc1 = ConvSubLayer(c1, [1, 2], drop_rate=drop_rate)
        self.enc2 = ConvSubLayer(c2, [1, 2], drop_rate=drop_rate)
        self.enc3 = DecoderLayer(c2, num_heads, drop_rate, pos_factor=4)
        self.enc4 = ConvSubLayer(c3, [1, 2], drop_rate=drop_rate)
        self.enc5 = DecoderLayer(c3, num_heads, drop_rate, pos_factor=2)

        # Pooling and upsampling layers
        self.pool = AveragePooling1D(2)
        self.upsample = UpSampling1D(2)

        # Skip connections
        self.skip_conv1 = Conv1D(c2, 3, padding='same')
        self.skip_conv2 = Conv1D(c3, 3, padding='same')
        self.skip_conv3 = Conv1D(c2 * 2, 3, padding='same')

        # Text and style encoders
        self.text_style_encoder = Text_Style_Encoder(c2 * 2, c2 * 4)
        self.att_dense = Dense(c2 * 2)
        self.att_layers = [DecoderLayer(c2 * 2, num_heads, drop_rate) for _ in range(num_layers)]

        # Decoder sub-layers
        self.dec3 = ConvSubLayer(c3, [1, 2], drop_rate=drop_rate)
        self.dec2 = ConvSubLayer(c2, [1, 1], drop_rate=drop_rate)
        self.dec1 = ConvSubLayer(c1, [1, 1], drop_rate=drop_rate)

        # Final dense layer for image output
        self.output_dense = Dense(3, activation='tanh')  # Producing 3 channels (RGB) scaled to [-1, 1]

    def call(self, images, text, sigma, style_vector):
        # Apply sigma transformations
        sigma = self.sigma_ffn(sigma)

        # Create mask and encode text and style
        text_mask = create_padding_mask(text)
        text = self.text_style_encoder(text, style_vector, sigma)

        # Initial transformation of image data
        x = self.input_dense(images)

        # Pass through encoder layers
        h1 = self.enc1(x, sigma)
        h2 = self.pool(h1)

        h2 = self.enc2(h2, sigma)
        h2, _ = self.enc3(h2, text, sigma, text_mask)
        h3 = self.pool(h2)

        h3 = self.enc4(h3, sigma)
        h3, _ = self.enc5(h3, text, sigma, text_mask)
        x = self.pool(h3)

        # Apply attention layers
        x = self.att_dense(x)
        for att_layer in self.att_layers:
            x, att = att_layer(x, text, sigma, text_mask)

        # Decoding with skip connections
        x = self.upsample(x) + self.skip_conv3(h3)
        x = self.dec3(x, sigma)

        x = self.upsample(x) + self.skip_conv2(h2)
        x = self.dec2(x, sigma)

        x = self.upsample(x) + self.skip_conv1(h1)
        x = self.dec1(x, sigma)

        # Final output layer to produce image
        output = self.output_dense(x)
        return output, att  # Only return image output and attention (no pen lifts)


def loss_fn(eps, score_pred, abar, bce=None):
    score_loss = tf.reduce_mean(tf.reduce_sum(tf.square(eps - score_pred), axis=-1))
    return score_loss


class InvSqrtSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Ensure step is float32 to avoid type issues
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)