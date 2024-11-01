import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import string
import pickle
import os
from tensorflow.image import resize


def explin(min, max, L):
    return tf.exp(tf.linspace(tf.math.log(min), tf.math.log(max), L))


def get_beta_set():
    beta_set = 0.02 + explin(1e-5, 0.4, 60)
    return beta_set


def show_image(img, name='', show_output=True, scale=1):
    # Display the generated image
    plt.figure(figsize=(scale, scale))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
    if name:
        plt.savefig('./' + name + '.png', bbox_inches='tight')
    if show_output:
        plt.show()
    else:
        plt.close()


def get_alphas(batch_size, alpha_set):
    alpha_indices = tf.random.uniform([batch_size, 1], maxval=len(alpha_set) - 1, dtype=tf.int32)
    lower_alphas = tf.gather_nd(alpha_set, alpha_indices)
    upper_alphas = tf.gather_nd(alpha_set, alpha_indices + 1)
    alphas = tf.random.uniform(lower_alphas.shape, maxval=1) * (upper_alphas - lower_alphas)
    alphas += lower_alphas
    alphas = tf.reshape(alphas, [batch_size, 1, 1])
    return alphas


def standard_diffusion_step(xt, eps, beta, alpha, add_sigma=True):
    x_t_minus1 = (1 / tf.sqrt(1 - beta)) * (xt - (beta * eps / tf.sqrt(1 - alpha)))
    if add_sigma:
        x_t_minus1 += tf.sqrt(beta) * tf.random.normal(xt.shape)
    return x_t_minus1


def new_diffusion_step(xt, eps, beta, alpha, alpha_next):
    x_t_minus1 = (xt - tf.sqrt(1 - alpha) * eps) / tf.sqrt(1 - beta)
    x_t_minus1 += tf.random.normal(xt.shape) * tf.sqrt(1 - alpha_next)
    return x_t_minus1


def run_batch_inference(model, beta_set, text, style, tokenizer=None, time_steps=480, diffusion_mode='new', show_every=None, show_samples=True, path=None):
    if isinstance(text, str):
        text = tf.constant([tokenizer.encode(text)+[1]])
    elif isinstance(text, list) and isinstance(text[0], str):
        text = tf.constant([tokenizer.encode(i)+[1] for i in text])

    bs = text.shape[0]
    L = len(beta_set)
    alpha_set = tf.math.cumprod(1 - beta_set)
    x = tf.random.normal([bs, time_steps, img_width, img_height, 3])  # Start with random noise images

    for i in range(L-1, -1, -1):
        alpha = alpha_set[i] * tf.ones([bs, 1, 1, 1, 1])
        beta = beta_set[i] * tf.ones([bs, 1, 1, 1, 1])
        a_next = alpha_set[i-1] if i > 1 else 1.0
        model_out, att = model(x, text, tf.sqrt(alpha), style)
        x = new_diffusion_step(x, model_out, beta, alpha, a_next)  # Update diffusion with image output

        if show_every is not None and i in show_every:
            plt.imshow(att[0][0])  # Visualize attention if needed
            plt.show()

    if show_samples:
        for i in range(bs):
            plt.imshow((x[i] + 1) / 2)  # Scale to [0, 1] for viewing
            plt.show()

    return x.numpy()



def pad_img(img, width, height):
    pad_len = width - img.shape[1]
    padding = np.full((height, pad_len, 1), 255, dtype=np.uint8)
    img = np.concatenate((img, padding), axis=1)
    return img


def pad_img_for_utils(img, width, height):
    img = tf.image.grayscale_to_rgb(img) if img.shape[-1] == 1 else img
    img = resize(img, (height, width))
    return img




def preprocess_data(path, max_text_len, img_width, img_height):
    with open(path, 'rb') as f:
        ds = pickle.load(f)

    images, texts, samples = [], [], []
    for item in ds:
        text, sample = item  # Adjust to match your dataset structure if needed

        # Load and pad the image (assuming `sample` is the image data)
        if sample.shape[1] < img_width:
            sample = pad_img(sample, img_width, img_height)  # Pad image to (img_height, img_width, 1)

        # Ensure the image is in the correct shape and format
        if sample.shape[:2] != (img_height, img_width):  # Check only height and width
            sample = resize(sample, (img_height, img_width))  # Resize to (img_height, img_width)

        # If grayscale, add channel dimension
        if sample.shape[-1] != 1:
            sample = tf.expand_dims(sample, axis=-1)  # Make sure shape is (img_height, img_width, 1)

        images.append(sample)

        # Process text and pad it
        if len(text) < max_text_len:
            zeros_text = np.zeros((max_text_len - len(text),))
            text = np.concatenate((text, zeros_text))
            texts.append(text)

        samples.append(sample)  # Append the processed image to samples

    images = np.array(images).astype('int8') # Convert list of images to a NumPy array
    texts = np.array(texts).astype('int32')  # Convert list of text arrays to NumPy
    samples = np.array(samples).astype('int8')  # Ensure samples have a consistent shape

    return images, texts, samples


def create_dataset(images, texts, samples, style_extractor, batch_size, buffer_size):
    # Convert images to dataset format
    images_dataset = tf.data.Dataset.from_tensor_slices(images)

    # Generate style vectors based on the samples using the style extractor
    samples_dataset = tf.data.Dataset.from_tensor_slices(samples).batch(batch_size)
    style_vectors = np.zeros((0, 1280))  # Adjust based on the output dimension of `style_extractor`
    for count, sample_batch in enumerate(samples_dataset):
        style_vec = style_extractor(sample_batch)
        style_vec = style_vec.numpy()
        if count == 0:
            style_vectors = np.zeros((0, style_vec.shape[1], 1280))
        style_vectors = np.concatenate((style_vectors, style_vec), axis=0)
    style_vectors = style_vectors.astype('float32')

    # Create a dataset of style vectors and texts
    style_vectors_dataset = tf.data.Dataset.from_tensor_slices(style_vectors)
    texts_dataset = tf.data.Dataset.from_tensor_slices(texts)

    # Zip images, texts, and style vectors into one dataset
    dataset = tf.data.Dataset.zip((images_dataset, texts_dataset, style_vectors_dataset))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



class Tokenizer:
    def __init__(self):
        self.tokens = {}
        self.chars = {}
        self.text = '_' + string.ascii_letters + string.digits + '.?!,\'\"- '
        self.numbers = np.arange(2, len(self.text) + 2)
        self.create_dict()
        self.vocab_size = len(self.text) + 2

    def create_dict(self):
        for char, token in zip(self.text, self.numbers):
            self.tokens[char] = token
            self.chars[token] = char
        self.chars[0], self.chars[1] = ' ', '<end>'

    def encode(self, text):
        tokenized = []
        for char in text:
            if char in self.text:
                tokenized.append(self.tokens[char])
            else:
                tokenized.append(2)  # Unknown character is '_'
        tokenized.append(1)  # End of sentence character
        return tokenized

    def decode(self, tokens):
        if isinstance(tokens, tf.Tensor):
            tokens = tokens.numpy()
        text = [self.chars[token] for token in tokens]
        return ''.join(text)
