import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import nn
import time

BATCH_SIZE = 96
BUFFER_SIZE = 3000
MAX_SEQ_LEN = 480
MAX_TEXT_LEN = 50
WIDTH = 800
HEIGHT = 96
L = 60

style_extractor = nn.StyleExtractor()
path = './data/train_strokes.p'
strokes, texts, samples = utils.preprocess_data(path, MAX_TEXT_LEN, MAX_SEQ_LEN, WIDTH, HEIGHT)
dataset = utils.create_dataset(strokes, texts, samples, 
                               style_extractor, BATCH_SIZE, BUFFER_SIZE)

testpath = './data/test_strokes.p'
teststrokes, testtexts, testsamples = utils.preprocess_data(testpath, MAX_TEXT_LEN, MAX_SEQ_LEN, WIDTH, HEIGHT)
testdataset = utils.create_dataset(teststrokes, testtexts, testsamples, 
                               style_extractor, BATCH_SIZE, BUFFER_SIZE)

ckpt_path = './weights/'

t = utils.Tokenizer()
beta_set = utils.get_beta_set()
alpha_set = tf.math.cumprod(1-beta_set)

model = nn.DiffusionWriter(num_layers=2, c1=128, c2=192, c3=256, drop_rate=0.0, num_heads=6)
lr = nn.InvSqrtSchedule(256, warmup_steps=10000)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, clipnorm=100)

for a,b,c in dataset.take(1):
    model_out, pen_lifts, att = model(a[:, :, :2], b, tf.random.uniform([len(a), 1, 1]), c)
    plt.imshow(att[0][0])
model.count_params()

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
train_loss = tf.keras.metrics.Mean()

@tf.function
def train_step(x, pen_lifts, text, style_vectors):
    alphas = utils.get_alphas(len(x), alpha_set)
    eps = tf.random.normal(tf.shape(x))
    x_perturbed = tf.sqrt(alphas) * x 
    x_perturbed += tf.sqrt(1 - alphas) * eps
    
    with tf.GradientTape() as tape:
        score, pl_pred, att = model(x_perturbed, text, tf.sqrt(alphas), style_vectors, training=True)
        loss = nn.loss_fn(eps, score, pen_lifts, pl_pred, alphas, bce)
        
    gradients = tape.gradient(loss, model.trainable_variables)  
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return score, att

def train(dataset, iterations, print_progress_every=1000):
    s = time.time()
    for count, (strokes, text, style_vectors) in enumerate(dataset.repeat(1000)):
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2:]
        model_out, att = train_step(strokes, pen_lifts, text, style_vectors)
        
        if optimizer.iterations%print_progress_every==0:
            print("Iteration %d, Loss %f, Time %ds" % (optimizer.iterations, train_loss.result(), time.time()-s))
            train_loss.reset_states()

        if (optimizer.iterations+1) %10000==0:
            save_path = ckpt_path + 'model_step%d.h5' % (optimizer.iterations+1)
            model.save_weights(save_path)
            
        if optimizer.iterations > iterations:
            model.save_weights('./weights/model.h5')
            break

train(dataset, 60000)
