import tensorflow as tf
import numpy as np
import time
import argparse
import os
import utils_offline  # Assuming utils_offline provides necessary utilities for offline data
import nn_offline  # Assuming nn_offline provides the offline model and necessary functions


@tf.function
def train_step(images, text, style_vectors, glob_args):
    model, alpha_set, train_loss, optimizer = glob_args
    alphas = utils_offline.get_alphas(len(images), alpha_set)
    images = tf.cast(images, tf.float32)  # Ensure images are float32 for compatibility
    eps = tf.random.normal(tf.shape(images))  # Adjust shape to match images

    x_perturbed = tf.sqrt(alphas) * images
    x_perturbed += tf.sqrt(1 - alphas) * eps

    with tf.GradientTape() as tape:
        score, att = model(x_perturbed, text, tf.sqrt(alphas), style_vectors, training=True)
        loss = nn_offline.loss_fn(eps, score, alphas)  # Adjust loss function, removing pen lifts

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    return loss  # Return loss for logging


def train(dataset, iterations, model, optimizer, alpha_set, log_dir, print_every=1000, save_every=10000):
    s = time.time()
    train_loss = tf.keras.metrics.Mean()

    # Set up TensorBoard writer
    summary_writer = tf.summary.create_file_writer(log_dir)

    for count, (images, text, style_vectors) in enumerate(dataset.repeat(5000)):
        # Prepare the training arguments, removing any dependency on pen_lifts
        glob_args = model, alpha_set, train_loss, optimizer
        current_loss = train_step(images, text, style_vectors, glob_args)  # Pass images instead of strokes

        # Log loss and print every `print_every` iterations
        if optimizer.iterations % print_every == 0:
            avg_loss = train_loss.result().numpy()
            print("Iteration %d, Loss %f, Time %ds" % (optimizer.iterations, avg_loss, time.time() - s))

            # Log to TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar("Loss", avg_loss, step=optimizer.iterations)

            train_loss.reset_states()

        # Save the model weights at `save_every` iterations
        if (optimizer.iterations + 1) % save_every == 0:
            save_path = './weights/model_step%d.h5' % (optimizer.iterations + 1)
            model.save_weights(save_path)

        # Stop training after reaching the specified number of iterations
        if optimizer.iterations > iterations:
            model.save_weights('./weights/model.h5')
            break

    summary_writer.close()  # Close the writer after training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', help='Number of training steps', default=120000, type=int)
    parser.add_argument('--batchsize', help='Batch size', default=96, type=int)
    parser.add_argument('--textlen', help='Text length during training', default=50, type=int)
    parser.add_argument('--width', help='Image width for resizing', default=1400, type=int)
    parser.add_argument('--warmup', help='Warmup steps', default=10000, type=int)
    parser.add_argument('--dropout', help='Dropout rate', default=0.0, type=float)
    parser.add_argument('--num_attlayers', help='Number of attention layers', default=2, type=int)
    parser.add_argument('--channels', help='Number of channels in the first layer', default=128, type=int)
    parser.add_argument('--print_every', help='Log every N iterations', default=100, type=int)
    parser.add_argument('--save_every', help='Save checkpoint every N iterations', default=1000, type=int)
    parser.add_argument('--log_dir', help='TensorBoard log directory', default='./logs', type=str)

    args = parser.parse_args()
    NUM_STEPS = args.steps
    BATCH_SIZE = args.batchsize
    MAX_TEXT_LEN = args.textlen
    WIDTH = args.width
    DROP_RATE = args.dropout
    NUM_ATTLAYERS = args.num_attlayers
    WARMUP_STEPS = args.warmup
    PRINT_EVERY = args.print_every
    SAVE_EVERY = args.save_every
    LOG_DIR = args.log_dir
    C1 = args.channels
    C2 = C1 * 3 // 2
    C3 = C1 * 2

    BUFFER_SIZE = 3000
    tokenizer = utils_offline.Tokenizer()
    beta_set = utils_offline.get_beta_set()
    alpha_set = tf.math.cumprod(1 - beta_set)

    style_extractor = nn_offline.StyleExtractor()
    model = nn_offline.DiffusionWriter(num_layers=NUM_ATTLAYERS, c1=C1, c2=C2, c3=C3, drop_rate=DROP_RATE)
    lr = nn_offline.InvSqrtSchedule(C3, warmup_steps=WARMUP_STEPS)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, clipnorm=100)

    path = './data/train_offline.p'  # Adjust path as needed for offline data
    images, texts, samples = utils_offline.preprocess_data(path, MAX_TEXT_LEN, WIDTH, 96)
    dataset = utils_offline.create_dataset(images, texts, samples, style_extractor, BATCH_SIZE, BUFFER_SIZE)

    train(dataset, NUM_STEPS, model, optimizer, alpha_set, LOG_DIR, PRINT_EVERY, SAVE_EVERY)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found, running on CPU.")
    main()
