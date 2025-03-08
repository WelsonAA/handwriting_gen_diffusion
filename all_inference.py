import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import nn
import argparse
import os
import preprocessing

def main():
    # Create a list of 200 lines of text strings
    text_lines = [
        "Life is beautiful.", "I love this place.", "Let's make memories.",
        "It's a great day.", "Time heals everything.", "Dream big, work hard.",
        "Happiness starts here.", "Never give up.", "Love conquers all.",
        "Stay strong, always.", "You are amazing.", "Live with purpose.",
        "Follow your heart.", "Embrace the journey.", "Trust your instincts.",
        "Never stop learning.", "Chase your dreams.", "Believe in yourself.",
        "The future's bright.", "Spread kindness always.", "Love is endless.",
        "Choose happiness daily.", "Stay true always.", "Life is a gift.",
        "Make today count.", "Success takes patience.", "Laugh more, worry less.",
        "Family is everything.", "Follow your passion.", "Find joy everywhere.",
        "Stay humble always.", "Never look back.", "Be the change.",
        "Seek new adventures.", "Keep moving forward.", "Enjoy the little things.",
        "Kindness changes lives.", "Stay positive always.", "Every day matters.",
        "Give with love.", "Be kind always.", "Cherish every moment.",
        "Love yourself first.", "Everything happens for a reason.", "Trust the process.",
        "Believe in miracles.", "Dreams do come true.", "Life is precious.",
        "Take it easy.", "Focus on the present.", "Breathe and relax.",
        "Appreciate simple things.", "Be proud always.", "Adventure awaits you.",
        "Every moment counts.", "Love makes everything better.", "The best is yet.",
        "Find peace within.", "Enjoy the ride.", "Keep it simple.",
        "Stay curious always.", "Smile more, worry less.", "You got this.",
        "Never stop dreaming.", "Enjoy the sunshine.", "Create your happiness.",
        "Always be grateful.", "Laugh until it hurts.", "Keep the faith.",
        "Be bold, be brave.", "Life is an adventure.", "Follow your dreams.",
        "You are unstoppable.", "Live without regrets.", "Be your best.",
        "Make every second count.", "Choose joy every day.", "Love is the answer.",
        "Take chances often.", "Smile at strangers.", "Always stay positive.",
        "Let love guide.", "Embrace new challenges.", "Live in harmony.",
        "Love your journey.", "Keep pushing forward.", "Your story matters.",
        "You are limitless.", "Find your passion.", "Always stay humble.",
        "Start with gratitude.", "Believe in yourself.", "Appreciate the moment.",
        "Dream it, do it.", "Stay true to yourself.", "You are enough.",
        "Be a light.", "Love what you do.", "Stay focused always.",
        "Celebrate small wins.", "Pursue your purpose.", "Always keep moving.",
        "You are incredible.", "Love your life.", "Appreciate what you have.",
        "Stay fearless always.", "Laugh often, love much.", "Make life meaningful.",
        "Never stop trying.", "Seek peace daily.", "Live with intention.",
        "Live for today.", "Find balance always.", "Believe in your dreams.",
        "Stay strong always.", "Dream, believe, achieve.", "Life is short.",
        "Keep chasing dreams.", "Love deeply, forgive quickly.", "Stay grounded always.",
        "Focus on your goals.", "Be happy now.", "Keep a positive mindset.",
        "Live with passion.", "Let it go.", "Believe in the impossible.",
        "Every day is a gift.", "Make it happen.", "Stay positive, stay strong.",
        "Keep striving forward.", "Don’t lose hope.", "Today is the day.",
        "Live without fear.", "Love without limits.", "Celebrate your victories.",
        "Take a deep breath.", "Keep dreaming big.", "Embrace new beginnings.",
        "Life is an opportunity.", "You’re doing great.", "Stay true to you.",
        "Enjoy the moment.", "The best is yet to come.", "Smile through challenges.",
        "Make it count.", "Go after it.", "Love makes life better.",
        "Seek what makes you happy.", "You are worthy.", "Life is a journey.",
        "Take risks often.", "Love overcomes all.", "Live with gratitude.",
        "Happiness is a choice.", "Your potential is endless.", "Keep pushing ahead.",
        "Everything is possible.", "You are a star.", "Keep the peace.",
        "Find joy in life.", "Love what you have.", "Stay strong in challenges.",
        "Stay the course.", "Seek happiness within.", "Life is an adventure.",
        "Make today special.", "Keep climbing higher.", "Let your heart guide.",
        "Believe in your power.", "The best is coming.", "Love every moment.",
        "Stay motivated always.", "Be a dreamer.", "Stay calm always.",
        "Believe in your strength.", "You are amazing.", "Keep shining bright.",
        "Live with courage.", "Let it be.", "Be proud of yourself.",
        "Create your own path.", "Choose positivity daily.", "Enjoy every moment.",
        "Keep working hard.", "Stay focused and strong.", "Embrace your uniqueness.",
        "Trust in your journey.", "Spread love everywhere.", "Make dreams happen.",
        "Life is an experience.", "Never stop believing.", "Chase happiness daily.",
        "Keep your head up.", "Inspire those around.", "Keep learning always.",
        "Live with love.", "You can do it.", "Seek your dreams.",
        "Let go of fear.", "The journey continues."
    ]

    # Ensure the 'paper_results' directory exists
    if not os.path.exists('my_results'):
        os.makedirs('my_results')

    # Parser setup for argument inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--textstring', help='The text you want to generate', default='I love Diffusion', type=str)
    parser.add_argument('--writersource', help="Path of the image of the desired writer", default="./assets/r06-412z-04.tif")
    parser.add_argument('--name', help="Path for generated image", default="./output/sample")
    parser.add_argument('--diffmode', help="What kind of y_t-1 prediction to use", default='standard', type=str)
    parser.add_argument('--show', help="Whether to show the sample", default=True, type=bool)
    parser.add_argument('--weights', help='The path of the loaded weights', default='./weights/model_step120000.h5', type=str)
    parser.add_argument('--seqlen', help='Number of timesteps in generated sequence', default=None, type=int)
    parser.add_argument('--num_attlayers', help='Number of attentional layers at lowest resolution', default=2, type=int)
    parser.add_argument('--channels', help='Number of channels at lowest resolution', default=128, type=int)

    args = parser.parse_args()

    # Process each text line from the list and save the results in the 'paper_results' directory
    for idx, line in enumerate(text_lines):
        args.textstring = line
        # Set output path to 'paper_results' directory
        args.name = f'my_results/sample_{idx + 1}.png'
        timesteps = len(args.textstring) * 16 if args.seqlen is None else args.seqlen
        timesteps = timesteps - (timesteps % 8) + 8  # Must be divisible by 8 due to downsampling layers

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus[0].name}")
        else:
            print("No GPU found, running on CPU.")
        if args.writersource is None:
            assetdir = os.listdir('./assets')
            sourcename = './assets/' + assetdir[np.random.randint(0, len(assetdir))]
        else:
            sourcename = args.writersource
        L = 60
        tokenizer = utils.Tokenizer()
        beta_set = utils.get_beta_set()
        alpha_set = tf.math.cumprod(1 - beta_set)

        C1 = args.channels
        C2 = C1 * 3 // 2
        C3 = C1 * 2
        style_extractor = nn.StyleExtractor()
        model = nn.DiffusionWriter(num_layers=args.num_attlayers, c1=C1, c2=C2, c3=C3)

        _stroke = tf.random.normal([1, 400, 2])
        _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
        _noise = tf.random.uniform([1, 1])
        _style_vector = tf.random.normal([1, 14, 1280])
        _ = model(_stroke, _text, _noise, _style_vector)
        # we have to call the model on input first
        model.load_weights(args.weights)

        writer_img = tf.expand_dims(preprocessing.read_img_for_inf(sourcename, 96), 0)
        style_vector = style_extractor(writer_img)
        utils.run_batch_inference(model, beta_set, args.textstring, style_vector,
                                  tokenizer=tokenizer, time_steps=timesteps, diffusion_mode=args.diffmode,
                                  show_samples=args.show, path=args.name)


if __name__ == '__main__':
    main()
