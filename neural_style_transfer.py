import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import MaxPooling2D

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
CONTENT_IMG_PATH = "./input/cat.jpg"
STYLE_IMG_PATH = "./input/starry_night.jpg"
GENERATED_IMG_PATH = "./output/generated_img.jpg"
IMG_SIZE = (400, 300)
NUM_COLOR_CHANNELS = 3
ALPHA = 10
BETA = 40
NOISE_RATIO = 0.6
CONTENT_LAYER_INDEX = 13
STYLE_LAYER_INDICES = [1, 4, 7, 12, 17]
STYLE_LAYER_COEFFICIENTS = [0.2, 0.2, 0.2, 0.2, 0.2]
NUM_ITERATIONS = 500
LEARNING_RATE = 2
VGG_IMAGENET_MEANS = np.array([103.939, 116.779, 123.68]).reshape((1, 1, 3)) # In blue-green-red order
LOG_GRAPH = False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def create_output_dir():
    """Create output dir if it does not exist."""
    cwd = os.getcwd()
    output_dir_path = os.path.join(cwd, "output")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

def load_img(path, size, color_means):
    """Load image from path, preprocess it, and return the image."""
    img = cv2.imread(path)
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32")
    img -= color_means
    img = np.expand_dims(img, axis=0)

    return img

def save_img(img, path, color_means):
    """Save image to path after postprocessing."""
    img += color_means
    img = np.clip(img, 0, 255)
    img = img.astype("uint8")
    cv2.imwrite(path, img)

def create_noisy_img(img, noise_ratio):
    """Add noise to img and return it."""
    noise = np.random.uniform(-20, 20, (img.shape[0], img.shape[1], img.shape[2], img.shape[3])).astype("float32")
    noisy_img = noise_ratio * noise + (1 - noise_ratio) * img

    return noisy_img

def create_output_tensors(input_variable, content_layer_index, style_layer_indices):
    """
    Create output tensors, using a pretrained Keras VGG19-model.
    Return tensors for content and style layers.
    """
    vgg_model = VGG19(weights="imagenet", include_top=False)
    layers = [l for l in vgg_model.layers]

    x = layers[1](input_variable)
    x_content_tensor = x
    x_style_tensors = []
    if 1 in style_layer_indices:
        x_style_tensors.append(x)

    for i in range(2, len(layers)):
        # Use layers from vgg model, but swap max pooling layers for average pooling
        if type(layers[i]) == MaxPooling2D:
            x = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            x = layers[i](x)

        # Store appropriate layer outputs
        if i == content_layer_index:
            x_content_tensor = x
        if i in style_layer_indices:
            x_style_tensors.append(x)

    return x_content_tensor, x_style_tensors

def content_cost(a_c, a_g):
    """Return a tensor representing the content cost."""
    _, n_h, n_w, n_c = a_c.shape

    return (1/(4 * n_h * n_w * n_c)) * tf.reduce_sum(tf.square(tf.subtract(a_c, a_g)))

def style_cost(a_s_layers, a_g_layers, style_layer_coefficients):
    """Return a tensor representing the style cost."""
    style_cost = 0
    for i in range(len(a_s_layers)):
        # Compute gram matrix for the activations of the style image
        a_s = a_s_layers[i]
        _, n_h, n_w, n_c = a_s.shape
        a_s_unrolled = tf.reshape(tf.transpose(a_s), [n_c, n_h*n_w])
        a_s_gram = tf.matmul(a_s_unrolled, tf.transpose(a_s_unrolled))

        # Compute gram matrix for the activations of the generated image
        a_g = a_g_layers[i]
        a_g_unrolled = tf.reshape(tf.transpose(a_g), [n_c, n_h*n_w])
        a_g_gram = tf.matmul(a_g_unrolled, tf.transpose(a_g_unrolled))

        # Compute style cost for the current layer
        style_cost_layer = (1/(4 * n_c**2 * (n_w* n_h)**2)) * tf.reduce_sum(tf.square(tf.subtract(a_s_gram, a_g_gram)))

        style_cost += style_cost_layer * style_layer_coefficients[i]
    
    return style_cost

def total_cost(content_cost, style_cost, alpha, beta):
    """Return a tensor representing the total cost."""
    return alpha * content_cost + beta * style_cost

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
create_output_dir()

# Load, resize, and preprocess content and style images
content_img = load_img(CONTENT_IMG_PATH, IMG_SIZE, VGG_IMAGENET_MEANS)
style_img = load_img(STYLE_IMG_PATH, IMG_SIZE, VGG_IMAGENET_MEANS)

# Create initial generated image, this is the starting point for the optimization process
generated_img_init = create_noisy_img(content_img, NOISE_RATIO)

# Create tensorflow variable that will be used as an input to the network.
# This variable will later be assigned generated_img_init and trained.
input_var = tf.Variable(content_img, dtype=tf.float32, expected_shape=(None, None, None, NUM_COLOR_CHANNELS), name="input_var")

# Create output tensors for the activations of the content and style layers,
# using a Keras VGG19-model pretrained on the ImageNet dataset.
x_content, x_styles = create_output_tensors(input_var, CONTENT_LAYER_INDEX, STYLE_LAYER_INDICES)

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

# Use the Keras session instead of creating a new one
with K.get_session() as sess:
    sess.run(tf.variables_initializer([input_var]))

    # Extract the layer activations for content and style images
    a_content = sess.run(x_content, feed_dict={K.learning_phase(): 0})
    sess.run(input_var.assign(style_img))
    a_styles = sess.run(x_styles, feed_dict={K.learning_phase(): 0})

    # Define the cost function
    J_content = content_cost(a_content, x_content)
    J_style = style_cost(a_styles, x_styles, STYLE_LAYER_COEFFICIENTS)
    J_total = total_cost(J_content, J_style, ALPHA, BETA)

    # Log the graph. To display use "tensorboard --logdir=log".
    if LOG_GRAPH:
        writer = tf.summary.FileWriter("log", sess.graph)
        writer.close()

    # Assign the generated random initial image as input
    sess.run(input_var.assign(generated_img_init))

    # Create the training operation
    train_op = optimizer.minimize(J_total, var_list=[input_var])
    sess.run(tf.variables_initializer(optimizer.variables()))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Train the generated image
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    for i in range(NUM_ITERATIONS):
        sess.run(train_op)

        if (i%20) == 0:
            print(
                "Iteration: " + str(i) +
                ", Content cost: " + "{:.2e}".format(sess.run(J_content)) +
                ", Style cost: " + "{:.2e}".format(sess.run(J_style)) +
                ", Total cost: " + "{:.2e}".format(sess.run(J_total))
            )

            # Save the generated image
            generated_img = sess.run(input_var)[0]
            save_img(generated_img, GENERATED_IMG_PATH, VGG_IMAGENET_MEANS)

    # Save the generated image
    generated_img = sess.run(input_var)[0]
    save_img(generated_img, GENERATED_IMG_PATH, VGG_IMAGENET_MEANS)
