import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.layers import  RandomFlip, RandomRotation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
])

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
def create_vit_layer(input_shape):
    num_patches = (input_shape[0] // 16) * (input_shape[1] // 16)  # Assuming 16x16 patches
    projection_dim = 64
    num_heads = 8
    transformer_layers = 8
    transformer_units = [projection_dim * 2, projection_dim]  # MLP layers size in the transformer

    inputs = layers.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size=16)(inputs)
    # Linear projection of patches
    encoded_patches = layers.Dense(units=projection_dim)(patches)

    # Positional encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches += position_embedding

    # Transformer layers
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(units=transformer_units[0], activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(units=projection_dim)(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    features = layers.GlobalAveragePooling1D()(representation)

    return models.Model(inputs=inputs, outputs=features)

def create_vit_lstm_model(input_shape, frames, num_classes):
    video_input = layers.Input(shape=(frames, *input_shape))
    
    # Data Augmentation
    augmented_frames = layers.TimeDistributed(data_augmentation)(video_input)
    
    # ViT with more complexity
    vit_layer = create_vit_layer(input_shape)
    processed_frames = layers.TimeDistributed(vit_layer)(augmented_frames)
    
    # Bidirectional LSTM with Attention
    lstm_out = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(processed_frames)
    attention = layers.Attention()([lstm_out, lstm_out])
    lstm_out = layers.GlobalAveragePooling1D()(attention)
    
    # Fully connected layer
    dense_out = layers.Dense(128, activation='relu')(lstm_out)
    dense_out = layers.Dropout(0.5)(dense_out)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(dense_out)
    
    # Define the model
    model = models.Model(inputs=video_input, outputs=output)
    
    return model