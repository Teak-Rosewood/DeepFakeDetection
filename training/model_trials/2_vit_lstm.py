import tensorflow as tf
from tensorflow.keras import layers, models

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

# Epoch 4/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 297s 3s/step - accuracy: 0.9314 - loss: 0.2779
# Epoch 5/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 298s 3s/step - accuracy: 0.9366 - loss: 0.2491
# Epoch 6/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 297s 3s/step - accuracy: 0.9275 - loss: 0.2659
# Epoch 7/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 297s 3s/step - accuracy: 0.9340 - loss: 0.2581
# Epoch 8/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 297s 3s/step - accuracy: 0.9479 - loss: 0.2253
# Epoch 9/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 297s 3s/step - accuracy: 0.9325 - loss: 0.2573
# Epoch 10/10
# 100/100 ━━━━━━━━━━━━━━━━━━━━ 298s 3s/step - accuracy: 0.9372 - loss: 0.2440