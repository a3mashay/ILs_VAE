#VAE class
class VAEWithPropertyPrediction(Model):
    def __init__(self, encoder, decoder, property_predictor, **kwargs):
        super(VAEWithPropertyPrediction, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.property_predictor = property_predictor
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        property_prediction = self.property_predictor(z)
        return [reconstructed, property_prediction, z_mean, z_log_var]  # Added z_mean & z_log_var for KL loss
    def get_config(self):
        return {}
# Example of a costume loss function
def custom_loss(y_true, y_pred):
    z_mean = y_pred[2]
    z_log_var = y_pred[3]
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    reconstruction_loss = MeanSquaredError()(y_true[0], y_pred[0])
    property_prediction_loss = MeanSquaredError()(y_true[1], y_pred[1])
    total_loss = reconstruction_loss + property_prediction_loss + 0.001 * kl_loss  # KL weight factor
    return total_loss

# KL loss can be included implicitly
output_dim = 2
property_predictor = build_property_predictor(latent_dim, output_dim)
vae_with_prop_pred = VAEWithPropertyPrediction(encoder, decoder, property_predictor)
vae_with_prop_pred.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=custom_loss)
