def build_property_predictor(latent_dim, output_dims): 
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(128, activation='relu')(latent_inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_dims, activation='linear')(x) 

    model = Model(latent_inputs, outputs, name='property_predictor')
    return model
