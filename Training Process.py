# Training process example
model_checkpoint_callback = ModelCheckpoint(
    filepath='/vae_model',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_format="tf"
)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)
history = vae_with_prop_pred.fit(
    X_train,
    [X_train, Y_train],
    epochs=100,
    batch_size=256,
    validation_data=(X_test, [X_test, Y_test]),
    callbacks=[model_checkpoint_callback, early_stopping_callback]
)