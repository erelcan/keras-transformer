from keras_transformer.training.factories.loss_factory import create_loss_function

loss_info = {
    "type": "custom_crossentropy",
    "parameters": {
      "num_classes": 1001,
      "mask_value": 1000,
      "from_logits": False,
      "label_smoothing": 0.1
    }
}

loss_fn = create_loss_function(loss_info)

print(loss_fn)