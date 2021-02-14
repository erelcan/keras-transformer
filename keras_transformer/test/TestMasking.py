import numpy as np
from keras.layers import Input, LSTM, Masking, Dense
from keras.models import Model


def test_masked_loss():
    max_sentence_length = 5
    num_of_features = 2
    input_tensor = Input(shape=(max_sentence_length, num_of_features))
    masked_input = Masking(mask_value=0)(input_tensor)
    output = LSTM(3, return_sequences=True)(masked_input)
    model = Model(input_tensor, output)
    model.compile(loss='mae', optimizer='adam')

    X = np.array([[[0, 0], [0, 0], [1, 0], [0, 1], [0, 1]],
                  [[0, 0], [0, 1], [1, 0], [0, 1], [0, 1]]])

    y_true = np.ones((2, max_sentence_length, 3))
    y_pred = model.predict(X)

    print(X)
    print(y_pred)
    print(y_true)

    unmasked_loss = np.abs(1 - y_pred).mean()
    masked_loss = np.abs(1 - y_pred[y_pred != 0]).mean()
    masked_loss *= 21 / 30
    # Observe that keras gets mean over all after setting  masked ones to masked value (zero).
    # Hence, it always divides the number of elements that exists originally!!

    print(y_pred[y_pred != 0])
    print(model.evaluate(X, y_true))
    print(masked_loss)
    print(unmasked_loss)

    masked_loss2 = np.array(np.abs(1 - y_pred[y_pred != 0]).tolist() + [0] * 9).mean()
    print(masked_loss2)
