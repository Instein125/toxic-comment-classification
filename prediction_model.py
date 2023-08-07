import tensorflow as tf
import pandas as pd

threshold = [0.65397644, 0.84165853, 0.55169016, 0.14203131, 0.4851602, 0.23365118]
model = tf.keras.models.load_model("C:/Users/dell/OneDrive/Documents/Deep learning project/Trained Model/BiLSTM")

def convert_to_binary(y_pred, thresholds):
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    y_pred_df = pd.DataFrame(y_pred, columns=class_names)
    binary_predictions_df = y_pred_df.copy()
    for i, col in enumerate(y_pred_df.columns):
        binary_predictions_df[col] = (y_pred_df[col] >= thresholds[i])
    return binary_predictions_df

def prediction(data):
    results = model.predict(data)
    results = convert_to_binary(results, threshold)
    result_dict = results.to_dict(orient='records')

    
    return result_dict