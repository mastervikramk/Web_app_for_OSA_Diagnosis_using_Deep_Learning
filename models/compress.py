import tensorflow as tf

def compress_tflite_model(input_model_path, output_model_path):
    # Load the existing TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_saved_model(input_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    # Save the quantized model
    with open(output_model_path, 'wb') as f:
        f.write(tflite_quantized_model)

    print("Quantized model compressed and saved successfully.")

# Example usage
input_model_path = 'C:/Users/vikram kumar/Desktop/web_app/Web_Development/osa_app/models/model1.tflite'
output_model_path = 'model1_compressed.tflite'
compress_tflite_model(input_model_path, output_model_path)
