import utils
import tensorflow as tf
import custom_extension

if __name__ == "__main__":
    #initialize global model

    X_train, y_train, X_test, y_test, label_encoder = utils.read_data()

    input_shape = X_train.shape[1:]
    nb_classes = len(label_encoder.classes_)
    class_weights = utils.get_class_weights(y_train)

    # y_test = utils.convert_to_categorical(y_test, nb_classes)
    print(y_test)
    print("-----------------")
    print(len(y_test))
    print("-----------------")
    print(y_test[0])
    print("-----------------")
    
    y_test = utils.convert_to_categorical(y_test, nb_classes)
    print(y_test)
    print("-----------------")
    print(len(y_test))
    print("-----------------")
    print(y_test[0])
    print("-----------------")

    X_test, y_test = custom_extension.sample_test(X_test, y_test, 1000)
    
    print(y_test)
    print("-----------------")
    print(len(y_test))
    print("-----------------")
    print(y_test[0])
    print("-----------------")
    test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))
    print(len(test_batched))