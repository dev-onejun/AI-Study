def check_pytorch():
    import torch

    x = torch.rand(5, 3)
    print(x)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print('there is no available gpu')

def check_sklearn():
    import sklearn

    sklearn.show_versions()

def check_tf_version():
    import tensorflow as tf

    print(tf.__version__)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def check_tf():
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

def check_tf_metal():
    import tensorflow as tf

    cifar = tf.keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64)

if __name__=='__main__':
    check_pytorch()
    check_sklearn()
    check_tf_version()
    check_tf()
    check_tf_metal()
