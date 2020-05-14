from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[4]

print(train_images.shape)
print(len(train_labels))
print(train_labels)
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()