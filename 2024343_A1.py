import struct
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE

#functions

#reading images and labels
def read_idx1(filename):
    with open(filename, 'rb') as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def read_idx3(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(
            f.read(), dtype=np.uint8
        ).reshape(num_images, rows, cols)
    return images

#filter images for 0,1,2 and make feature vectors
def filter_num(x, labels, images):
    num_images = []
    for i in range(0,60000):
        if (labels[i] == x):
            num_images.append(images[i])  

    return np.array(num_images)

def resize_images(images):
    new_images = []
    for i in range(0,len(images)):
        new_images.append(np.reshape(images[i],784))
    return np.array(new_images)/255

#mle estimates for mean and covariance
def mean_mle(samples):
    return sum(samples)/len(samples)

def cov_matrix_mle(samples,mean):
    var = []
    for i in range(0,len(samples)):
        y = np.reshape((samples[i]-mean),(784,1)) @ np.reshape((samples[i]-mean),(1,784))
        var.append(y)
    return np.array(sum(var))/len(samples)

#linear and quadratic discriminant computers
def compute_params_qda(mean,variance,prior):
    quadratic_term = (-0.5)*(np.linalg.inv(variance + 0.0001 * np.eye(784)))
    linear_term = mean @ (np.linalg.inv(variance + 0.0001 * np.eye(784)))
    constant_term =  (-0.5) * (mean @ (np.linalg.inv(variance + 0.0001 * np.eye(784))) @ mean + np.log(np.linalg.det(variance) if np.linalg.det(variance) > 0 else 1)) + np.log(prior)
    return (quadratic_term,linear_term,constant_term)

def compute_params_lda(mean,variance,prior):
    linear_term = mean @ (np.linalg.inv(variance + 0.0001 * np.eye(784)))
    constant_term =  (-0.5) * (mean @ (np.linalg.inv(variance + 0.0001 * np.eye(784))) @ mean) + np.log(prior)
    return (linear_term,constant_term)

def linear_discriminant(point,params):
    return params[0] @ point + params[1]  

def quadratic_discriminant(point,params):
    return point @ params[0] @ point + params[1] @ point + params[2]

#classification decision algorithm
def classify(test,mle):
    cl = 0
    cq = 0
    params = {}
    for i in range(3):
        params[i] = {}
        params[i]["qda"] = compute_params_qda(mle[i][0], mle[i][1], 1)
        params[i]["lda"] = compute_params_lda(mle[i][0], (mle[0][1] + mle[1][1] + mle[2][1])/3,1)

    num = 0
    for i in range(3):
        for j in test[i]:
            #linear discriminant
            linear = [0,0,0]
            quadratic = [0,0,0]
            for k in range(3):
                linear[k] = linear_discriminant(j,params[k]["lda"])
                quadratic[k] = quadratic_discriminant(j,params[k]["qda"])

            l_pred = 0
            q_pred = 0
            for l in range(3):
                if (linear[l] == max(linear)):
                    l_pred = l
                if (quadratic[l] == max(quadratic)):
                    q_pred = l

            if (l_pred == i):
                cl+=1
            if (q_pred == i):
                cq+=1

            num+=1
            print(f"Test #{num} Actual = {i}\n")
            print(f"Linear Discriminant\nClass Zero:{linear[0]}\tOne:{linear[1]}\tTwo:{linear[2]}\nLDA Prediction = {l_pred}\n")
            print(f"Quadratic Discriminant\nClass Zero:{quadratic[0]}\tOne:{quadratic[1]}\tTwo:{quadratic[2]}\nLDA Prediction = {q_pred}\n")

    print(f"\nAccuracy\nLDA Accuracy = {cl/300}\nQDA Accuracy = {cq/300}")

#read MNIST images and labels
labels = read_idx1(r"C:\Users\Mayank\Downloads\SML_A1_DATASET\train-labels.idx1-ubyte")
images = read_idx3(r"C:\Users\Mayank\Downloads\SML_A1_DATASET\train-images.idx3-ubyte")

#conversion to feature columns
images = resize_images(images)

train = {}
test = {}
mle = {}

for i in range(3):
    #filter for class 0,1,2 in corresponding iteration
    class_data = filter_num(i,labels,images)

    #random permutation of number, seed for reproducibility
    np.random.seed(14)
    random_data = np.random.permutation(len(class_data))

    #apply this to data and select 0-99 & 100-199 for train and test, these are random since permutation was random
    train[i] = class_data[random_data[:100]]
    test[i] = class_data[random_data[100:200]]

    #compute mle estimates using train data
    mean = mean_mle(train[i])
    cov = cov_matrix_mle(train[i], mean)
    mle[i] = (mean,cov)

    print(f"Determinant of Covariance Matrix is {np.linalg.det(cov)}, so it is singular")

#run classification, report accuracy and discriminant values
classify(test,mle)

#t-SNE plots for train and test data
X_train = np.vstack([train[0], train[1], train[2]])
X_test  = np.vstack([test[0],  test[1],  test[2]])

y_train = np.array(
    [0]*len(train[0]) +
    [1]*len(train[1]) +
    [2]*len(train[2])
)

y_test = np.array(
    [0]*len(test[0]) +
    [1]*len(test[1]) +
    [2]*len(test[2])
)
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

split = len(X_train)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto"
)

X_embedded = tsne.fit_transform(X_all)

X_train_2d = X_embedded[:split]
X_test_2d  = X_embedded[split:]

plt.figure(figsize=(7, 6))

plt.scatter(
    X_train_2d[:,0], X_train_2d[:,1],
    c=y_train, cmap="tab10",
    marker="o", alpha=0.7, label="Train"
)

plt.scatter(
    X_test_2d[:,0], X_test_2d[:,1],
    c=y_test, cmap="tab10",
    marker="x", alpha=0.7, label="Test"
)

plt.colorbar(label="Digit (0,1,2)")
plt.legend()
plt.title("t-SNE Plot")
plt.show()
