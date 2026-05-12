import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime

name=input("Enter the name of aspirant from keyboard:  ")

temp_image=np.zeros((200,1500,3), dtype="uint8")

# Define the text and its properties
text = "Hi " + name + "  Project_02"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White color in BGR
thickness = 2
line_type = cv2.LINE_AA

# Add text to the image
cv2.putText(temp_image, text, (50, 50), font, font_scale, font_color, thickness, line_type)


# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Add date and time to the image
cv2.putText(temp_image, current_datetime, (50, 150), font, font_scale, font_color, thickness, line_type)


# Display the image with text
cv2.imshow('Image with Text', temp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load unhealthy images
path1="D:\\MOOC\\PROJECT_02\\DATASET\\RS_Rust 2633.JPG"

# read an image
image=cv2.imread(path1)


#Display an image
cv2.imshow("Unhealthy",image)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# Split an image
B,G,R=cv2.split(image)


#Flatten channels
B=B.flatten()
G=G.flatten()
R=R.flatten()

# Create dataset
df=pd.DataFrame()


df["Red"]=R
df["Green"]=G
df["Blue"]=B


#Transform colour model
HSV=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split an image
H,S,V=cv2.split(HSV)



H=H.flatten()
S=S.flatten()
V=V.flatten()


df["H"]=H
df["S"]=S
df["V"]=V



LAB=cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
L,A,B1=cv2.split(LAB)


L=L.flatten()
A=A.flatten()
B1=B1.flatten()

df["L"]=L
df["A"]=A
df["B1"]=B1

df["Label"]=1
df.shape



# Load healthy images
path1="D:\\MOOC\\PROJECT_02\\DATASET\\healthy_01.jpg"

# read an image
image=cv2.imread(path1)


#Display an image
cv2.imshow("Healthy",image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Split an image
B,G,R=cv2.split(image)


#Flatten channels
B=B.flatten()
G=G.flatten()
R=R.flatten()

# Create dataset
df1=pd.DataFrame()


df1["Red"]=R
df1["Green"]=G
df1["Blue"]=B


#Transform colour model
HSV=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split an image
H,S,V=cv2.split(HSV)



H=H.flatten()
S=S.flatten()
V=V.flatten()


df1["H"]=H
df1["S"]=S
df1["V"]=V



LAB=cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
L,A,B1=cv2.split(LAB)


L=L.flatten()
A=A.flatten()
B1=B1.flatten()

df1["L"]=L
df1["A"]=A
df1["B1"]=B1


df1["Label"]=2
print(df1.shape)

# Add two pandas dataframe
data=pd.concat([df,df1], axis=0)
data.shape


y=data["Label"]
x=data.drop("Label", axis=1)


# Divide dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.20)


# Creating and fitting the KNN classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report



k = 7  # Choose the number of neighbors (you can experiment with different values)
knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(X_train, y_train)


# Making predictions on the test set
y_pred = knn_classifier.predict(X_test)



# Performance metrics of model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Displaying classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))


def load_images_from_folder(folder_path):
    image_list = []
    image_names = []

    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can add more image extensions if needed)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', ".JPG")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                # Append the image and its name to the lists
                image_list.append(image)
                image_names.append(filename)

    return image_list, image_names


# Please give path to your dataset from your laptop or desktop
folder_path = 'D:\\MOOC\\PROJECT_02\\DATASET'

images, image_names = load_images_from_folder(folder_path)

for image, image_name in zip(images, image_names):
    r,w,c=image.shape
    # Split an image
    B, G, R = cv2.split(image)

    # Flatten channels
    B = B.flatten()
    G = G.flatten()
    R = R.flatten()

    # Create dataset
    df1 = pd.DataFrame()

    df1["Red"] = R
    df1["Green"] = G
    df1["Blue"] = B

    # Transform colour model
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split an image
    H, S, V = cv2.split(HSV)

    H = H.flatten()
    S = S.flatten()
    V = V.flatten()

    df1["H"] = H
    df1["S"] = S
    df1["V"] = V

    LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B1 = cv2.split(LAB)

    L = L.flatten()
    A = A.flatten()
    B1 = B1.flatten()

    df1["L"] = L
    df1["A"] = A
    df1["B1"] = B1

    y_pred = knn_classifier.predict(df1)
    print(y_pred)
    import numpy as np

    mask = y_pred.reshape(r, w)




    white_pixels = np.sum(mask == 1)
    Total_number_pixels = r * w

    Red_pixels_percentage = (white_pixels / Total_number_pixels) * 100

    print("Red_pixels_percentage:{}".format(Red_pixels_percentage))

    if Red_pixels_percentage >= 40:
        print("The leaf is Unhealthy")

        cv2.imshow(image_name, image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        # Please give path to folder where you want store the classified dataset
        path1 = "D:\\MOOC\\PROJECT_02\\UNHEALTHY\\"

        cv2.imwrite(path1 + image_name + ".png", image)



    else:
        print("The leaf is Healthy")
        cv2.imshow(image_name, image)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        # Please give path to folder where you want store the classified dataset
        path2 = "D:\\MOOC\\PROJECT_02\\HEALTHY\\"

        cv2.imwrite(path2 + image_name + ".png", image)


#Dont make any changes to below code
temp_image=np.zeros((400,2000,3), dtype="uint8")

# Define the text and its properties
text = name + " Your Project_02 is completed."
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White color in BGR
thickness = 2
line_type = cv2.LINE_AA

# Add text to the image
cv2.putText(temp_image, text, (50, 50), font, font_scale, font_color, thickness, line_type)

text ="Please visit UNHEALTHY and HEALTHY folders"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # White color in BGR
thickness = 2
line_type = cv2.LINE_AA

# Add text to the image
cv2.putText(temp_image, text, (50, 150), font, font_scale, font_color, thickness, line_type)


# Get current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Add date and time to the image
cv2.putText(temp_image, current_datetime, (50, 200), font, font_scale, font_color, thickness, line_type)


# Display the image with text
cv2.imshow('Image with Text', temp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()