import streamlit as st
import pandas as pd
from streamlit_image_select import image_select
import torch
from model.model import CNN
import cv2
import torchvision.transforms as transforms
import plotly.express as px

@st.cache_data
def load_model():
    model = CNN()
    model = torch.load('./model/model.pth').to('cpu')
    model.eval()
    return model


def main():
        model  = load_model()
        st.title('Cat Classifier Report')

        st.write("""Cats, scientifically known as Felis catus, are beloved domesticated mammals that have been companions
                to humans for thousands of years. They come in a wide variety of breeds, each with distinctive characteristics,
                including coat patterns, colors, body shapes, and sizes. Cats are known for their agility, curiosity,
                and independent nature, making them popular pets around the world.

                \nUsing CNN to Predict Cat's Breed:
                Convolutional Neural Networks (CNNs) are a type of deep learning model specifically designed for image analysis tasks.
                They have proven to be highly effective in tasks like image classification, where the goal is to categorize images into different classes.""")

        st.markdown("""
                ## Most common cat breeds
                - Persian
                - Russian Blue
                - Siamese
                - Turkish Angora
                - Siberian
                - Maine Coon
                - Bengal
                """)


        st.markdown("""
                ### Base Architecture - ResNet18:
                - The ResNet18 architecture consists of several convolutional and pooling layers, followed by fully connected layers at the end for classification.
                - It introduces the concept of *residual blocks*, allowing the network to learn deeper representations and avoiding the vanishing gradient problem in deep networks.
                - The residual blocks contain *skip connections* that add the original input to the output of the convolutional layers. This helps the network learn differences instead of having to learn all features from scratch.

                ### Modifications in the Model:
                - Pre-trained weights of ResNet18 are loaded into the initialized model. These weights are trained on the ImageNet dataset, which means the network already has general knowledge of patterns and features in images.
                - The original fully connected layer is replaced with a custom sequence of layers that adapt to the specific classification problem.
                - A linear layer is added that reduces the dimensionality of the features from the original fully connected layer to 512 features. This allows the model to fit better to the specific characteristics of the problem being addressed.
                - A ReLU activation function is applied after the linear layer to introduce non-linearity in the model.
                - A dropout layer is added to reduce the risk of overfitting. The dropout layer randomly deactivates a percentage of units during training, helping to prevent excessive reliance on certain features.
                - Another linear layer is added that ultimately maps the 512 features to the output classes. The softmax function is used on the output to obtain class probabilities.
                """) 

        st.header("Metrics")


        st.write('The training loop had 30 epochs, and was evaluated the Loss and Accuracy.')


        st.components.v1.html(open('./metrics/loss.html', 'r', encoding='utf-8').read(), width=700, height=500)
        st.components.v1.html(open('./metrics/accuracy.html', 'r', encoding='utf-8').read(), width=700, height=500)

        st.write('The model obtained an accuracy of 83.5% for the test set.')
        st.components.v1.html(open('./metrics/conf_mat.html', 'r', encoding='utf-8').read(), width=700, height=500)


        report = pd.read_csv('./metrics/report.csv')
        class_report = report.iloc[0:7, :]
        class_report.columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        st.dataframe(class_report, use_container_width=True)

        st.markdown("""
                ## Conclusions:
                - In general the model performs well, however, certain classes are deficient in accuracy and recall.
                - Classes 0,1,3,4,5 have good accuracy and recall metrics, however, class 0 has low recall.
                - Classes 2 and 6 have low accuracy and low recall, indicating that the model does not work well for these classes.
                """)

        st.write("""
                ### Recomendations:
                When reviewing classes 0, 2, 6 in the confusion matrix it is observed that these classes are cross-predicted between them, this makes sense since the cat species they represent are very similar physically, it is recommended to use data augmentation techniques and increase the images of these classes so that the model can differentiate them more accurately.
                """)
     
        st.header('Model implementation')

        img = image_select(
        label="Select a cat",
        images=['./img_example/siames.jpg','./img_example/angora.jpg','./img_example/persian.jpg',
                './img_example/bengali.jpg','./img_example/maine.jpg','./img_example/rusian.jpg','./img_example/siberian.jpg'
        ],
        captions=["Siamese", 'Angora','Persian','Bengali', 'Maine Coon','Rusian','Siberian']
        )

        img = cv2.imread(img)
        img = cv2.resize(img, (256, 256))

        tensor_img = transforms.ToTensor()(img)
        labels = ['angora', 'bengali', 'mainecoon', 'persian', 'russianblue', 'siamese', 'siberian']
        pred = model.predict(tensor_img.unsqueeze(1).permute(1,0,2,3))
        idx = torch.argmax(pred).item()
        st.markdown(f'Label predicted: **{labels[idx]}** with **{torch.max(pred).item():.2f}** % confidence.')

        fig = px.bar(x=labels, y= pred.detach().numpy()[0], color = pred.detach().numpy()[0], color_continuous_scale='viridis', labels={'x': 'Cat Breed', 'y': 'Prediction Probability'},
                title="Cat Breed Classification Predictions", range_y= [0,1.1])

        st.plotly_chart(fig)

        st.write('If you want to review the source code of the imaging, preprocessing, model creation and training and its implementation you can click on the following link. [Source Code](https://github.com/luisemmanuelavilaleon/Cat-Classifier.git)')

if __name__ == "__main__":
      main()