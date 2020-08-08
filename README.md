# potato-carrot-classifier

#OVERVIEW
The above classifier was made with the intentions of working with google images as the input data to the classifier
It is a simple classifier made by using libraries such as numpy,pandas etc...
It uses just 10 images as of now because of the lack of processing power but feel free to use more images for improving the model
I will put up all the images on github

#HOW TO GET THE DATA
Choose a Classifier first like i have chosen potato vs carrot
Search in google images for potato and open the javascript console
Start scrolling down the images keeping the console open until the images you want
Go to download_images_google folder and type the lines one by one from the file image_pull.js into the console 
A file will be then downloaded named urls3.txt in your downloads folder which will have all the images urls upto which you have scrolled
Then run gen_images.py to download all the images to the folder you specify to
The way you run the command is as follows:-
python gen_images.py --urls "<url>" --output "<output>"
Once you have the images in the folder you need to make all the images of the same dimention
  
#HOW TO CHOOSE THE DIMENTION
What i did was i downloaded all the images that i needed for my train set and stored them in thier respective folders by following the above steps
Then i took the average width and height of all the images in potatoes folder and did the same for carrots folder
Then i took the average of these 2 and got the averages and got the final width and height
The number of channels remained the same

#HOW TO RUN THE MODEL
python train.py can be used for running the model

The following code has no prediction function as i feel its not going to be too accurate because of lack of computational power
A lot of manual data cleaning was needed to be done before training

It is although a very good method incase you dont have enough images for a model or if you dont get the dataset on kaggle or somewhere else!

[The data can be accessed over here](https://drive.google.com/drive/folders/1SaJAFZjWMzR8ucrFYTAqouNiQqKTUJQR?usp=sharing)
