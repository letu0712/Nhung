import pyrebase

config = {
    "apiKey": "AIzaSyC03tBFCHdZKDaJTTIZRPJcNbw4hplci74",
    "authDomain": "embedded-project-2c61c.firebaseapp.com",
    "databaseURL": "https://embedded-project-2c61c.firebaseio.com",
    "databaseURL": "https://embedded-project-2c61c-default-rtdb.firebaseio.com/", 
    "projectId": "embedded-project-2c61c",
    "storageBucket": "embedded-project-2c61c.appspot.com",
    "messagingSenderId": "419272253129",
    "appId": "1:419272253129:web:c5374158344dbc7931f718",
    "measurementId": "G-ZLM4Y3NS9T"
}

firebase = pyrebase.initialize_app(config)
# storage = firebase.storage()
# auth = firebase.auth()

# image = "image.png"


# #Upload image
# storage.child(image).put(image)

# #Get URL of image
# email = "user1@gmail.com"
# password = "123456"
# user = auth.sign_in_with_email_and_password(email, password)
# url = storage.child(image).get_url(user["idToken"])
# print(url)

database = firebase.database()

data = {"name" : "25 23:12:12 2022"}
database.child("images").set(data)
