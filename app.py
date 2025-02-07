# Importing modules
import numpy as np
import streamlit as st
import cv2
import pandas as pd
import random

from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import base64
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

auth_manager = SpotifyClientCredentials(
    client_id="b17735824bcb422092458fbc3d68334b",
    client_secret="b1c94404bdd846259e31b491dcd845ec"
)
spotify = spotipy.Spotify(auth_manager=auth_manager)

def get_spotify_link(song_name, artist_name):
    try:
        # Search for the track on Spotify
        query = f"track:{song_name} artist:{artist_name}"
        results = spotify.search(q=query, type="track", limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            return track['external_urls']['spotify']
        else:
            return None
    except Exception as e:
        print(f"Error fetching Spotify link: {e}")
        return None

df = pd.read_csv(r"C:\Users\myraj\OneDrive\Desktop\Emotion-based-music-recommendation-system-main\muse_v3.csv")


df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name','emotional','pleasant','link','artist']]
print(df)

df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
print(df)

df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(list):

    data = pd.DataFrame()

    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
             data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)

    elif len(list) == 2:
        times = [30,20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':    
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':             
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])

    elif len(list) == 3:
        times = [55,20,15]
        for i in range(len(list)): 
            v = list[i]          
            t = times[i]

            if v == 'Neutral':              
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':               
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':             
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:      
                data = pd.concat([df_sad.sample(n=t)])


    elif len(list) == 4:
        times = [30,29,18,9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral': 
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':              
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':              
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':               
                data =pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:              
               data = pd.concat([df_sad.sample(n=t)])
    else:
        times = [10,7,6,5,2]
        for i in range(len(list)):           
            v = list[i]         
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':           
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'fear':           
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'happy':          
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)])

    print("data of list func... :",data)
    return data

def pre(l):

    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    print("Processed Emotions:", result)
    if not result:  # Check if result is empty
        random_emotion = random.choice(["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"])
        result.append(random_emotion)

    # result = [item for items, c in Counter(l).most_common()
    #           for item in [items] * c]

    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
            print(result)
    print("Return the list of unique emotions in the order of occurrence frequency :",ul)
    return ul
    




model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))


model.load_weights(r"C:\Users\myraj\OneDrive\Desktop\Emotion-based-music-recommendation-system-main\model.h5")


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

print("Loading Haarcascade Classifier...")
face = cv2.CascadeClassifier(r"C:\Users\myraj\OneDrive\Desktop\Emotion-based-music-recommendation-system-main\haarcascade_frontalface_default.xml")
if face.empty():
    print("Haarcascade Classifier failed to load.")
else:
    print("Haarcascade Classifier loaded successfully.")

page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>"
            , unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>"
            , unsafe_allow_html=True)

col1,col2,col3 = st.columns(3)

list = []
with col1:
    pass
with col2:
    if st.button('SCAN EMOTION(Click here)'):

        count = 0
        list.clear()

        while True:

            ret, frame = cap.read()
            if not ret:
                break
            
            face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")          
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)          
            count = count + 1

            for (x, y, w, h) in faces:               
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)               
                roi_gray = gray[y:y + h, x:x + w]               
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)             
                prediction = model.predict(cropped_img)             
                max_index = int(np.argmax(prediction))

                list.append(emotion_dict[max_index])

                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)              
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

            if cv2.waitKey(10) & 0xFF == ord('s'):
                break
            if count >= 20:
                break
        cap.release()
        cv2.destroyAllWindows()

        list = pre(list)
        if len(list)==0:
            list.append("Neutral")
        
        st.success(" emotions successfully detected") 
        

with col3:
    pass

new_df = fun(list)
st.write("")

st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>"
            , unsafe_allow_html=True)

st.write("---------------------------------------------------------------------------------------------------------------------")

try:
  
    for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(30)):
        spotify_link = get_spotify_link(n, a)
        if spotify_link:
            # Show Spotify link if available
            st.markdown(f"<h4 style='text-align: center;'><a href='{spotify_link}' target='_blank'>{i+1}. {n}</a></h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center; color: grey;'><i>{a}</i></h5>", unsafe_allow_html=True)
        else:
            # Fallback to the normal link
            st.markdown(f"<h4 style='text-align: center;'><a href='{l}' target='_blank'>{i+1}. {n}</a></h4>", unsafe_allow_html=True)
            st.markdown(f"<h5 style='text-align: center; color: grey;'><i>{a}</i></h5>", unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")

except:
    pass


