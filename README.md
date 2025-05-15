# Signify

Signify is a real-time, AI-powered translator that combines American Sign Language (ASL) recognition, facial emotion detection, and speech/text translation in a unified cross-platform application. Designed to bridge communication gaps across different modalities, Signify runs both on mobile (React Native + Expo) and web platforms (React.js), with backend inference powered by FastAPI and TensorFlow.

The ASL recognition system uses Mediapipe’s Hand Landmarks to extract 21 keypoints from video frames, which are normalized and fed into machine learning models trained with both Random Cut Forest (RCF) and Multilayer Perceptron (MLP) architectures. The final MLP model is deployed using TensorFlow.js for client-side inference, enabling real-time predictions without requiring server-side video streaming.

Facial emotion detection is implemented using Mediapipe FaceMesh landmarks, feeding into a lightweight MLP model trained on the FER-2013 dataset. This browser-based approach improves responsiveness and protects user privacy by avoiding raw video transmission.

Speech input and output are powered by Google Cloud APIs, enabling bi-directional communication through speech-to-text and text-to-speech features. The system supports dynamic translation using Google Translate and also integrates the OpenAI API for more natural conversational output in specific language pairs.

The project includes real-time UI feedback, speech history tracking, and accessible design optimized for both desktop and mobile environments. Emotion and gesture predictions are displayed alongside live video, and spoken words are vocalized using the Web Speech API.

Signify is fully modular and scalable, with all models containerized and served via FastAPI endpoints. The entire system is designed with performance, privacy, and accessibility in mind.

Attached below are  demo videos of our application:

Signify- Cross Platform Mobile Application (demo): 
https://youtube.com/shorts/7OFRZthaKcI?feature=share

Signify-Web Application(single user demo):
https://youtu.be/GmYCFh-sC_I

Signify-Web Application(multi user demo):
https://youtu.be/FTU7O3Csyx4

Signify-Web Application(normal and mute person demo): 
https://youtu.be/5hmrs29DyGY
