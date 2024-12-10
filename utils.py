# Function to capture pic from webcam
import cv2
from io import BytesIO
from PIL import Image
import base64

def capture_image():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if frame was captured
        if not ret:
            print("Error: Failed to capture image.")
            cap.release()
            cv2.destroyAllWindows()
            return None

        # Display the frame to the user
        cv2.imshow("Press Space to Capture Image", frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # Check if the Space bar (32) was pressed
        if key == 32:  # Space key
            # Convert the frame (BGR format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL image
            image = Image.fromarray(frame_rgb)

            # Convert the image to bytes
            image_bytes = BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes.seek(0)

            # Convert image to base64
            image_base64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

            # Release resources and close windows
            cap.release()
            cv2.destroyAllWindows()

            return image_base64

        # Exit if the user presses 'q'
        if key == ord('q'):
            print("Exiting capture.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        

# Function for text to speech conversion
from gtts import gTTS
from playsound import playsound

def tts(text): # Online service
    mp3_fp = BytesIO()
    # Create a gTTS object
    speech = gTTS(text, lang="en", tld = "co.uk", slow = False)
    # Save the speech
    speech.save("output.mp3")
    # Play the speech
    playsound("output.mp3")
    
    """
    from gtts import gTTS
    from io import BytesIO

    mp3_fp = BytesIO()
    tts = gTTS('hello', lang='en')
    tts.write_to_fp(mp3_fp)

    # Load `mp3_fp` as an mp3 file in
    # the audio library of your choice
    """
    return 1

# stt
import speech_recognition as sr

def stt():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)  # Reduces noise
        recognizer.energy_threshold = 4000
        print("Listening... Speak into the microphone.")
        
        try:
            # Listen for the audio
            audio = recognizer.listen(source)
            print("Processing your speech...")

            # Transcribe using Google's free Web Speech API
            text = recognizer.recognize_google(audio)
            print("You said:", text)
            return text

        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech.")
        except sr.RequestError as e:
            print(f"Could not request results from the service; {e}")

if __name__ == "__main__":
    text = stt()
    # print(tts(text))
