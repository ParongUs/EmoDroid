import time
import random
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from streamlit_autorefresh import st_autorefresh
import io


# Load models and utilities outside of the function calls to avoid reloading them repeatedly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

def capture_and_save_owner_embedding():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to RGB and detect faces
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        boxes, _ = mtcnn.detect(pil_img)
        
        if boxes is not None:
            # Extract the face with the largest area (most likely the main subject)
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            largest_face_index = areas.index(max(areas))
            largest_face = mtcnn.extract(pil_img, [boxes[largest_face_index]], None)
            
            # Save the embedding of the largest detected face
            owner_embedding = resnet(largest_face).detach()
            torch.save(owner_embedding, 'owner_embedding.pt')
            
            print("Owner's face embedding captured and saved.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return owner_embedding

def is_owner_present():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    owner_embedding = torch.load('owner_embedding.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    owner_detected = False  # Flag to indicate if the owner is detected
    start_time = time.time()

    try:
        while time.time() - start_time < 5:  # Check for 5 seconds
            ret, frame = cap.read()
            if not ret:
                continue  # Skip this loop if frame is not captured correctly

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Detect faces
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is not None:
                faces = mtcnn.extract(pil_img, boxes, None)
                embeddings = resnet(faces).detach()

                for embedding in embeddings:
                    # Calculate distance to the owner's embedding
                    distance = (embedding - owner_embedding).norm().item()
                    if distance < 0.6:  # threshold for recognition, tune based on your dataset
                        owner_detected = True
                        break

            if owner_detected:
                break  # Stop checking if owner is already detected

            # Display the frame (optional)
            cv_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imshow('Webcam', cv_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally: 
        cap.release()
        cv2.destroyAllWindows()

    return owner_detected

def continuous_emotion_detection():
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    emotions = []
    try:
        for _ in range(5):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Convert frame to grayscale for face detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    # Extract the face ROI (Region of Interest)
                    face_roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2RGB)  # Use original frame to get RGB ROI

                    # Perform emotion analysis on the face ROI
                    try:
                        # Perform emotion analysis on the face ROI
                        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                        # Determine the dominant emotion
                        emotion = result[0]['dominant_emotion']
                        emotions.append(emotion)
                    except Exception as e:
                        print(f"Error in emotion analysis: {e}")

        # Count occurrences of each emotion
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}

        # Find the emotion with the highest count
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        return dominant_emotion

    finally:
        # Release the capture and close all windows
        cap.release()
        

    else:
        return None

def play_game(user_choice):
    choices = ["taş", "kağıt", "makas"]
    emo_choice = random.choice(choices)
    
    if user_choice == emo_choice:
        return f"Ben de {emo_choice} seçtim! Berabere!", "emo_face.png"
    
    win_map = {"taş": "makas", "kağıt": "taş", "makas": "kağıt"}
    if win_map[user_choice] == emo_choice:
        return f"Sen {user_choice} seçtin, ben {emo_choice}... Kazandın! Tebrikler!", "emo_lose.png"
    else:
        return f"Sen {user_choice} seçtin, ben {emo_choice}... Ben kazandım! Hahaha!", "emo_win.png"

def process_audio(audio_bytes):
    r = sr.Recognizer()
    audio_file = io.BytesIO(audio_bytes)
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language="tr-TR")
        return text
    except:
        return ""


def chat(user_input):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    if user_input.lower() == 'quit':
        return "Beep boop! Bye!"

    # Encode user input and generate a response
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Ensure no previous context is included if it's leading to duplication of input in response
    chat_output_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    
    # Decode the response
    bot_output = tokenizer.decode(chat_output_ids[0], skip_special_tokens=True)

    # Optionally strip the repeated input from the response
    if user_input.lower() in bot_output.lower():
        bot_output = bot_output[len(user_input):].strip()

    return bot_output


def main_page():
    # Auto-refresh every 30 seconds for proactive dialogue
    st_autorefresh(interval=30000, key="proactive_refresh")

    if "current_face" not in st.session_state:
        st.session_state["current_face"] = "emo_face.png"
    
    if "last_interaction_time" not in st.session_state:
        st.session_state["last_interaction_time"] = time.time()

    if "game_mode" not in st.session_state:
        st.session_state["game_mode"] = False

    st.markdown("# EMO Robot: Seninle Sohbet Etmek İçin Sabırsızlanıyorum!")
    st.image(st.session_state["current_face"], use_column_width=True)
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Proactive dialogue check
    current_time = time.time()
    if current_time - st.session_state["last_interaction_time"] > 30:
        proactive_messages = [
            "Orada mısın? Sıkıldım...",
            "Hey! Benimle biraz konuşur musun?",
            "Yeni bir şeyler yapalım mı?",
            "Seni izliyorum... 👀",
            "Hadi oyun oynayalım!"
        ]
        chosen_msg = random.choice(proactive_messages)
        st.session_state["chat_history"].append(("🤖", chosen_msg))
        st.session_state["last_interaction_time"] = current_time

    # Voice interface
    st.write("Sesle konuşmak için mikrofonu kullan:")
    audio_bytes = audio_recorder(text="Kayıt Başlat", icon_size="2x")
    
    user_input = ""
    if audio_bytes:
        with st.spinner("Seni dinliyorum..."):
            user_input = process_audio(audio_bytes)
            if user_input:
                st.info(f"Dediğin: {user_input}")

    # Text input
    text_input = st.text_input("Veya buraya yaz:", key="text_input")
    send_button = st.button("Gönder", key="send_button")

    final_input = user_input if user_input else text_input

    if (send_button or user_input) and final_input:
        st.session_state["last_interaction_time"] = time.time()
        
        # Game logic trigger
        if "oyun oynamak istiyorum" in final_input.lower() or "taş kağıt makas" in final_input.lower():
            st.session_state["game_mode"] = True
            bot_output = "Harika! Taş mı, Kağıt mı, yoksa Makas mı? Seçimini yap!"
            st.session_state["current_face"] = "emo_face.png"
        elif st.session_state["game_mode"] and any(x in final_input.lower() for x in ["taş", "kağıt", "makas"]):
            choice = [x for x in ["taş", "kağıt", "makas"] if x in final_input.lower()][0]
            bot_output, face = play_game(choice)
            st.session_state["current_face"] = face
            st.session_state["game_mode"] = False # Reset after one round
        else:
            bot_output = chat(final_input)
            st.session_state["current_face"] = "emo_face.png"

        st.session_state["chat_history"].append(("👶", final_input))
        st.session_state["chat_history"].append(("🤖", bot_output))
        
        # Check for emotion detection opportunity
        if len(st.session_state["chat_history"]) % 5 == 0:
            emotion_response = detect_emotion()
            if emotion_response:
                st.session_state["chat_history"].append(("🤖", emotion_response))

    # Sidebar to display chat history
    st.sidebar.markdown("# Sohbet Geçmişi")
    for speaker, message in st.session_state["chat_history"]:
        st.sidebar.text(f"{speaker}: {message}")


def main():
    if "owner_present_checked" not in st.session_state:
        # Initially check if the owner is present
        owner_present = is_owner_present()
        st.session_state["owner_present_checked"] = True
    else:
        owner_present = True 

    if owner_present:
        # Directly go to the main page if owner is recognized
        main_page()
    else:
        if st.button("Get to Know Me", key="get_to_know"):
            capture_and_save_owner_embedding()
            st.markdown("Now that we've met, let's chat!")
            if st.button("Lets Chat!", key="lets_chat"):
                main_page()
        else:
            st.markdown("# Sorry, I don't talk to strangers.")
            st.image("emo_stranger.png")


if __name__ == "__main__":
    main()
