import os
import cv2
from fer import FER
from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier
from together import Together
from googletrans import Translator

# Initialize the Together client with your API key
together_client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
translator = Translator()

# Initialize facial emotion recognition model
emotion_detector = FER()

# Sample student data for the learning speed classifier
X = [[30, 5], [25, 3], [40, 7], [15, 1], [20, 2]]  # Mock data (task_completion_time, mistakes_made)
y = ['slow', 'medium', 'slow', 'fast', 'fast']  # Class labels (slow, medium, fast)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)


# Function to query LLama model
def query_agent(user_question, learning_speed, emotion):
    try:
        response = together_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an AI Robotic tutor on STEM content, specifically computer science for young people aged 10-14. Tailor content based on the student's learning speed: {learning_speed} and their emotional state: {emotion}."""
                },
                {
                    "role": "user",
                    "content": f"{user_question}"
                }
            ],
            max_tokens=1500,
            temperature=0.7,
            top_p=1,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>"]
        )
        
        if 'choices' in response and len(response.choices) > 0:
            return response.choices[0]['message']['content']
        else:
            return "No valid response from the API."
    except Exception as e:
        return f"An error occurred: {e}"

# Function to translate text to Swahili
def translate_to_swahili(text):
    try:
        translated = translator.translate(text, dest='sw')
        return translated.text
    except Exception as e:
        return f"Translation error: {e}"

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = ""
    learning_speed = ""
    emotion = ""
    
    if request.method == 'POST':
        user_question = request.form.get('question')
        language = request.form.get('language', 'en')
        
        # Get student learning speed from classifier (replace this with real data)
        new_student_data = [[35, 6]]  # Mock data: Task completion time and mistakes
        learning_speed = clf.predict(new_student_data)[0]
        
        # Capture emotions from the webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        emotion_results = emotion_detector.detect_emotions(frame)
        cap.release()
        
        if emotion_results:
            dominant_emotion = max(emotion_results[0]['emotions'], key=emotion_results[0]['emotions'].get)
            emotion = dominant_emotion
        else:
            emotion = "neutral"
        
        # Generate personalized response based on user query, learning speed, and emotion
        if user_question:
            answer = query_agent(user_question, learning_speed, emotion)
            
            # Translate answer to Swahili if needed
            if language == 'sw':
                answer = translate_to_swahili(answer)
    
    return render_template('index.html', answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
