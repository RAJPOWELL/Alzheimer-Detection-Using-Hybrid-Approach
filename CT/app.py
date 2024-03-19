from flask import Flask, render_template, request, redirect, url_for
import random

app = Flask(__name__)

# Define question bank with 30 questions
question_bank = [
    {"question": "What is 2 + 2?", "answer": "4"},
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Who wrote 'Romeo and Juliet'?", "answer": "Shakespeare"},
    {"question": "What is the square root of 16?", "answer": "4"},
    {"question": "What is the chemical symbol for water?", "answer": "H2O"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the powerhouse of the cell?", "answer": "Mitochondria"},
    {"question": "What is the tallest mammal?", "answer": "Giraffe"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "Who is known as the 'Father of Computers'?", "answer": "Charles Babbage"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "What is the freezing point of water in Celsius?", "answer": "0"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"question": "What is the formula for the area of a circle?", "answer": "πr^2"},
    {"question": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
    {"question": "What is the chemical symbol for oxygen?", "answer": "O"},
    {"question": "Who wrote 'To Kill a Mockingbird'?", "answer": "Harper Lee"},
    # Add more questions here
]

# Shuffle question bank to randomize question order
random.shuffle(question_bank)

# Define variables for tracking user progress and score
current_question_index = 0
score = 0
question_limit = 10
timer_duration = 7  # Time limit for each question in seconds

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_test():
    global current_question_index, score
    current_question_index = 0
    score = 0
    return redirect(url_for('question'))

@app.route('/question')
def question():
    global current_question_index
    if current_question_index < min(len(question_bank), question_limit):
        question = question_bank[current_question_index]['question']
        return render_template('question.html', question=question, question_number=current_question_index+1)
    else:
        return redirect(url_for('result'))

@app.route('/answer', methods=['POST'])
def answer():
    global current_question_index, score
    user_answer = request.form['answer']
    if user_answer.lower() == question_bank[current_question_index]['answer'].lower():
        score += 1
    current_question_index += 1
    return redirect(url_for('question'))

@app.route('/result')
def result():
    global score
    if score >= 27:
        result_message = "You do not have Alzheimer's disease."
    else:
        result_message = "You may have Alzheimer's disease. Please consult a doctor for further evaluation."
    return render_template('result.html', score=score, result_message=result_message)

if __name__ == '__main__':
    app.run(debug=True)
