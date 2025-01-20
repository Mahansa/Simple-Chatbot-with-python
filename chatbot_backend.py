from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chatbot_response(input_text, chat_history_ids=None):
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    with torch.no_grad():
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

def search_internet(query):
    api_key = "Use your own API" 
    url = "Your-URL{query}&api_key={api_key}"
    
    try:
        response = requests.get(url)
        search_results = response.json()
        if "organic_results" in search_results:
            answer = search_results["organic_results"][0]["snippet"]
            return answer
        else:
            return "Sorry i can't find anything on internet. Pls try again."
    except Exception as e:
        return f"Can't find anything on in internet: {str(e)}"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("input")
    chat_history_ids = None
    if user_input:
        if "?" in user_input:
            internet_answer = search_internet(user_input)
            return jsonify({"response": internet_answer})

        try:
            response, chat_history_ids = chatbot_response(user_input, chat_history_ids)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500
    return jsonify({"error": "Input tidak valid"}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
