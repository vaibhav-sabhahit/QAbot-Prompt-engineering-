import pandas as pd
from scipy.spatial import distance
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import numpy as np
from numpy.linalg import norm
from nltk.corpus import stopwords
import os
import openai
from flask import Flask, request, jsonify

#openai.api_base = os.environ.get('END_POINT_NAME', "https://cog-yrpdocqrwlfw2.openai.azure.com/")
openai.api_base = "define API end point"
openai.api_type = os.environ.get('API_TYPE', "azure")
openai.api_version = os.environ.get('API_VERSION', "2023-03-15-preview")
#deployment_name = os.environ.get('deployment_name', 'chat')
deployment_name = "ChatGPT-turbo-test"
#defineAPI key
port = int(os.environ.get("PORT", 8080))


def chatgpt_call(test_prompt, df):
    # ('bert-base-nli-mean-tokens')
    model = SentenceTransformer('all-mpnet-base-v2')
    prompt_embedding = model.encode(test_prompt)
    score = []
    for j, i in enumerate(df['bert_embedding']):
        similarity = np.dot(prompt_embedding, i) / \
            (norm(prompt_embedding)*norm(i))
        score.append(similarity)
    df['score'] = score
    df.sort_values(by='score', ascending=False, inplace=True)
    df = df.reset_index(drop=True)
    context = df['text'][0]
    print('Context:', context)
    memprompt = 'You are an AI assistant for customer service'
    text_prompt = 'Context:'+context+'Question:'+test_prompt
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[{"role": "system", "content": ''}, {
            "role": "user", "content": text_prompt}],
        temperature=0.8,
        max_tokens=1000)
    print('Question:'+test_prompt+'\n'+'\n'+'Answer:'+response.to_dict()
          ["choices"][0]["message"]['content'])
    return(response.to_dict()
           ["choices"][0]["message"]['content'])


app = Flask(__name__)


@ app.route("/message", methods=["POST"])
def process_message():
    input_json = request.get_json()
    df = pd.read_pickle('prototype_embedding.pkl')
    text = input_json["message"]
    chatgpt_response = chatgpt_call(text, df)
    return jsonify({'response': chatgpt_response}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
