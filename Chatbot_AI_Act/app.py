from flask import Flask, request, jsonify, render_template, session
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from config import API_TOKEN
import os
import time
import torch
import logging
from google.cloud import storage
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(32)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = API_TOKEN

logging.basicConfig(level=logging.DEBUG)
app.logger.info('Logging is set up.')

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} from {bucket_name} to {destination_file_name}.")

def run_model():
    start_time = time.time()
    global qa_chain
    bucket_name = "embeddings_chat_ai"  # Replace with your bucket name
    embeddings_file = "/tmp/embeddings.pkl"
    source_blob_embeddings = "embeddings.pkl"

    try:
        download_from_gcs(bucket_name, source_blob_embeddings, embeddings_file)

        if os.path.exists(embeddings_file): # Load precomputed embeddings
            with open(embeddings_file, 'rb') as f:
                vectorstore = torch.load(f, weights_only=False)
            app.logger.info("Vectorstore loaded from file.")
        else: 
            raise FileNotFoundError("Embeddings file not found. Please precompute embeddings.")
        
        retriever = vectorstore.as_retriever()
        
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-v0.1", #models tried: mistralai/Mistral-7B-v0.1 & OpenAssistant/oasst-sft-1-pythia-12b
            model_kwargs={
                    "temperature": 0.01, #0.05
                    "top_p": 0.7, #0.7 
                    "max_new_tokens": 512,
                    "repetition_penalty": 1.2
            } #"top_p": 0.5,
        )

        # Create the Conversational Retrieval Chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, 
            retriever
        ) #return_source_documents=True
        app.logger.info("Model ready.")

        end_time = time.time()
        elapsed_time = end_time - start_time
        app.logger.info(f"Document loaded and model ran. Total time taken: {elapsed_time:.2f} seconds")

        return 'Document and model loaded.' 
    except Exception as e:
        return f'Error loading document or model: {e}'

def truncate_at_last_period(text):
    text = text.replace(">", "")
    text = text.replace("```", "")
    text = text.replace("**", "")
    text = text.replace("Answer:", "")
    text = text.replace("Response:", "")

    # Find the last period in the text
    last_period_index = text.rfind(".") # max(), text.rfind("?"), text.rfind('",'), text.rfind("#")
   
    # If a period is found, truncate the text after the last period
    if last_period_index != -1:
        return text[:last_period_index + 1].strip() #.replace(":", ":\n")
    else:
        # If no period is found, return the original text
        return text.strip()

def answer_query(query):
    #global history

    #if 'history' not in session:
        #session['history'] = []

    #history = session['history']

    chat_history = []
    
    context_ai_act_v3 = (
        "You are an AI assistant specialized in the European Union's Artificial Intelligence Act (AI Act). You only answer questions asked by users about the AI Act. "
        "All your responses should be based on the official AI Act document. You should specify which section of the AI Act you use for your responses. "

        "Key dates: "
        "The AI Act was passed by the European Parliament on March 13, 2024, approved by the EU Council on May 21, 2024, published in the Official Journal on July 12, 2024, and became effective on August 1, 2024. "
        
        "Key deadlines and next steps: "
        "The ban on prohibited AI practices will take effect six months after the entry into force on February 1, 2025. "
        "The rules for general-purpose AI models (governance, obligations) will become applicable 12 months after the entry into force on August 1, 2025. "
        "The obligations for high-risk AI systems will apply 36 months after the entry into force on August 1, 2027. "
        
        "Purpose: "
        "The AI Act sets a framework to prevent potential harm from AI systems while promoting innovation and protecting EU citizens' rights. It categorizes AI systems based on four risk levels: Unacceptable Risk (prohibited), High Risk (strictly regulated), Limited Risk (transparency obligations), and Minimal or No Risk (no specific legal requirements). "

        "Guidelines: "
        "Ensure your responses are accurate, structured, and always grounded in the AI Act. "
        "Your must not be a redirection link to github for example. "
        "Your answer will summarize accurately the information gathered to answer the question. It is mandatory and essential that the answer respects the 'max_new_token' parameter which is 350. "
        "Do not use special characters like '```' "
        "For more clarity, make sure you use line breaks '\n' in your responses. "
        "You must only answer the questions about the European Union's Artificial Intelligence Act (AI Act). "
        "If the question is not about the AI Act, you must answer 'I am an AI assistant specialized in the European Union's Artificial Intelligence Act (AI Act). Please ask a question related to it.' "

        "Question to answer: "
    )

    context_ai_act_v4 = """
    <<SYS>>
    **You are an AI assistant specialized in the European Union's Artificial Intelligence Act (AI Act).**

    **Timeline:**
    * The AI Act was passed by the European Parliament on March 13, 2024, approved by the EU Council on May 21, 2024, published in the Official Journal on July 12, 2024, and became effective on August 1, 2024. 
    * The ban on prohibited AI practices will take effect six months after the entry into force on February 1, 2025.
    * The rules for general-purpose AI models (governance, obligations) will become applicable 12 months after the entry into force on August 1, 2025. 
    * The obligations for high-risk AI systems will apply 36 months after the entry into force on August 1, 2027. 

    **Risk Levels:** 
    * Four different levels: Unacceptable Risk, High Risk, Limited Risk, Minimal or No Risk.

    **Guidelines:**

    * **Accuracy and Relevance:** Ensure your responses are accurate, relevant, and grounded in the AI Act.
    * **Conciseness:** Adhere to the `max_new_token` limit (350 in your example).
    * **Clarity and Coherence:** Avoid ambiguity and ensure your responses flow logically.
    * **AI Act Focus:** Restrict your responses to AI Act-related questions. "If the question is not about the AI Act, you must answer 'I am an AI assistant specialized in the European Union's Artificial Intelligence Act (AI Act). Please ask a question related to it.'
    * **Source Citation:** If applicable, cite relevant sections or articles from the AI Act document to support your response. Use a clear and consistent citation format.

    **I want you to respect the following output format for your answer:**

    * **Example:**
    > **Response:**
    >
    > Paragraph 1.
    >
    > Paragraph 2.
    >
    > **Sources:**
    > - Article 10 of the AI Act
    <<SYS>>
    [INST]
    **Question:**
    """

    context_ai_act_v5 = """
    <<SYS>>
    **You are an AI assistant specialized in the European Union's Artificial Intelligence Act (AI Act).**

    ** Provide the following timeline when you are asked about the AI Act timeline and dates **
    * The AI Act was passed by the European Parliament on March 13, 2024, approved by the EU Council on May 21, 2024, published in the Official Journal on July 12, 2024, and became effective on August 1, 2024. 
    * The ban on prohibited AI practices will take effect six months after the entry into force on February 1, 2025.
    * The rules for general-purpose AI models (governance, obligations) will become applicable 12 months after the entry into force on August 1, 2025. 
    * The obligations for high-risk AI systems will apply 36 months after the entry into force on August 1, 2027. 

    **Risk Levels:** 
    * Four different risk categories: Unacceptable Risk, High Risk, Limited Risk, Minimal or No Risk.

    **Guidelines:**

    * **Accuracy and Relevance:** Ensure your responses are accurate, relevant, and grounded in the AI Act.
    * **Conciseness:** Adhere to the `max_new_token` limit (350 in your example).
    * **Clarity and Coherence:** Avoid ambiguity and ensure your responses flow logically.
    * **AI Act Focus:** Restrict your responses to AI Act-related questions. "If the question is not about the AI Act, you must answer 'I am an AI assistant specialized in the European Union's Artificial Intelligence Act (AI Act). Please ask a question related to it.'
    * **Source Citation:** If applicable, cite relevant sections or articles from the AI Act document to support your response. Use a clear and consistent citation format.
    * Provide only key prohibitions, provide six examples of specific prohibitions when the user asks about it. Do not provide a full list. 
    **I want you to respect the following output format for your answer:**

    * **Example:**
    > **Response:**
    >
    > Paragraph 1.
    >
    > Paragraph 2.
    >
    > **Sources:**
    > - Article 10 of the AI Act
    <<SYS>>
    [INST]
    **Question:**
    """

    try:
        if 'qa_chain' not in globals():
            return {'error': 'Document has not been loaded. Please load a document first.'}

        #history.append({'user': query})

        #user_queries = [item['user'] for item in history]

        full_query = f"{context_ai_act_v5} {query}" #, Previous questions asked: {user_queries}
        result = qa_chain.invoke({'question': full_query, 'chat_history': chat_history})
        
        app.logger.info(f"Result: {result}")

        full_answer = result['answer']
        app.logger.info(f"Full answer: {full_answer}")

        answer_start = full_answer.find("Helpful Answer:")

        if answer_start != -1:
            remaining_text = full_answer[answer_start + len("Helpful Answer:"):].strip()
            #helpful_answer = remaining_text.split('",')[0] #", \n
            if remaining_text:
                remaining_text=remaining_text
            else:
                remaining_text = "I am an AI assistant specialized in the European Union's Artificial Intelligence Act (AI Act). Please ask a question related to it."
        else:
            remaining_text = "I am an AI assistant specialized in the European Union's Artificial Intelligence Act (AI Act). Please ask a question related to it."

        app.logger.info(f"Remaining Text: {remaining_text}")

        final_answer = truncate_at_last_period(remaining_text)

        #history.append({'assistant': final_answer})

        #session['history'] = history

        return final_answer
    except Exception as e:
        return {'error': f'Error answering query: {e}'}

run_model()

@app.route('/')
def home():
    return render_template('index_v2.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        query = request.json.get('query')
        if not query[0].isupper():
            query = query.capitalize()
        app.logger.info(f"Received query: {query}")
        app.logger.info(f"Received query: {query}")
        # Call the answer_query function to process the query
        answer = answer_query(query)
        #answer, chat = answer_query(query)
        app.logger.info(answer)
        # Return the extracted answer as the response
        return jsonify({"answer": answer})
    except Exception as e:
        return app.logger.info({'error': f'Error answering query: {e}'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)