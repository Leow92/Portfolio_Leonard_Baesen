<h1 align="left">Creation of a chatbot answering questions on the AI Act</h1>

![image](https://github.com/user-attachments/assets/f214be02-df4b-4caf-b605-a59224e227a6)

This interactive tool empowers users to explore the European Union's Artificial Intelligence Act (AI Act).<br>
It utilizes the capabilities of the <a href="https://mistral.ai/fr/news/announcing-mistral-7b/" target="_blank">Mistral 7B</a> model and Retrieval Augmented Generation (RAG) technology to provide insightful answers to your questions about the <a href="https://www.europarl.europa.eu/doceo/document/TA-9-2024-0138_EN.pdf" target="_blank">EU AI Act</a>.<br>
Ask away!

## Table of Contents

- [Project Overview](#project-overview)
- [Technology Used](#technology-used)
- [Installation and Setup](#installation-and-setup)
- [Contributor](#contributor)

## Project Overview

This Flask webapp is designed to create a chatbot capable of answering questions about the AI Act. It leverages Retrieval Augmented Generation (RAG) to provide informative and relevant responses based on the provided PDF document.

The code is located in the **flask_app_mistral_v7** directory.

## Technology Used

* **Python:** Coded in Python 3.10
* **Flask:** A popular Python web framework for building web applications.
* **Google Cloud Platform:** The cloud infrastructure where the webapp is deployed.
* **Hugging Face:** A platform for sharing and using machine learning models, including the Mistral 7B language model.
* **Mistral 7B:** The language model used for generating responses.
* **PDF AI Act:** The document containing the information about the AI Act.
* **Docker:** A containerization platform used to package the application and its dependencies.
* **LangChain:** A library for building LLM applications, providing tools for data retrieval, question answering, and more.

## Installation and Setup

1.  **Clone the Repository:**

    ```sh
    git clone [https://github.com/Sailpeak/Chat_AI_Act.git](https://github.com/Sailpeak/Chat_AI_Act.git)
    ```

2.  **Install Dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3.  **Set Up Google Cloud Project:**

    * Create a new Google Cloud project.
    * Enable the necessary APIs (e.g., Cloud Run, Cloud Storage).
    * Create a service account with appropriate permissions.

4.  **Configure Environment Variables:**

    * Set environment variables like `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`, and any other required credentials.

5.  **Push your image on Artifact Registry:**

    * **Build and Tag Your Image:**

        ```sh
        docker build -t gcr.io/<your-project-id>/<your-image-name> .
        ```

    * **Push the Image:**

        ```sh
        docker push gcr.io/<your-project-id>/<your-image-name>
        ```

6.  **Deploy on Cloud Run:** Select the docker image from Artifact Registry and deploy.

## Contributor

Léonard Baesen ([https://github.com/Leow92](https://github.com/Leow92))