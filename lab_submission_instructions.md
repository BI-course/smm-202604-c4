# Lab Submission Instructions

## Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:** 202604-c4

**Member 1:**

| **Details**                                                                                                                           | **Comment**            |
|:--------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|
| **Student ID**                                                                                                                        | 150367                 |
| **Name**                                                                                                                              | Christine Nessy Mungla |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | My contribution to the lab focused on the implementation of the regression and classification endpoints. I worked on the development branch feature/regression-classification-endpoints, which involved working on the API functionality for integrating machine learning models with the application. Through this task, I learned how regression and classification models can be exposed through API endpoints to make predictions. I also gained a better understanding of backend development, how endpoints handle requests and responses, and how machine learning functionality can be integrated into an application.https://github.com/BI-course/smm-202604-c4/tree/feature/regression-classification-endpoints                      |

**Member 2:**

| **Details**                                                                                                                           | **Comment**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| :------------------------------------------------------------------------------------------------------------------------------------ |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Student ID**                                                                                                                        | 169223                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Name**                                                                                                                              | Lukwago Bernard Desmond                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** | In the lab, I was Member 2, responsible for the clustering and association-rule endpoints in the shared api.py. I added a customer segmentation endpoint that loads the saved preprocessor and K-Means model and returns the segment name and profile. I also added a streaming recommender endpoint that loads the saved association rules and recommends shows from what a viewer has watched. Both models load once at startup and are never retrained. I added input checks that return clear errors, and a popular-shows fallback when no rule matches. <br/>I tested both with curl and found that our models were saved with scikit-learn 1.6.1 while the project pinned a newer version. I then set the pin to 1.6.1 and removed mlxtend, which conflicted with it. I committed on my own feature branch, and shared the version changes with the Docker and Render owners. What I learned is that a trained pipeline can be saved and served as an API, and that matching library versions, data types and the deployment setup is as important as the model itself. I also practised careful Git use and teamwork on a shared file. Link To Branch: https://github.com/BI-course/smm-202604-c4/tree/feature/clustering-recommender-endpoints |

**Member 3:**

| **Details**                                                                                                                           | **Comment** |
| :------------------------------------------------------------------------------------------------------------------------------------ | :---------- |
| **Student ID**                                                                                                                        |             |
| **Name**                                                                                                                              |             |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |             |

**Member 4:**

| **Details** | **Comment** |
| :---------- | :---------- |
| **Student ID** | 167284 |
| **Name** | Nyaga Gacheru |
| **What part of the lab did you personally<br>contribute to (provide a link to the<br>branch(es)), and what did you learn from it?** | **Contribution:** Deployed our Flask ML API (`api.py`) publicly on Render (`https://smm-202604-c4-nyaga.onrender.com`) via Gunicorn, built and connected the custom HTML, CSS, and Vanilla JavaScript frontend pages (`index.html`, `daily_sales_revenue_regressor.html`, `distress_risk_classifier.html`, `customer_segmenter.html`, `streaming_recommender.html`, and `config.js`) with client-side missing-input validation and API error handling, and fixed a scikit-learn `SimpleImputer` `float64` input-casting bug in the daily sales revenue regression endpoint.<br><br>**Branch & Pull Request:**<br>- Feature Branch: https://github.com/BI-course/smm-202604-c4/tree/feature/render-frontend<br>- Merged PR (#4): https://github.com/BI-course/smm-202604-c4/pull/4<br><br>**What I Learned:**<br>1. **Cloud Deployment & Environment Parity:** Learned how to configure Gunicorn and environment settings on Render so the exact same saved model artifacts and Flask endpoints run reliably outside a local development machine.<br>2. **API Integration & Error Handling:** Learned how decoupled frontends consume REST endpoints via `fetch()`, why client-side input validation and structured JSON errors (`400 Bad Request`) are essential, and how data-type mismatches (integer vs. `float64` arrays in scikit-learn pipelines) surface during live API integration. |


**Member 5:**

| **Details**                                                                                                                            | **Comment** |
| :------------------------------------------------------------------------------------------------------------------------------------ | :--  -------- |
| **Student ID**                 167753                                                                                                    |                |
| **Name**                    Clara Waithaka                                                                                                       |              |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?** |Developed and deployed a Gradio-based web application integrating all four machine learning models through the group's Flask API hosted on Render. Implemented a single Gradio interface with four tabs for Daily Sales Revenue Prediction, Business Distress Risk Classification, Customer Segmentation, and Streaming Recommendations. Configured API communication, handled prediction responses, and prepared the application for deployment on Hugging Face Spaces, including ZeroGPU startup registration and dependency configuration.

Successfully tested all four model prediction functionalities locally and on the public Hugging Face Space.
|                https://github.com/BI-course/smm-202604-c4/tree/feature/gradio-space                                                                                                                       |

**Member 6:**

| **Details** | **Comment** |
| :---------- | :---------- |

| **Student ID** 165828 **Name** Daniela Dudi Awino
| |
| **What part of the lab did you personally<br/>contribute to (provide a link to the<br/>branch(es)), and what did you learn from it?**
I built the Streamlit Community Cloud app, which acts as a no-code demo UI. I designed a four-tab interface (one for each model) that loads the pre-trained .joblib artifacts directly into memory using joblib and @st.cache_resource. Through this, I learned how to translate backend machine learning models into interactive, user-friendly web applications, how to manage virtual environments and dependencies (imbalanced-learn), and how to deploy Streamlit apps to the cloud. Link to my branch: | https://github.com/BI-course/smm-202604-c4/tree/feature/gradio-space

## Chosen Level of Difficulty

**Specify the chosen level of difficulty** (baseline, intermediate, or advanced): Baseline, intermediate and advanced

## Video Demonstration

Submit the link to a short video (**not more than 10 minutes**) demonstrating your complete solution at whichever difficulty level you chose. Please ensure that the lecturer has rights to view the video.

At minimum, the video must show, in one continuous walkthrough:

1. `api.py` running **locally**, with a live request/response against each model endpoint you implemented.
2. If you attempted the Intermediate or Advanced tier: the same API running inside its **Gunicorn + Nginx Docker container**, reached through the reverse proxy, not the Flask development server.
3. If you attempted the Advanced tier: your **Hugging Face Space** and your **Streamlit Community Cloud app**, each opened live in a browser, with at least one prediction made on each tab, and your **Render**-hosted API answering a request from Postman or `curl`.

**Why this matters for grading:** free-tier public deployments are not guaranteed to still exist by the time this is graded. The video is treated as the authoritative record of what was actually working at submission time; a broken link at grading time is not penalized if the video clearly shows that deployment working.

Note that you are required to submit the link to the video and NOT the video itself. The video should NOT be uploaded to your repository—that would be a misuse of GitHub.

**Link to the video:** https://drive.google.com/drive/folders/1sg5Q8jIuHVP8aKD14hicga8JCEeGhS8S?usp=sharing

## Public URLs

Fill in only the rows that apply to your chosen difficulty level (see Part C below). Leave a row blank if your tier does not require that deployment target.

| **Deployment target**                             | **URL**                                                     |
| :------------------------------------------------ | :---------------------------------------------------------- |
| Render (the full `api.py`, all endpoints)         |       https://smm-202604-c4-nyaga.onrender.com                                                |
| Hugging Face Space (Gradio, one tab per model)    |    https://huggingface.co/spaces/ClaritaW/business-intelligence-models                                                         |
| Streamlit Community Cloud app (one tab per model) | https://smm-202604-c4-ucssviswc2txbs2t3pf6kx.streamlit.app/ |

---

## Scenario

Across earlier labs, **your group** was assigned its own dataset and trained your own models for four distinct tasks:

1. A **regression** task
2. A **classification** task
3. A **k-Means clustering** task
4. An **Apriori association-rule mining** task

Different groups worked with different datasets and, in some cases, different algorithms for each task -- there is no single shared reference repository or model list for the whole class. Wherever this document refers to "your regressor," "your classifier," "your clustering model," or "your recommender," it means **whichever model your own group already trained and saved for that task**, not a specific named algorithm.

You are now required to build a single Flask API (`api.py`) that serves **your own group's four models**, loaded from disk (never retrained inside `api.py`), and to demonstrate that API running correctly across a growing number of deployment surfaces depending on the difficulty tier you choose.

### Part A: Regression and Classification

- Locate the trained regression and classification model files your group produced and saved in your earlier labs on Regression and Classification.
- Update [api.py](api.py) to include end-points that load these two models from disk and serve predictions from them.

### Part B: Clustering and Association Rule Mining

- Locate the trained clustering model and the association rules your group produced and saved in your earlier labs on Clustering and Association Rule Mining.
- Update [api.py](api.py) to include end-points that:
  - Load the clustering model from disk and predict which cluster a new client belongs to
  - Load the association rules from disk and recommend products based on them

**Note 1:** **`api.py` is NOT production-grade as it is.** It is only meant for demonstration purposes. Scalability and security must be taken into consideration before deploying an API in a production environment.

**Note 2:** Some students often treat the API as an afterthought, focusing only on training ML models. In practice, **the API is the product** -- it is how others interact with your model. The "hidden" learning here is that the delivery mechanism (API design, usability, error handling, and even documentation) often matters more to stakeholders in the industry than the models themselves.

### Part C: Deployment Breadth (Read Before Choosing Your Tier)

All three tiers below ask you to serve **the same four models** (your regressor, your classifier, your clustering model, your recommender) from **one, unmodified `api.py`**. The model files and the Flask routes do not change between tiers -- only how many different places that one working API is shown to run correctly.

**Baseline (Required):**

- Update `api.py` to serve **at least three of your four** models, loaded from disk.
- Run and demonstrate `api.py` **locally** with Flask's development server, with a successful request against each implemented endpoint.

**Intermediate (Recommended):**

- Update `api.py` to serve **all four** of your models, loaded from disk.
- Dockerize your Flask API using **Gunicorn**, and place it behind an **Nginx** reverse proxy (see the reverse-proxy lab notes for the full Docker Compose setup).
- Deploy the same, unmodified `api.py` to **Render**, and demonstrate a real request against it from outside your own machine (cURL or Postman).

**Advanced (Optional):**

- Everything required for Intermediate, plus:
  - Create a web page (or pages) using Basic HTML, CSS, and Vanilla JavaScript that demonstrates the use of the API
  - Implement basic error handling (e.g., missing inputs)
  - Build **one Gradio app with a tab per model** (all four of your models) and publish it as a **Hugging Face Space**
  - Build **one Streamlit app with a tab per model**, covering the same four models, and publish it on **Streamlit Community Cloud**
  - All deployment targets -- local, Dockerized/Nginx, Render, Hugging Face, and Streamlit -- must serve models loaded **from the same trained artifacts**; retraining different models per platform does not satisfy this requirement

Why this is production-relevant beyond the model itself:

- **Portability:** the same container image and the same trained artifacts run unmodified on your laptop, inside Docker, and on a managed cloud host.
- **Consistency:** no "but it works on my machine" gap between what you demo locally and what a domain expert sees on a public URL.
- **Separation of concerns:** a REST API (Render), a no-code demo UI (Hugging Face/Streamlit), and your own frontend (the Advanced-tier HTML pages) are three different ways of exposing the _same_ underlying models to three different kinds of user -- a developer integrating your API, a non-technical domain expert validating it, and an end customer.

---

## Marking Guide

Refer to the marking guide available here: [lab_submission_marking_guide.md](lab_submission_marking_guide.md)
