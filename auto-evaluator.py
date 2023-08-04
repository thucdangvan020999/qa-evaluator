import os
from typing import List

import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain

from text_utils import (
    GRADE_ANSWER_PROMPT,
    GRADE_ANSWER_PROMPT_BIAS_CHECK,
    GRADE_ANSWER_PROMPT_FAST,
    GRADE_ANSWER_PROMPT_OPENAI,
    GRADE_DOCS_PROMPT,
    GRADE_DOCS_PROMPT_FAST,
)

# Keep dataframe in memory to accumulate experimental results
if "existing_df" not in st.session_state:
    summary = pd.DataFrame(
        columns=["Foundation LLM", "Retrieval score", "Answer score"]
    )
    st.session_state.existing_df = summary
else:
    summary = st.session_state.existing_df


def grade_model_answer(
    predicted_dataset: List, predictions: List, grade_answer_prompt: str
) -> List:
    # Grade the distilled answer
    st.info("`Grading model answer ...`")
    # Set the grading prompt based on the grade_answer_prompt parameter
    if grade_answer_prompt == "Fast":
        prompt = GRADE_ANSWER_PROMPT_FAST
    elif grade_answer_prompt == "Descriptive w/ bias check":
        prompt = GRADE_ANSWER_PROMPT_BIAS_CHECK
    elif grade_answer_prompt == "OpenAI grading prompt":
        prompt = GRADE_ANSWER_PROMPT_OPENAI
    else:
        prompt = GRADE_ANSWER_PROMPT

    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), prompt=prompt
    )

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        predicted_dataset, predictions, question_key="question", prediction_key="result"
    )

    return graded_outputs


def grade_model_retrieval(gt_dataset: List, predictions: List, grade_docs_prompt: str):
    # Grade the docs retrieval
    st.info("`Grading relevance of retrieved docs ...`")

    # Set the grading prompt based on the grade_docs_prompt parameter
    prompt = (
        GRADE_DOCS_PROMPT_FAST if grade_docs_prompt == "Fast" else GRADE_DOCS_PROMPT
    )

    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), prompt=prompt
    )

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        gt_dataset, predictions, question_key="question", prediction_key="result"
    )
    return graded_outputs


def run_evaluation(uploaded_file, grade_prompt):
    st.info("`Running evaluation ...`")
    df = 0
    df = pd.read_csv(uploaded_file).dropna()
    gt_dataset = df.apply(
        lambda x: {"question": x["question"], "answer": x["answer"]}, axis=1
    ).tolist()

    predictions_list = df.apply(
        lambda x: {
            "question": x["question"],
            "answer": x["answer"],
            "result": x["result"],
        },
        axis=1,
    ).tolist()

    retrieved_docs = df.apply(
        lambda x: {
            "question": x["question"],
            "answer": x["answer"],
            "result": x["result"],
        },
        axis=1,
    ).tolist()

    answers_grade = grade_model_answer(gt_dataset, predictions_list, grade_prompt)
    retrieval_grade = grade_model_retrieval(gt_dataset, retrieved_docs, grade_prompt)
    return predictions_list, answers_grade, retrieval_grade


# Auth
st.sidebar.image("img/diagnostic.jpg")

with st.sidebar.form("user_input"):
    oai_api_key = st.text_input("`OpenAI API Key:`", type="password").strip()

    grade_prompt = st.radio(
        "`Grading style prompt`",
        ("Fast", "Descriptive", "Descriptive w/ bias check", "OpenAI grading prompt"),
        index=0,
    )

    submitted = st.form_submit_button("Submit evaluation")

# App
st.header("`Question Answering Evaluator`")
st.info("I am an evaluation tool for question-answering.")

with st.form(key="file_inputs"):
    uploaded_file = st.file_uploader(
        "`Please upload a file to evaluate (.csv or .xlsx):` ", type=["csv", "xlsx"]
    )
    foundation_llm = st.text_area("What is Foundation LLM ? ", height=None)

    submitted = st.form_submit_button("Submit files")

if uploaded_file:
    st.write(foundation_llm)
    os.environ["OPENAI_API_KEY"] = oai_api_key
    # Load csv/xslx file
    result_qa, graded_answers, graded_retrieval = run_evaluation(
        uploaded_file, grade_prompt
    )
    # Assemble outputs
    dataframe = pd.DataFrame(result_qa)
    dataframe["docs score"] = [g["text"] for g in graded_retrieval]
    dataframe["answer score"] = [g["text"] for g in graded_answers]

    # rename df
    dataframe.rename(columns={"question": "Question"}, inplace=True)
    dataframe.rename(columns={"answer": "Expected Answer"}, inplace=True)
    dataframe.rename(columns={"result": "Observed Answer"}, inplace=True)
    dataframe.rename(columns={"docs score": "Retrieval Relevancy Score"}, inplace=True)
    dataframe.rename(columns={"answer score": "Answer Similarity Score"}, inplace=True)

    # Summary statistics

    correct_answer_count = len(
        [
            text
            for text in dataframe["Answer Similarity Score"]
            if "INCORRECT" not in text
        ]
    )
    correct_docs_count = len(
        [
            text
            for text in dataframe["Retrieval Relevancy Score"]
            if "Context is relevant: True" in text
        ]
    )
    percentage_answer = (correct_answer_count / len(graded_answers)) * 100
    percentage_docs = (correct_docs_count / len(graded_retrieval)) * 100

    st.subheader("`Run Results`")
    # st.info(
    #     "`I will grade the chain based on: 1/ the relevance of the retrived documents relative to the question and 2/ "
    #     "the summarized answer relative to the ground truth answer. You can see (and change) to prompts used for "
    #     "grading in text_utils`"
    # )
    st.dataframe(data=dataframe, use_container_width=True)
    # Accumulate results
    st.subheader("`Aggregate Results`")
    # st.info(
    #     "`Retrieval and answer scores are percentage of retrived documents deemed relevant by the LLM grader ("
    #     "relative to the question) and percentage of summarized answers deemed relevant (relative to ground truth "
    #     "answer), respectively. The size of point correponds to the latency (in seconds) of retrieval + answer "
    #     "summarization (larger circle = slower).`"
    # )
    new_row = pd.DataFrame(
        {
            "Foundation LLM": [foundation_llm],
            "Retrieval score": [percentage_docs],
            "Answer score": [percentage_answer],
        }
    )
    summary = pd.concat([summary, new_row], ignore_index=True)
    st.dataframe(data=summary, use_container_width=True)
    st.session_state.existing_df = summary

    # # Dataframe for visualization
    # show = summary.reset_index().copy()
    # show.columns = ["foundation_llm", "Retrieval score", "Answer score"]
    # c = (
    #     alt.Chart(show)
    #     .mark_circle()
    #     .encode(
    #         x="Retrieval score",
    #         y="Answer score",
    #         color="foundation_llm",
    #         tooltip=["foundation_llm", "Retrieval score", "Answer score"],
    #     )
    # )
    # st.altair_chart(c, use_container_width=True, theme="streamlit")
