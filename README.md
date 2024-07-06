# CrewAI-Groq Assignment Evaluator

## Overview

The CrewAI-Groq Assignment Evaluator is a Jupyter Notebook designed to help users evaluate assignments using CrewAI and Groq. This README provides a brief guide on how to set up and use the notebook.

## Installation

To get started, you need to install the required packages. Run the following commands in a Jupyter cell:

```python
!pip install crewai
!pip install 'crewai[tools]'
!pip install langchain_groq
```

## Setup

1. **Import Libraries:**

   Import the necessary libraries by including the following in your notebook:

   ```python
   import os
   from crewai import Agent, Task, Crew, Process
   from crewai_tools import SerperDevTool
   from langchain_groq import ChatGroq
   import getpass
   ```

2. **Set API Key:**

   Set your Groq API key using the following code snippet:

   ```python
   os.environ['GROQ_API_KEY'] = getpass.getpass('Enter your Groq API key: ')
   ```

3. **Initialize LLM:**

   Initialize the language model with your API key:

   ```python
   llm = ChatGroq(model="gemma-7b-it", groq_api_key=os.environ['GROQ_API_KEY'])
   ```

## Usage

To interact with the model, you can use the `invoke` method:

```python
response = llm.invoke("hello how are you?")
print(response)
```

## Assignment

The notebook includes an example assignment for creating personalized article summaries. Below is the assignment text:

```python
assignment_text = """
Assignment: Personalized Article Summaries

Goal:

Develop a prompt that can summarize news articles into statements of a specified length and focus on information specific to your interests.

Summarize a few news articles yourself. Your summaries should be the length, tone, and writing style that you prefer. Make sure to discuss the information that you personally find interesting from the articles. Then, use your example summaries to create a prompt that the AI can follow.
"""
```

## Crew Setup

To handle the evaluation process, instantiate a crew with a sequential process. This crew includes various agents tasked with assessing different aspects of the assignment, such as clarity, relevance, accuracy, tone, and the use of examples. You can adjust the verbosity to control the level of logging detail.

## Contributing

Feel free to submit issues or pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For any further questions or help, please refer to the official documentation of the libraries used in this project.
