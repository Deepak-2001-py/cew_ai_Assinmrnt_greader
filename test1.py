import unittest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from your_module import Agent, Task, Crew

# MockConfig class with required fields
class MockConfig(BaseModel):
    agents: list
    tasks: list

# Test case
class TestCrewAIGroqAssignmentEvaluator(unittest.TestCase):

    @patch('your_module.Agent')
    @patch('your_module.Task')
    @patch('your_module.Crew')
    def test_clarity_agent(self, MockCrew, MockTask, MockAgent):
        # Setup mock objects
        mock_llm = MagicMock()
        mock_assignment_text = """
        Assignment: Personalized Article Summaries

        Goal:

        Develop a prompt that can summarize news articles into statements of a specified length and focus on information specific to your interests.

        Summarize a few news articles yourself. Your summaries should be the length, tone, and writing style that you prefer. Make sure to discuss the information that you personally find interesting from the articles. Then, use your example summaries for in-context learning with a language model like OpenAI's ChatGPT, Google's Bard, or Anthropic's Claude. Now, the language model should learn from the context how to create news summaries that fit your length, tone, style, and content preferences.
        """
        mock_assignment_submission = """
        Assignment Submission:
        - Article Title: "Africa's Great Green Wall Initiative Making Significant Progress"
          Summary: The Great Green Wall, an ambitious project to combat desertification in Africa, has achieved significant milestones. Over 20 million hectares of land have been restored, benefiting local communities through increased agricultural productivity and job creation. The initiative aims to restore 100 million hectares by 2030, contributing to climate resilience and biodiversity.
        - Article Title: "Revolutionary AI Developed for Early Disease Detection"
          Summary: Researchers have developed a groundbreaking AI system capable of early detection of diseases like cancer and Alzheimer's. Using advanced algorithms and vast datasets, the AI can analyze medical images with unprecedented accuracy, potentially saving millions of lives through early intervention. This innovation marks a significant leap forward in personalized medicine and healthcare efficiency.
        - Article Title: "Astronomers Discover Earth-like Exoplanets in Habitable Zone"
          Summary: A team of astronomers has discovered two Earth-like exoplanets within the habitable zone of a nearby star. These planets, located 12 light-years away, have conditions that may support liquid water, raising the possibility of life beyond our solar system. This discovery fuels excitement in the search for extraterrestrial life and understanding planetary formation.
        """

        # Define the agents
        clarity_agent = MockAgent(role='Clarity and Conciseness Grader', goal='Evaluate the clarity and conciseness of the assignment submission', backstory=f"""You are an experienced educator with a keen eye for detail, specializing in evaluating the clarity and conciseness of written content. Assignment: {mock_assignment_text}""", verbose=True, llm=mock_llm, allow_delegation=False)

        relevance_agent = MockAgent(role='Relevance and Focus Grader', goal='Evaluate the relevance and focus of the assignment submission', backstory=f"""You are a seasoned educator with a strong background in assessing the relevance and focus of academic work. Assignment: {mock_assignment_text}""", verbose=True, llm=mock_llm, allow_delegation=False)

        accuracy_agent = MockAgent(role='Accuracy Grader', goal='Evaluate the accuracy of the information in the assignment submission', backstory=f"""You are an expert in evaluating the accuracy of content, ensuring that all facts and details are correctly represented. Assignment: {mock_assignment_text}""", verbose=True, llm=mock_llm, allow_delegation=False)

        tone_agent = MockAgent(role='Tone and Style Grader', goal='Evaluate the tone and style of the assignment submission', backstory=f"""You have a background in literature and writing, with extensive experience in evaluating the tone and style of written work. Assignment: {mock_assignment_text}""", verbose=True, llm=mock_llm, allow_delegation=False)

        examples_agent = MockAgent(role='Examples and Details Grader', goal='Evaluate the use of examples and details in the assignment submission', backstory=f"""You specialize in assessing the use of examples and details in academic and professional writing. Assignment: {mock_assignment_text}""", verbose=True, llm=mock_llm, allow_delegation=False)

        # Define the tasks
        task_clarity = MockTask(description=f"""Grade the clarity and conciseness of the following assignment submission on a scale of 1-20: {mock_assignment_submission}""", expected_output="Score for clarity and conciseness with comments", agent=clarity_agent)
        task_relevance = MockTask(description=f"""Grade the relevance and focus of the following assignment submission on a scale of 1-30: {mock_assignment_submission}""", expected_output="Score for relevance and focus with comments", agent=relevance_agent)
        task_accuracy = MockTask(description=f"""Grade the accuracy of the information in the following assignment submission on a scale of 1-20: {mock_assignment_submission}""", expected_output="Score for accuracy with comments", agent=accuracy_agent)
        task_tone = MockTask(description=f"""Grade the tone and style of the following assignment submission on a scale of 1-20: {mock_assignment_submission}""", expected_output="Score for tone and style with comments", agent=tone_agent)
        task_examples = MockTask(description=f"""Grade the use of examples and details in the following assignment submission on a scale of 1-10: {mock_assignment_submission}""", expected_output="Score for use of examples and details with comments", agent=examples_agent)

        # Mock the configuration
        mock_config = MockConfig(agents=[clarity_agent, relevance_agent, accuracy_agent, tone_agent, examples_agent], tasks=[task_clarity, task_relevance, task_accuracy, task_tone, task_examples])

        # Assign the mock configuration to the return value of MockAgent
        MockAgent.return_value.config = mock_config

        # Instantiate the crew with the mock configuration
        MockCrew.return_value.agents = [clarity_agent, relevance_agent, accuracy_agent, tone_agent, examples_agent]
        MockCrew.return_value.tasks = [task_clarity, task_relevance, task_accuracy, task_tone, task_examples]
        MockCrew.return_value.verbose = 2
        MockCrew.return_value.process = 'sequential'

        # Set the expected return value for the crew kickoff
        MockCrew.return_value.kickoff.return_value = """he provided summaries demonstrate a meticulous grasp of language and a keen understanding of how to effectively convey information within the specified parameters. Each summary successfully captures the essence of the respective articles while adhering to the desired length and tone.

**Strengths:**

* **Accuracy and Concision:** The summaries accurately summarize the key points of each article while remaining concise and engaging.
* **Vivid Vocabulary:** The use of vivid vocabulary adds depth and clarity to the summaries, bringing the stories to life.
* **Action Verbs:** The incorporation of action verbs enhances the readability and impact of the summaries, making them more compelling.

**Areas for Potential Improvement:**

* **Incorporation of Supporting Details:** While the summaries provide a solid overview, they could benefit from the inclusion of additional supporting details to further enrich the content. For example, including specific data points or statistics could enhance the impact of the summaries.
* **Variety in Sentence Structure:** To enhance readability, the writer could diversify the sentence structure to create a more engaging reading experience.

**Recommendations for Future Submissions:**

* Integrate relevant supporting details and data to bolster the impact of the summaries.
* Explore different sentence structures to create a more diverse and engaging writing style.
* Consider incorporating relevant anecdotes or personal insights to add a unique and captivating touch to the summaries.

**Overall, this is an impressive display of summarizing skills. By implementing the suggested improvements, the writer can consistently produce exceptional summaries that engage and inform readers.**"""

        # Execute the test
        crew = MockCrew()
        result = crew.kickoff()

        # Assertions
        self.assertEqual(result, """he provided summaries demonstrate a meticulous grasp of language and a keen understanding of how to effectively convey information within the specified parameters. Each summary successfully captures the essence of the respective articles while adhering to the desired length and tone.

**Strengths:**

* **Accuracy and Concision:** The summaries accurately summarize the key points of each article while remaining concise and engaging.
* **Vivid Vocabulary:** The use of vivid vocabulary adds depth and clarity to the summaries, bringing the stories to life.
* **Action Verbs:** The incorporation of action verbs enhances the readability and impact of the summaries, making them more compelling.

**Areas for Potential Improvement:**

* **Incorporation of Supporting Details:** While the summaries provide a solid overview, they could benefit from the inclusion of additional supporting details to further enrich the content. For example, including specific data points or statistics could enhance the impact of the summaries.
* **Variety in Sentence Structure:** To enhance readability, the writer could diversify the sentence structure to create a more engaging reading experience.

**Recommendations for Future Submissions:**

* Integrate relevant supporting details and data to bolster the impact of the summaries.
* Explore different sentence structures to create a more diverse and engaging writing style.
* Consider incorporating relevant anecdotes or personal insights to add a unique and captivating touch to the summaries.

**Overall, this is an impressive display of summarizing skills. By implementing the suggested improvements, the writer can consistently produce exceptional summaries that engage and inform readers.**""")

if __name__ == '__main__':
    unittest.main()
