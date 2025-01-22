import base64
import json
from mimetypes import guess_type

import requests
from pydantic import BaseModel, Field
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

llm = AzureChatOpenAI(
    model=gpt_config['model_name'],
    deployment_name=gpt_config['deployment_name'],
    openai_api_key=gpt_config['api_key'],
    openai_api_version=gpt_config['api_version'],
    azure_endpoint=gpt_config['api_base'],
    temperature=gpt_config['temperature']
)

def generate_hw01(question):
    examples = [
        {
            "input":"2024年台灣10月紀念日有哪些?",
            "output":
            {
                "Result": [
                    {
                        "date": "2024-10-10",
                        "name": "國慶日"
                    },
                    {
                        "date": "2024-10-11",
                        "name": "重陽節"
                    }
                ]
            }
        }
    ]

    example_prompt = FewShotChatMessagePromptTemplate(
        examples = examples,
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        ),
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "只查詢台灣的行事曆的節日，輸出只要JSON格式，可以有多個節日"),
            example_prompt,
            ("human", "{input}"),
        ]
    )

    chain = final_prompt | llm
    response_chain = chain.invoke({"input":question})
    response_json = json.dumps(response_chain.content, indent=4, ensure_ascii=False).encode('utf8').decode().replace("```json\\n","").replace("\\n```","")
    response = json.loads(response_json)
    return response

def generate_hw02(question):
    #final_prompt = ChatPromptTemplate.from_messages(
    #    [
    #        ("ai", "{agent_scratchpad}"),
    #        ("system", "只查詢台灣的行事曆的節日，輸出只要JSON格式，可以有多個節日"),
    #        ("human", "{input}"),
    #    ]
    #)

    #tools = [get_holidays]
    #agent = create_openai_functions_agent(llm, tools, final_prompt)
    #agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    #agent_response = agent_executor.invoke({"input": question})
    #print(agent_response)

    query_year = get_year(question)
    query_month = get_month(question)
    response = get_holidays("TW", query_year, query_month)

    return response

    
def generate_hw03(question2, question3):
    prompt = ChatPromptTemplate.from_messages([
        ("ai", "{holiday_list}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    history_handler = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )

    result_part = history_handler.invoke(
        {"holiday_list": generate_hw02(question2),
         "question": question3},
        config={"configurable": {"session_id": "holidays"}},
    )

    response = history_handler.invoke(
        {"holiday_list": generate_hw02(question2),
         "question": "請根據問題列出台灣的紀念日，以 JSON 格式輸出:date : add : 這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。 reason : 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。"},
        config={"configurable": {"session_id": "holidays"}},
    )
    response_json = json.dumps(response.content, indent=4, ensure_ascii=False).encode('utf8').decode().replace("```json\\n","").replace("\\n```","")
    response_json = "{ \"Result\": [" + response_json.replace("\\n","").replace("\\","").replace("\"{","{").replace("}\"","}") + " ]}"
    response2 = json.loads(response_json)
    return response2

def generate_hw04(question):
    image_path = './baseball.png'
    image_data_url = local_image_to_data_url(image_path)

    # Define the message with an image and text
    messages = [
        SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": "Please parse the data from the image to get the country name and point. Answer the question based on the specific country's point from the image. Only return the point.",
                },
            ],
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_data_url},},
            ],
        )
    ]
    response = llm.invoke(messages)

    score = int(response.content.strip())
    result = json.dumps({"Result": {"score": score}}, indent=2)
    return result

def get_year(question):
    response = llm.invoke([
        SystemMessage(content="你只能回答問題中的年份, 並以數字表示"),
        HumanMessage(content=question),
    ])
    return response.content

def get_month(question):
    response = llm.invoke([
        SystemMessage(content="你只能回答問題中的月份, 並以數字表示"),
        HumanMessage(content=question),
    ])
    return response.content

#@tool
def get_holidays(country, year, month):
    """
    Retrieves holidays for a given country, year, and month using the Calendarific API.

    Parameters:
        country (str): The country code (e.g., 'US' for the United States).
        year (int): The year (e.g., 2023).
        month (int): The month (e.g., 10 for October).

    Returns:
        list: A list of holidays with details.
    """
    base_url = "https://calendarific.com/api/v2/holidays"
    api_key = "oYvP6SX9B9c6OrCwvyoWqAEPU3XHoXHK"
    params = {
        'api_key': api_key,
        'country': country,
        'year': year,
        'month': month
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        holidays = response.json().get('response', {}).get('holidays', [])
    else:
        return response.raise_for_status()

    results = []
    for holiday in holidays:
        name = holiday.get("name", "Unknown Holiday")
        date = holiday.get("date", {}).get("iso", "Unknown Date")
        results.append({"date": date, "name": name})

    results_json = json.dumps(results)
    results_json = "{ \"Result\": " + results_json + " }"
    return results_json

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

#print(demo("2024年台灣10月紀念日有哪些?"))
#print(generate_hw01("2024年台灣4月紀念日有哪些?"))
#print(generate_hw02("2024年台灣4月紀念日有哪些?"))
#print(get_holidays("TW", 2024, 10))
#print(generate_hw03("2024年台灣4月紀念日有哪些?", '根據先前的節日清單，這個節日是否有在該月份清單？{"date": "4-4", "name": "兒童節"}'))
#print(generate_hw04("請問中華台北的積分是多少?"))
