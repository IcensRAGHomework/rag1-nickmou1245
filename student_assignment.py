import json
import traceback
import requests
from langchain.agents import AgentExecutor, create_openai_functions_agent

from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool

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
    pass
    
def generate_hw04(question):
    pass

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
print(generate_hw01("2024年台灣4月紀念日有哪些?"))
print(generate_hw02("2024年台灣4月紀念日有哪些?"))
#print(get_holidays("TW", 2024, 10))