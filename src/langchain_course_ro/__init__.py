from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 




load_dotenv(override=True)


def main():
    print("Hello from langchain-course-ro!")
   
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is an industrialist, futurist and former United States public official who is the chief executive officer and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and became the only trillionaire (in terms of US dollars) in June 2026; as of November 2026, Forbes estimates his net worth to be $849.7 billion.

Musk was born into the wealthy Musk family in Pretoria, South Africa. He emigrated to Canada in 1989; he has Canadian citizenship as his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California, United States, to pursue business ventures. In 1995, he co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.

In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company. Musk joined Tesla as an angel investor in 2004 and became its CEO and product architect in 2008. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service, and made significant changes, including rebranding it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion. Musk became the first US-dollar trillionaire upon the initial public offering of SpaceX in June 2026, though a SpaceX stock collapse brought him under the threshold by the next month.
    """
    
    summary_template = """ 
    Given the information {information} abot a person, I want you to create:
    1. A short summary of the person in 1-2 sentences.
    2. Two interesting facts about the person.
    """

    summary_prompt = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    #llm = ChatOpenAI(model_name="gpt-5", temperature=0)
    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    chain = summary_prompt | llm
    response = chain.invoke({"information": information})

    print(response.content)
   
    
if __name__ == "__main__":
    main()
