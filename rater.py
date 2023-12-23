from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
import json


class ResumeRating(BaseModel):
    rating: int = Field(description="the rating of the resume", ge=0, le=100)
    pros: list[str] = Field(description="the pros of the resume and how it fit the job description, should be as short as possible", max_length=100)
    cons: list[str] = Field(description="the cons of the resume and how it does not fit the job description, should be as short as possible", max_length=100)



class Rater:
    def __init__(self, rating_instructions: str):

        resume_rating_parser = PydanticOutputParser(pydantic_object=ResumeRating)  # type: ignore

        resume_rating_parser = OutputFixingParser.from_llm(
            parser=resume_rating_parser,
            llm=ChatOpenAI(),
        )


        chat_model = ChatOpenAI(model="gpt-4-1106-preview")


        system_message_prompt = SystemMessagePromptTemplate.from_template(
        """you are an expert resume ranker who can score each resume based on "{scoring_instructions}".
        Your output will only be in a raw json format with the following schema: {output_schema}""",
            partial_variables={
                "output_schema": json.dumps(ResumeRating.model_json_schema()),
                "scoring_instructions": rating_instructions,
            }
        )


        human_message_prompt = HumanMessagePromptTemplate.from_template(
            "here is a resume: {resume}"
        )


        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])



        self.chain = chat_prompt | chat_model | resume_rating_parser



    def rate(self, resume_pdf_path: str):
        loader = PyPDFLoader(resume_pdf_path)
        pages = loader.load_and_split()

        resume_content = "\n".join(p.page_content for p in pages)

        result = self.chain.invoke({
            "resume": resume_content,
        })
        assert isinstance(result, ResumeRating)

        return result
