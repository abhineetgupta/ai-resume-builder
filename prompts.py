import logging
import os
from datetime import date, datetime
from typing import Callable, List

from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from langchain import LLMChain
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import utils

# create logger
logger = logging.getLogger(__name__)

# confirm presence of openAI API key
if "OPENAI_API_KEY" not in os.environ:
    logger.info(
        "OPENAI_API_KEY not found in environment. User will be prompted to enter their key."
    )
    prompt = "Enter your OpenAI API key:"
    os.environ["OPENAI_API_KEY"] = input(prompt)


def create_llm(**kwargs):
    # set LLM provider
    chat_model = kwargs.pop("chat_model", ChatOpenAI)
    # set default model
    if "model_name" not in kwargs:
        kwargs["model_name"] = "gpt-3.5-turbo"
    return chat_model(**kwargs)


def format_list_as_string(l: list) -> str:
    if isinstance(l, list):
        return "\n".join(l)
    return str(l)


def format_prompt_inputs_as_strings(prompt_inputs: list[str], **kwargs):
    """Convert values to string for all keys in kwargs matching list in prompt inputs"""
    return {
        k: format_list_as_string(v) for k, v in kwargs.items() if k in prompt_inputs
    }


def parse_date(d: str) -> datetime:
    """Given an arbitrary string, parse it to a date"""
    # set default date to January 1 of current year
    default_date = datetime(date.today().year, 1, 1)
    try:
        return dateparser.parse(str(d), default=default_date)
    except dateparser._parser.ParserError as e:
        logger.error(f"Date input `{d}` could not be parsed.")
        raise e


def datediff_years(start_date: str, end_date: str) -> float:
    """Get difference between arbitrarily formatted dates in fractional years to the floor month"""
    datediff = relativedelta(parse_date(end_date), parse_date(start_date))
    return datediff.years * 1.0 + datediff.months / 12.0


# Pydantic class that defines the format to be returned by the LLM
class Job_Description(BaseModel):
    """Description of a job posting"""

    company: str = Field(
        ..., description="Name of the company that has the job opening"
    )
    job_title: str = Field(..., description="Job title")
    team: str = Field(
        ...,
        description="Name of the team within the company. Team name should be null if it's not known.",
    )
    job_summary: str = Field(
        ..., description="Brief summary of the job, not exceeding 100 words"
    )
    applying_deadline: str = Field(
        ...,
        description="The last date until when job applications will be accepted. Date should be null if it's not known.",
    )
    salary: str = Field(
        ...,
        description="Salary amount or range. Salary should be null if it's not known.",
    )
    duties: List[str] = Field(
        ...,
        description="The roles, responsibilities and duties of the job as an itemized list, not exceeding 500 words",
    )
    skill_and_experience_requirements: List[str] = Field(
        ...,
        description="The skills and experience required for the job as an itemized list, not exceeding 500 words",
    )
    is_fully_remote: bool = Field(
        ...,
        description="Does the job have an option to work fully (100%) remotely? Hybrid or partial remote is marked as `False`. Fully remote should be null if it's not known.",
    )


class Job_Skills(BaseModel):
    """Skills from a job posting"""

    technical_skill_requirements: List[str] = Field(
        ...,
        description="An itemized list of technical skills like programming languages and tools",
    )
    soft_skill_requirements: List[str] = Field(
        ...,
        description="An itemized list of non-technical Soft skills. Some examples of soft skills are: communication, leadership, adaptability, teamwork, problem solving, critical thinking, time management",
    )


# Pydantic class that defines each item to be returned by the LLM
class Resume_Item(BaseModel):
    item: str = Field(..., description="Represents one bullet point")
    relevance: int = Field(..., enum=[1, 2, 3, 4, 5])


# Pydantic class that defines a list of bullet items to be returned by the LLM
class Resume_List(BaseModel):
    items: List[Resume_Item] = Field(..., description="A list of bullet points")


# Pydantic class that defines a list of improvements to be returned by the LLM
class Resume_Improvements(BaseModel):
    # missing_requirements: List[str] = Field(
    #     ...,
    #     description="List of missing education, experience or skills"
    # )
    improvements: List[str] = Field(
        ..., description="List of suggestions for improvement"
    )


# Pydantic class that defines a list of skills to be returned by the LLM
class Resume_Skills(BaseModel):
    software_skills: List[str] = Field(
        ...,
        description="An itemized list of technical skills like programming languages and tools",
    )
    soft_skills: List[str] = Field(
        ...,
        description="An itemized list of non-technical Soft skills. Some examples of soft skills are: communication, leadership, adaptability, teamwork, problem solving, critical thinking, time management",
    )


# Pydantic class that defines the summary to be returned by the LLM
class Resume_Summary(BaseModel):
    summary: str = Field(
        ..., description="Summary of the resume as it relates to the job posting"
    )


class Job_Post:
    def __init__(self, posting: str, **llm_kwargs):
        self.posting = posting
        self.llm_kwargs = llm_kwargs

        self.parsed_job = None

    def _parser_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for extracting information in structured formats."
            ),
            HumanMessage(
                content="Use the given format to extract information from the following input:"
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(
                content="Tips: Make sure to answer in the correct format and stick to any word or item limits."
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return create_structured_output_chain(
            Job_Description, llm, prompt, **chain_kwargs
        )

    def _skills_extractor_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for extracting information in structured formats."
            ),
            HumanMessage(
                content="Use the given format to extract information from the following input:"
            ),
            HumanMessagePromptTemplate.from_template(
                "Job Posting:\n"
                "\nThe ideal candidate is able to perform the following duties:\n{duties}\n"
                "\nThe ideal candidate has the following experience and skills:\n{skill_and_experience_requirements}"
            ),
            HumanMessage(
                content="Tips: Make sure to answer in the correct format and stick to any word or item limits."
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return create_structured_output_chain(Job_Skills, llm, prompt, **chain_kwargs)

    def parse_job_post(self, **chain_kwargs) -> dict:
        parsed_job = self._parser_chain(**chain_kwargs).predict(input=self.posting)
        parsed_job = parsed_job.dict()
        job_skills = self._skills_extractor_chain(**chain_kwargs).predict(**parsed_job)
        self.parsed_job = parsed_job | job_skills.dict()
        return self.parsed_job


class Resume_Builder:
    def __init__(
        self,
        raw_resume: dict,
        job_post: Job_Post,
        **llm_kwargs,
    ):
        # Constants
        self.MAX_SECTION_ITEMS = 5

        self.raw = raw_resume
        self.job_post = job_post
        self.llm_kwargs = llm_kwargs

        # parse job post if not already
        if not self.job_post.parsed_job:
            _ = self.job_post.parse_job_post()

        self.degrees = self._get_degrees(self.raw)
        self.experiences_raw = utils.get_dict_field(
            field="experiences", resume=self.raw
        )
        self.skills_raw = utils.get_dict_field(field="skills", resume=self.raw)
        self.projects_raw = utils.get_dict_field(field="projects", resume=self.raw)

        self.basic_info = utils.get_dict_field(field="basic", resume=self.raw)
        self.education = utils.get_dict_field(field="education", resume=self.raw)
        self.projects = None
        self.experiences = None
        self.skills = None
        self.summary = None

    def _section_rewriter_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content=(
                    "You are an expert Resume Writer proficient in revising a section into key bullet points that directly address the requirements in the job posting. You strictly follow the provided steps."
                )
            ),
            HumanMessage(
                content=(
                    "Your goal is to read and understand a job posting, and then condense and re-phrase the provided Resume text into a few bullet points that demonstrate my expertise and relevance for the job."
                )
            ),
            HumanMessage(
                content=(
                    "\nWe will follow the following steps:"
                    "\nStep 1: I will give you a job posting."
                    "\nStep 2: Then I will give you input text from my Resume."
                    "\nStep 3: You must condense and rephrase the Resume input from Step 2 into key bullet points that meet the following criteria:"
                    "\n    Criterion 1: Each bullet point must demonstrate my expertise and skills for the job posting."
                    "\n    Criterion 2: Each bullet point must be written such that there is a direct link from it to the duties, experience, and skill requirements mentioned in the job posting. It must include language and keywords from the job posting."
                    "\n    Criterion 3: Use action verbs. Give tangible and concrete examples, and include success metrics when available."
                    "\n    Criterion 4: Find keywords such as leader, teamplayer, expert, etc. from the job posting. In each bullet point, include these words to describe my skills and experience such that that it makes me stand out from other candidates applying to this job."
                    "\n    Criterion 5: Do not provide specific names of people or companies from my Resume input. Instead focus on what I accomplished and write about how my experience would allow me to excel at the job."
                    f"\n    Criterion 6: You must limit to no more than {self.MAX_SECTION_ITEMS} bullet points."
                    "\nStep 4: Finally you must rate the relevance of each bullet point to the job posting between 1 and 5."
                )
            ),
            HumanMessage(content=("Let us begin...")),
            HumanMessage(
                content="Step 1: I am providing the job posting, which includes four sections about the job - <Duties>, <Experience requirements>,  <Technical skills>, and <Non-technical skills>. The entire job posting is enclosed in three backticks:"
            ),
            HumanMessagePromptTemplate.from_template(
                "``` Job Posting:\n"
                "\n<Duties>\n{duties}\n"
                "\n<Experience requirements>\n{skill_and_experience_requirements}\n"
                "\n<Technical skills>\n{technical_skill_requirements}\n"
                "\n<Non-technical skills>\n{soft_skill_requirements}\n"
                "```"
            ),
            HumanMessage(
                content="Step 2: I am providing input from my Resume that could include the tasks that I completed, and any feedback that I received from my manager and peers. It is enclosed in four backticks:"
            ),
            HumanMessagePromptTemplate.from_template(
                "```` Section from my Resume:\n{section}\n````"
            ),
            HumanMessage(
                content="Now complete the rest of the steps, starting at Step 3."
            ),
            HumanMessage(
                content="Tips: Make sure to answer in the correct format, match all listed criteria, and stick to any word or item limits."
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return create_structured_output_chain(Resume_List, llm, prompt, **chain_kwargs)

    def _skill_selector_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content=(
                    "You are an expert Resume Reviewer proficient in extracting information a Resume. You strictly follow the provided steps."
                )
            ),
            HumanMessage(
                content=(
                    "Your goal is to read and understand both a job posting and my Resume, and extract a list of those skills that I possess which are the most relevant for the job."
                )
            ),
            HumanMessage(
                content=(
                    "\nWe will follow the following steps:"
                    "\nStep 1: I will give you the job posting."
                    "\nStep 2: Then I will give you my Resume."
                    "\nStep 3: You must extract a list of Technical Skills from my Resume in Step 2 that match the skills in the job posting from Step 1. Limit this list to the 10 most relevant Software Skills."
                    "\nStep 4: You must extract a list of non-technical Soft Skills from my Resume in Step 2 that match the non-technical soft skills in the job posting from Step 1. Limit this list to the 10 most relevant Soft Skills."
                )
            ),
            HumanMessage(content=("Let us begin...")),
            HumanMessage(
                content="Step 1: I am providing the job posting, which includes two sections about the job - <Technical skills> and <Non-technical skills>. The entire job posting is enclosed in three backticks:"
            ),
            HumanMessagePromptTemplate.from_template(
                "``` Job Posting:"
                "\n<Technical skills>\n{technical_skill_requirements}\n"
                "\n<Non-technical skills>\n{soft_skill_requirements}\n"
                "```"
            ),
            HumanMessage(
                content="Step 2: I am providing my Resume, which includes two sections: <My Work Experience> and <My Projects>. My entire Resume is enclosed in four backticks:"
            ),
            HumanMessagePromptTemplate.from_template(
                "```` My Resume:"
                "\n<My Work Experience>\n{experiences}\n"
                "\n<My Projects>\n{projects}\n"
                "````"
            ),
            HumanMessage(
                content="Now complete the rest of the steps starting at Step 3, including any sub-steps."
            ),
            HumanMessage(
                content="Tips: Make sure to answer in the correct format and  and stick to any word or item limits."
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return create_structured_output_chain(
            Resume_Skills, llm, prompt, **chain_kwargs
        )

    def _improver_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content=(
                    "You are an expert Resume Reviewer proficient in making suggestions for improvements to a Resume. You strictly follow the provided steps."
                )
            ),
            HumanMessage(
                content=(
                    "Your goal is to read and understand both a job posting and my Resume, identify the requirements from the job posting that are missing in my Resume, and finally make suggestions for improvements."
                )
            ),
            HumanMessage(
                content=(
                    "\nWe will follow the following steps:"
                    "\nStep 1: I will give you a job posting."
                    "\nStep 2: Then I will give you my Resume."
                    "\nStep 3: You must create an internal list of those requirements from the job posting from Step 1 that are NOT present in my Resume from Step 2. Follow these sub-steps:"
                    "\n   Sub-Step 3a: Identify an internal list of those requirements from the job posting that are NOT present in my Resume."
                    "\n   Sub-Step 3b: For each missing requirement that you identified in Sub-Step 3a, check again whether that requirement is in fact present in my Resume. Note that a requirement may be present in my Resume in a different phrasing than in the job posting. If the requirement is present in my Resume, remove it from your list of missing requirements."
                    "\n   Sub-Step 3c: Sub-steps are complete. Move to the next Step."
                    "\nStep 4: You must finally create a list of suggestions for improving my Resume based on the final list of missing requirements from the previous step, and the job posting."
                )
            ),
            HumanMessage(content=("Let us begin...")),
            HumanMessage(
                content="Step 1: I am providing the job posting, which includes two sections about the job - <Duties> and <Experience requirements>. The entire job posting is enclosed in three backticks:"
            ),
            HumanMessagePromptTemplate.from_template(
                "``` Job Posting:"
                "\n<Duties>\n{duties}\n"
                "\n<Experience requirements>\n{skill_and_experience_requirements}\n"
                "```"
            ),
            HumanMessage(
                content="Step 2: I am providing my Resume, which includes four sections - <My Education Degrees>, <My Work Experience>, <My Projects>, and <My Skills>. My entire Resume is enclosed in four backticks:"
            ),
            HumanMessagePromptTemplate.from_template(
                "```` My Resume:"
                "\n<My Education Degrees>\n{degrees}\n"
                "\n<My Work Experience>\n{experiences}\n"
                "\n<My Projects>\n{projects}\n"
                "\n<My Skills>\n{skills}\n"
                "````"
            ),
            HumanMessage(
                content="Now complete the rest of the steps starting at Step 3, including any sub-steps."
            ),
            HumanMessage(
                content="Tips: Make sure to answer in the correct format, and stick to any word or item limits."
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return create_structured_output_chain(
            Resume_Improvements, llm, prompt, **chain_kwargs
        )

    def _summary_writer_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content=(
                    "You are an expert Resume Writer proficient in summarizing a resume based on a job posting. You strictly follow the provided steps."
                )
            ),
            HumanMessage(
                content=(
                    "Your goal is to read and understand both a job posting and my Resume, and summarize the Resume for applying to the job."
                )
            ),
            HumanMessage(
                content=(
                    "\nWe will follow the following steps:"
                    "\nStep 1: I will give you a job posting."
                    "\nStep 2: Then I will give you my Resume."
                    "\nStep 3: You must summarize my Resume from Step 2 into a short paragraph that highlights my accomplishments, relevant skills, experience, "
                    "expertise, and other credentials that demonstrate how I am the most suitable candidate to apply for the job. "
                    "The Summary must not exceed 80 words. I am providing below some examples to familiarize you with the writing style:"
                    # "\nSummary Example:"
                    # "\nAccomplished and enthusiastic teacher with 8+ years of experience in elementary and middle schools. Versatile background includes teaching history, geography, English, and social studies while supporting student growth and development using age-appropriate learning tools. Calm and flexible attitude creates an atmosphere that encourages student learning, innovation, and imagination. Recognized for fostering inclusive classrooms where all students are welcomed and supported."
                    "\nSummary Example:"
                    "\nTechnical project manager with over seven years of experience managing both agile and waterfall projects for large technology organizations. Key strengths include budget management, contract and vendor relations, client-facing communications, stakeholder awareness, and cross-functional team management. Excellent leadership, organization, and communication skills, with special experience bridging large teams and providing process in the face of ambiguity."
                )
            ),
            HumanMessage(content=("Let us begin...")),
            HumanMessage(
                content="Step 1: I am providing a summary of the job posting, enclosed in three backticks:"
            ),
            HumanMessagePromptTemplate.from_template(
                "``` Job Posting: {job_summary} ```"
            ),
            HumanMessage(
                content="Step 2: I am providing my Resume, which includes four sections - <My Education Degrees>, <My Work Experience>, <My Projects>, and <My Skills>. My entire Resume is enclosed in four backticks:"
            ),
            HumanMessagePromptTemplate.from_template(
                "```` My Resume:"
                "\n<My Education Degrees>\n{degrees}\n"
                "\n<My Work Experience>\n{experiences}\n"
                "\n<My Projects>\n{projects}\n"
                "\n<My Skills>\n{skills}\n"
                "````"
            ),
            HumanMessage(
                content="Now complete the rest of the steps starting at Step 3."
            ),
            HumanMessage(
                content="Tips: Make sure to answer in the correct format, and stick to any word or item limits."
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return create_structured_output_chain(
            Resume_Summary, llm, prompt, **chain_kwargs
        )

    def _get_degrees(self, resume: dict):
        result = []
        for degrees in utils.generator_key_in_nested_dict("degrees", resume):
            for degree in degrees:
                if isinstance(degree["names"], list):
                    result.extend(degree["names"])
                elif isinstance(degree["names"], str):
                    result.append(degree["names"])
        return result

    def _format_skills_for_prompt(self, skills: list) -> list:
        result = []
        for cat in skills:
            curr = ""
            if cat.get("category", ""):
                curr += f"{cat['category']}: "
            if "skills" in cat:
                curr += "Proficient in "
                curr += ", ".join(cat["skills"])
                result.append(curr)
        return result

    def _get_cumulative_time_from_titles(self, titles) -> int:
        result = 0.0
        for t in titles:
            if "startdate" in t and "enddate" in t:
                if t["enddate"] == "current":
                    last_date = date.today().strftime("%Y-%m-%d")
                else:
                    last_date = t["enddate"]
            result += datediff_years(start_date=t["startdate"], end_date=last_date)
        return round(result)

    def _format_experiences_for_prompt(self) -> list:
        result = []
        for exp in self.experiences:
            curr = ""
            if "titles" in exp:
                exp_time = self._get_cumulative_time_from_titles(exp["titles"])
                curr += f"{exp_time} years experience in:\n"
            if "highlights" in exp:
                curr += format_list_as_string(exp["highlights"])
                curr += "\n"
                result.append(curr)
        return result

    def _combine_skills_in_category(self, l1: list[str], l2: list[str]):
        """Combines l2 into l1 without lowercase duplicates"""
        # get lowercase items
        l1_lower = {i.lower() for i in l1}
        for i in l2:
            if i.lower() not in l1_lower:
                l1.append(i)

    def _combine_skill_lists(self, l1: list[dict], l2: list[dict]):
        """Combine l2 skills list into l1 without duplicating lowercase category or skills"""
        l1_categories_lowercase = {s["category"].lower(): i for i, s in enumerate(l1)}
        for s in l2:
            if s["category"].lower() in l1_categories_lowercase:
                self._combine_skills_in_category(
                    l1[l1_categories_lowercase[s["category"].lower()]]["skills"],
                    s["skills"],
                )
            else:
                l1.append(s)

    def rewrite_section(self, section: list | str, **chain_kwargs) -> dict:
        chain = self._section_rewriter_chain(**chain_kwargs)
        chain_inputs = format_prompt_inputs_as_strings(
            prompt_inputs=chain.prompt.input_variables,
            **self.job_post.parsed_job,
            section=section,
        )
        section_revised = chain.predict(**chain_inputs).dict()
        # sort section based on relevance in descending order
        section_revised = sorted(
            section_revised["items"], key=lambda d: d["relevance"] * -1
        )
        return [s["item"] for s in section_revised]

    def rewrite_unedited_experiences(self, **chain_kwargs) -> dict:
        result = []
        for exp_raw in self.experiences_raw:
            # create copy of raw experience to update
            exp = dict(exp_raw)
            experience_unedited = exp.pop("unedited", None)
            if experience_unedited:
                # rewrite experience using llm
                exp["highlights"] = self.rewrite_section(
                    section=experience_unedited, **chain_kwargs
                )
            result.append(exp)

        return result

    def extract_skills(self, **chain_kwargs) -> dict:
        chain = self._skill_selector_chain(**chain_kwargs)
        chain_inputs = format_prompt_inputs_as_strings(
            prompt_inputs=chain.prompt.input_variables,
            **self.job_post.parsed_job,
            degrees=self.degrees,
            experiences=self._format_experiences_for_prompt(),
            skills=self._format_skills_for_prompt(self.skills_raw),
            projects=self.projects,
        )

        extracted_skills = chain.predict(**chain_inputs).dict()
        result = []
        if "software_skills" in extracted_skills:
            result.append(
                dict(category="Technical", skills=extracted_skills["software_skills"])
            )
        if "soft_skills" in extracted_skills:
            result.append(
                dict(category="Non-technical", skills=extracted_skills["soft_skills"])
            )
        # Add skills from raw file
        self._combine_skill_lists(result, self.skills_raw)
        return result

    def suggest_improvements(self, **chain_kwargs) -> dict:
        chain = self._improver_chain(**chain_kwargs)
        chain_inputs = format_prompt_inputs_as_strings(
            prompt_inputs=chain.prompt.input_variables,
            **self.job_post.parsed_job,
            degrees=self.degrees,
            projects=self.projects,
            experiences=self._format_experiences_for_prompt(),
            skills=self._format_skills_for_prompt(self.skills),
        )

        return chain.predict(**chain_inputs).dict()["improvements"]

    def create_summary(self, **chain_kwargs) -> dict:
        chain = self._summary_writer_chain(**chain_kwargs)
        chain_inputs = format_prompt_inputs_as_strings(
            prompt_inputs=chain.prompt.input_variables,
            **self.job_post.parsed_job,
            degrees=self.degrees,
            projects=self.projects,
            experiences=self._format_experiences_for_prompt(),
            skills=self._format_skills_for_prompt(self.skills),
        )

        return chain.predict(**chain_inputs).dict()["summary"]

    def finalize(self) -> dict:
        return dict(
            basic=self.basic_info,
            summary=self.summary,
            education=self.education,
            projects=self.projects,
            experiences=self.experiences,
            skills=self.skills,
        )
