import logging
import os
from datetime import datetime
from typing import List

import langchain
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from langchain import LLMChain
from langchain.cache import InMemoryCache
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
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

# Set up LLM cache
langchain.llm_cache = InMemoryCache()


def create_llm(**kwargs):
    # set LLM provider
    chat_model = kwargs.pop("chat_model", ChatOpenAI)
    # set default model
    if "model_name" not in kwargs:
        kwargs["model_name"] = "gpt-3.5-turbo"
    # set default cache to False so reruns generate new outputs
    if "cache" not in kwargs:
        kwargs["cache"] = False
    return chat_model(**kwargs)


def format_list_as_string(l: list, list_sep: str = "\n- ") -> str:
    if isinstance(l, list):
        return list_sep + list_sep.join(l)
    return str(l)


def format_prompt_inputs_as_strings(prompt_inputs: list[str], **kwargs):
    """Convert values to string for all keys in kwargs matching list in prompt inputs"""
    return {
        k: format_list_as_string(v) for k, v in kwargs.items() if k in prompt_inputs
    }


def parse_date(d: str) -> datetime:
    """Given an arbitrary string, parse it to a date"""
    # set default date to January 1 of current year
    default_date = datetime(datetime.today().year, 1, 1)
    try:
        return dateparser.parse(str(d), default=default_date)
    except dateparser._parser.ParserError as e:
        langchain.llm_cache.clear()
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
    salary: str = Field(
        ...,
        description="Salary amount or range. Salary should be null if it's not known.",
    )
    duties: List[str] = Field(
        ...,
        description="The role, responsibilities and duties of the job as an itemized list, not exceeding 500 words",
    )
    qualifications: List[str] = Field(
        ...,
        description="The qualifications, skills, and experience required for the job as an itemized list, not exceeding 500 words",
    )
    is_fully_remote: bool = Field(
        ...,
        description="Does the job have an option to work fully (100%) remotely? Hybrid or partial remote is marked as `False`. Use `None` if the answer is not known.",
    )


class Job_Skills(BaseModel):
    """Skills from a job posting"""

    technical_skills: List[str] = Field(
        ...,
        description="An itemized list of technical skills, including programming languages, technologies, and tools. Examples: Python, MS Office, Machine learning, Marketing, Optimization, GPT",
    )
    non_technical_skills: List[str] = Field(
        ...,
        description="An itemized list of non-technical Soft skills. Examples: Communication, Leadership, Adaptability, Teamwork, Problem solving, Critical thinking, Time management",
    )


# Pydantic class that defines each highlight to be returned by the LLM
class Resume_Section_Highlight(BaseModel):
    highlight: str = Field(..., description="one highlight")
    relevance: int = Field(
        ..., description="relevance of the bullet point", enum=[1, 2, 3, 4, 5]
    )


# Pydantic class that defines a list of highlights to be returned by the LLM
class Resume_Section_Highlighter_Output(BaseModel):
    plan: List[str] = Field(..., description="itemized <Plan>")
    additional_steps: List[str] = Field(..., description="itemized <Additional Steps>")
    work: List[str] = Field(..., description="itemized <Work>")
    final_answer: List[Resume_Section_Highlight] = Field(
        ..., description="itemized <Final Answer> in the correct format"
    )


# Pydantic class that defines a list of skills to be returned by the LLM
class Resume_Skills(BaseModel):
    technical_skills: List[str] = Field(
        ...,
        description="An itemized list of technical skills",
    )
    non_technical_skills: List[str] = Field(
        ...,
        description="An itemized list of non-technical skills",
    )


# Pydantic class that defines a list of skills to be returned by the LLM
class Resume_Skills_Matcher_Output(BaseModel):
    plan: List[str] = Field(..., description="itemized <Plan>")
    additional_steps: List[str] = Field(..., description="itemized <Additional Steps>")
    work: List[str] = Field(..., description="itemized <Work>")
    final_answer: Resume_Skills = Field(
        ..., description="<Final Answer> in the correct format"
    )


class Resume_Summarizer_Output(BaseModel):
    plan: List[str] = Field(..., description="itemized <Plan>")
    additional_steps: List[str] = Field(..., description="itemized <Additional Steps>")
    work: List[str] = Field(..., description="itemized <Work>")
    final_answer: str = Field(..., description="<Final Answer> in the correct format")


# Pydantic class that defines a list of improvements to be returned by the LLM
class Resume_Improvements(BaseModel):
    section: str = Field(
        ...,
        enum=[
            "summary",
            "education",
            "experience",
            "projects",
            "skills",
            "spelling and grammar",
            "other",
        ],
    )
    improvements: List[str] = Field(
        ..., description="itemized list of suggested improvements"
    )


class Resume_Improver_Output(BaseModel):
    plan: List[str] = Field(..., description="itemized <Plan>")
    additional_steps: List[str] = Field(..., description="itemized <Additional Steps>")
    work: List[str] = Field(..., description="itemized <Work>")
    final_answer: List[Resume_Improvements] = Field(
        ..., description="<Final Answer> in the correct format"
    )


class Extractor_LLM:
    def __init__(self):
        # The extractor LLM is hard-coded to the cheaper gpt3.5 model
        self.extractor_llm = create_llm(
            chat_model=ChatOpenAI,
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            cache=True,
        )

    def _extractor_chain(self, pydantic_object, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content="You are a world class algorithm for extracting information in structured formats."
            ),
            HumanMessage(
                content="Use the given format to extract information from the following input:"
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
            HumanMessage(content="Tips: Make sure to answer in the correct format."),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        return create_structured_output_chain(
            pydantic_object, llm=self.extractor_llm, prompt=prompt, **chain_kwargs
        )

    def extract_from_input(self, pydantic_object, input: str, **chain_kwargs) -> dict:
        try:
            return (
                self._extractor_chain(pydantic_object=pydantic_object, **chain_kwargs)
                .predict(input=input)
                .dict()
            )
        except Exception as e:
            print("Encountered exception during parsing input. See below:")
            print(e)


class Job_Post(Extractor_LLM):
    def __init__(self, posting: str):
        super().__init__()
        self.posting = posting
        self.parsed_job = None

    def parse_job_post(self, **chain_kwargs) -> dict:
        parsed_job = self.extract_from_input(
            pydantic_object=Job_Description, input=self.posting, **chain_kwargs
        )
        job_skills = self.extract_from_input(
            pydantic_object=Job_Skills,
            input=self.posting,
            **chain_kwargs,
        )
        self.parsed_job = parsed_job | job_skills
        return self.parsed_job


class Resume_Builder(Extractor_LLM):
    def __init__(
        self,
        resume: dict,
        parsed_job: dict,
        is_final: bool = False,
        llm_kwargs: dict = dict(),
    ):
        super().__init__()

        self.resume = resume
        self.parsed_job = parsed_job
        self.llm_kwargs = llm_kwargs

        self.degrees = self._get_degrees(self.resume)
        self.basic_info = utils.get_dict_field(field="basic", resume=self.resume)
        self.education = utils.get_dict_field(field="education", resume=self.resume)
        self.projects = utils.get_dict_field(field="projects", resume=self.resume)
        self.experiences = utils.get_dict_field(field="experiences", resume=self.resume)
        self.skills = utils.get_dict_field(field="skills", resume=self.resume)
        self.summary = utils.get_dict_field(field="summary", resume=self.resume)
        if not is_final:
            self.experiences_raw = self.experiences
            self.experiences = None

            self.skills_raw = self.skills
            self.skills = None

            self.projects_raw = self.projects
            self.projects = None

            self.summary_raw = self.summary
            self.summary = None

    def _section_highlighter_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content=(
                    "You are an expert technical writer. "
                    "Your goal is to strictly follow all the provided <Steps> and meet all the given <Criteria>."
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "<Job Posting>"
                "\nThe ideal candidate is able to perform the following duties:{duties}\n"
                "\nThe ideal candidate has the following qualifications:{qualifications}\n"
                "\nThe ideal candidate has the following skills:{technical_skills}\n{non_technical_skills}"
            ),
            HumanMessagePromptTemplate.from_template("<Master Resume>{section}"),
            HumanMessage(
                content="<Instruction> Identify the relevant portions from the <Master Resume> that match the <Job Posting>, "
                "rephrase these relevant portions into highlights, and rate the relevance of each highlight to the <Job Posting> on a scale of 1-5."
            ),
            HumanMessage(
                content="<Criteria> "
                "\n- Each highlight must be based on what is mentioned in the <Master Resume>."
                # "\n- Include highlights from <Master Resume> that may be tangentially related to the <Job Posting>."
                # "\n- Combine any similar highlights into a single one."
                "\n- In each highlight, include how that experience in the <Master Resume> demonstrates an ability to perform duties mentioned in the <Job Posting>."
                "\n- In each highlight, try to include action verbs, give tangible and concrete examples, and include success metrics when available."
                # "\n- Each highlight must exceed 50 words, and may include more than 1 sentence."
                "\n- Grammar, spellings, and sentence structure must be correct."
            ),
            HumanMessage(
                content=(
                    "<Steps>"
                    "\n- Create a <Plan> for following the <Instruction> while meeting all the <Criteria>."
                    "\n- What <Additional Steps> are needed to follow the <Plan>?"
                    "\n- Follow all steps one by one and show your <Work>."
                    "\n- Verify that highlights are reflective of the <Master Resume> and not the <Job Posting>. Update if necessary."
                    "\n- Verify that all <Criteria> are met, and update if necessary."
                    "\n- Provide the answer to the <Instruction> with prefix <Final Answer>."
                )
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return LLMChain(
            llm=llm,
            prompt=prompt,
            return_final_only=False,
            **chain_kwargs,
        )

    def _skills_matcher_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content=(
                    "You are an expert technical writer. "
                    "Your goal is to strictly follow all the provided <Steps> and meet all the given <Criteria>."
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "<Job Posting>"
                "\nThe ideal candidate has the following skills:{technical_skills}\n{non_technical_skills}"
            ),
            HumanMessagePromptTemplate.from_template(
                "<Resume>" "\nExperience:{projects}\n{experiences}"
            ),
            HumanMessage(
                content="<Instruction> Extract technical and non-technical skills from the <Resume> "
                "that match the skills required in the <Job Posting>."
            ),
            HumanMessage(
                content="<Criteria> "
                "\n- Each skill must be based on what is mentioned in the <Resume>."
                "\n- Technical skills are programming languages, technologies, and tools. Examples: Python, MS Office, Machine learning, Marketing, Optimization, GPT"
                "\n- Non-technical skills are soft skills. Communication, Leadership, Adaptability, Teamwork, Problem solving, Critical thinking, Time management"
                "\n- Each skill must be written in sentence case."
            ),
            HumanMessage(
                content=(
                    "<Steps>"
                    "\n- Create a <Plan> for following the <Instruction> while meeting all the <Criteria>."
                    "\n- What <Additional Steps> are needed to follow the <Plan>?"
                    "\n- Follow all steps one by one and show your <Work>."
                    "\n- Verify that skills are reflective of the <Resume> and not the <Job Posting>. Update if necessary."
                    "\n- Verify that all <Criteria> are met, and update if necessary."
                    "\n- Provide the answer to the <Instruction> with prefix <Final Answer>."
                )
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return LLMChain(
            llm=llm,
            prompt=prompt,
            return_final_only=False,
            **chain_kwargs,
        )

    def _summary_writer_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content=(
                    "You are an expert technical writer. "
                    "Your goal is to strictly follow all the provided <Steps> and meet all the given <Criteria>."
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "<Job Posting>\n{company}\n{job_summary}"
            ),
            HumanMessagePromptTemplate.from_template(
                "<Resume>"
                "\nEducation:{degrees}\n"
                "\nExperience:{projects}\n{experiences}\n"
                "\nSkills:{skills}"
            ),
            HumanMessage(
                content="<Instruction> Create a Professional Summary from my <Resume>."
            ),
            HumanMessage(
                content="<Criteria>"
                "\n- Summary must showcase that I will be ideal for the <Job Posting>."
                "\n- Summary must not exceed 200 words."
                "\n- Summary must highlight my skills and experiences."
                "\n- Include any relevant keywords from the <Job Posting> that also describe my experience from the <Resume>."
                "\n- Grammar, spellings, and sentence structure must be correct."
            ),
            # HumanMessage(
            #     content="<Examples>"
            #     "\n- Technical project manager with 7+ years of experience managing both agile and waterfall projects for large technology organizations. "
            #     "Key strengths include budget management, contract and vendor relations, client-facing communications, stakeholder awareness, and cross-functional "
            #     "team management. Excellent leadership, organization, and communication skills, with special experience bridging large teams and providing process "
            #     "in the face of ambiguity."
            #     "\n- Customer-oriented full sales cycle SMB account executive with 3+ years of experience maximizing sales and crushing quotas. Skilled at "
            #     "building trusted, loyal relationships with high-profile clients, resulting in best-in-class performance for client retention. Passionate "
            #     "about contributing my skills in sales to the role for Account Business Manager at Company."
            # ),
            HumanMessage(
                content=(
                    "<Steps>"
                    "\n- Create a <Plan> for following the <Instruction> while meeting all the <Criteria>."
                    # "\n- Use the <Examples> for writing style typically used for summaries."
                    "\n- What <Additional Steps> are needed to follow the <Plan>?"
                    "\n- Follow all steps one by one and show your <Work>."
                    "\n- Verify that summary is reflective of my <Resume> and not the <Job Posting>. Update if necessary."
                    "\n- Verify that all <Criteria> are met, and update if necessary."
                    "\n- Provide the answer to the <Instruction> with prefix <Final Answer>."
                )
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return LLMChain(
            llm=llm,
            prompt=prompt,
            return_final_only=False,
            **chain_kwargs,
        )

    def _improver_chain(self, **chain_kwargs) -> LLMChain:
        prompt_msgs = [
            SystemMessage(
                content=(
                    "You are an expert critic. "
                    "Your goal is to strictly follow all the provided <Steps> and meet all the given <Criteria>."
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "<Job Posting>"
                "\nThe ideal candidate is able to perform the following duties:{duties}\n"
                "\nThe ideal candidate has the following qualifications:{qualifications}\n"
                "\nThe ideal candidate has the following skills:{technical_skills}\n{non_technical_skills}"
            ),
            HumanMessagePromptTemplate.from_template(
                "<Resume>"
                "\nSummary:{summary}\n"
                "\n{experiences}"
                "\n{projects}"
                "\n{education}"
                "\n{skills}"
            ),
            HumanMessage(
                content="<Instruction> Critique my <Resume>, and suggest how I can improve it "
                "so it showcases that I am the ideal candidate who meets all the requirements of the <Job Posting>."
            ),
            HumanMessage(
                content="<Criteria>"
                "\n- Find any spelling or grammar errors in my <Resume>, and include suggestions to fix them."
                "\n- Compile all your suggestions from <Work> into the <Final Answer>."
                "\n- For each suggestion in the <Final Answer>, include the respective <Resume> section, what needs to be improved, and how to improve it."
                "\n- In the <Final Answer>, include a section for Spelling and Grammar, including any suggestions for improvements."
            ),
            HumanMessage(
                content=(
                    "<Steps>"
                    "\n- Create a <Plan> for following the <Instruction> while meeting all the <Criteria>."
                    "\n- What <Additional Steps> are needed to follow the <Plan>?"
                    "\n- Follow all steps one by one and show your <Work>."
                    "\n- Verify that all <Criteria> are met, and update if necessary."
                    "\n- Provide the answer to the <Instruction> with prefix <Final Answer>."
                )
            ),
        ]
        prompt = ChatPromptTemplate(messages=prompt_msgs)

        llm = create_llm(**self.llm_kwargs)
        return LLMChain(
            llm=llm,
            prompt=prompt,
            return_final_only=False,
            **chain_kwargs,
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
                    last_date = datetime.today().strftime("%Y-%m-%d")
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
                curr += f"{exp_time} years experience in:"
            if "highlights" in exp:
                curr += format_list_as_string(exp["highlights"], list_sep="\n  - ")
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

    def _print_debug_message(self, chain_kwargs: dict, chain_output_unformatted: str):
        message = "Final answer is missing from the chain output."
        if not chain_kwargs.get("verbose"):
            message += "\nChain output:\n" + chain_output_unformatted
        print(message)

    def rewrite_section(self, section: list | str, **chain_kwargs) -> dict:
        chain = self._section_highlighter_chain(**chain_kwargs)
        chain_inputs = format_prompt_inputs_as_strings(
            prompt_inputs=chain.prompt.input_variables,
            **self.parsed_job,
            section=section,
        )
        section_revised_unformatted = chain.predict(**chain_inputs)
        if "verbose" in chain_kwargs and chain_kwargs["verbose"]:
            print("Chain output:\n" + section_revised_unformatted)
        section_revised = self.extract_from_input(
            pydantic_object=Resume_Section_Highlighter_Output,
            input=section_revised_unformatted,
        )
        if not section_revised or "final_answer" not in section_revised:
            self._print_debug_message(chain_kwargs, section_revised_unformatted)
            return None
        # sort section based on relevance in descending order
        section_revised = sorted(
            section_revised["final_answer"], key=lambda d: d["relevance"] * -1
        )
        return [s["highlight"] for s in section_revised]

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

    def extract_matched_skills(self, **chain_kwargs) -> dict:
        chain = self._skills_matcher_chain(**chain_kwargs)
        # We don't use the skills provided in raw resume because the LLM focuses too much n them
        # Instead we will combine those skills with the ones extracted by the LLM
        chain_inputs = format_prompt_inputs_as_strings(
            prompt_inputs=chain.prompt.input_variables,
            **self.parsed_job,
            degrees=self.degrees,
            experiences=self._format_experiences_for_prompt(),
            projects=self.projects,
        )
        extracted_skills_unformatted = chain.predict(**chain_inputs)
        if "verbose" in chain_kwargs and chain_kwargs["verbose"]:
            print("Chain output:\n" + extracted_skills_unformatted)
        extracted_skills = self.extract_from_input(
            pydantic_object=Resume_Skills_Matcher_Output,
            input=extracted_skills_unformatted,
        )
        if not extracted_skills or "final_answer" not in extracted_skills:
            self._print_debug_message(chain_kwargs, extracted_skills_unformatted)
            return None
        extracted_skills = extracted_skills["final_answer"]
        result = []
        if "technical_skills" in extracted_skills:
            result.append(
                dict(category="Technical", skills=extracted_skills["technical_skills"])
            )
        if "non_technical_skills" in extracted_skills:
            result.append(
                dict(
                    category="Non-technical",
                    skills=extracted_skills["non_technical_skills"],
                )
            )
        # Add skills from raw file
        self._combine_skill_lists(result, self.skills_raw)
        return result

    def write_summary(self, **chain_kwargs) -> dict:
        chain = self._summary_writer_chain(**chain_kwargs)
        chain_inputs = format_prompt_inputs_as_strings(
            prompt_inputs=chain.prompt.input_variables,
            **self.parsed_job,
            degrees=self.degrees,
            projects=self.projects,
            experiences=self._format_experiences_for_prompt(),
            skills=self._format_skills_for_prompt(self.skills),
        )
        summary_unformatted = chain.predict(**chain_inputs)
        if "verbose" in chain_kwargs and chain_kwargs["verbose"]:
            print("Chain output:\n" + summary_unformatted)
        summary = self.extract_from_input(
            pydantic_object=Resume_Summarizer_Output, input=summary_unformatted
        )
        if not summary or "final_answer" not in summary:
            self._print_debug_message(chain_kwargs, summary_unformatted)
            return None
        return summary["final_answer"]

    def suggest_improvements(self, **chain_kwargs) -> dict:
        chain = self._improver_chain(**chain_kwargs)
        chain_inputs = format_prompt_inputs_as_strings(
            prompt_inputs=chain.prompt.input_variables,
            **self.parsed_job,
            education=utils.dict_to_yaml_string(dict(Education=self.education)),
            projects=utils.dict_to_yaml_string(dict(Projects=self.projects)),
            summary=self.summary,
            experiences=utils.dict_to_yaml_string(dict(Experiences=self.experiences)),
            skills=utils.dict_to_yaml_string(dict(Skills=self.skills)),
        )
        improvements_unformatted = chain.predict(**chain_inputs)
        if "verbose" in chain_kwargs and chain_kwargs["verbose"]:
            print("Chain output:\n" + improvements_unformatted)
        improvements = self.extract_from_input(
            pydantic_object=Resume_Improver_Output, input=improvements_unformatted
        )
        if not improvements or "final_answer" not in improvements:
            self._print_debug_message(chain_kwargs, improvements_unformatted)
            return None
        return improvements["final_answer"]

    def finalize(self) -> dict:
        return dict(
            basic=self.basic_info,
            summary=self.summary,
            education=self.education,
            experiences=self.experiences,
            projects=self.projects,
            skills=self.skills,
        )
