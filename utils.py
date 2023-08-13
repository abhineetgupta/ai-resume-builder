import logging
import os
import subprocess
import sys

# import pdflatex
from jinja2 import Environment, FileSystemLoader
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO
from ruamel.yaml.error import YAMLError

# create logger
logger = logging.getLogger(__name__)

# create yaml
yaml = YAML()


def read_yaml(yaml_text: str = "", filename: str = "") -> dict:
    if not yaml_text and not filename:
        logger.warning("Neither yaml text nor filename have been provided.")
        return None
    if yaml_text:
        try:
            return yaml.load(yaml_text)
        except YAMLError as e:
            logger.error(f"The text could not be read.")
            raise e
    with open(filename, "r") as stream:
        try:
            return yaml.load(stream)
        except YAMLError as e:
            logger.error(f"The {filename} could not be read.")
            raise e
        except Exception as e:
            logger.error(
                f"The {filename} could not be written due to an unknown error."
            )
            raise e


def read_jobfile(filename: str) -> str:
    with open(filename, "rb") as stream:
        try:
            return stream.read()
        except OSError as e:
            logger.error(f"The {filename} could not be read.")
            raise e


def write_yaml(d: dict, filename: str = None) -> None:
    yaml.allow_unicode = True
    if filename:
        with open(filename, "wb") as stream:
            try:
                yaml.dump(d, stream)
            except YAMLError as e:
                logger.error(f"The {filename} could not be written.")
                raise e
            except Exception as e:
                logger.error(
                    f"The {filename} could not be written due to an unknown error."
                )
                raise e
    return yaml.dump(d, sys.stdout)


def dict_to_yaml_string(d: dict) -> str:
    yaml.allow_unicode = True
    stream = StringIO()
    yaml.dump(d, stream=stream)
    return stream.getvalue()


def generator_key_in_nested_dict(keys: [str | list], nested: dict):
    if hasattr(nested, "items"):
        for k, v in nested.items():
            if (isinstance(keys, list) and k in keys) or k == keys:
                yield v
            if isinstance(v, dict):
                for result in generator_key_in_nested_dict(keys, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in generator_key_in_nested_dict(keys, d):
                        yield result


def get_dict_field(field: str, resume: dict):
    try:
        return resume[field]
    except KeyError as e:
        message = f"`{field}` is missing in raw resume."
        logger.warning(message)
    return None


def generate_pdf(yaml_file: str, template_file: str = None) -> str:
    # set default template file. file location is relative to `templates` directory
    if not template_file:
        template_file = "resume.tex"

    dirname, basename = os.path.split(yaml_file)
    filename, ext = os.path.splitext(basename)
    filename = os.path.join(dirname, filename)

    yaml_data = read_yaml(filename=yaml_file)

    # Set up jinja environment and template
    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        block_start_string="\BLOCK{",
        block_end_string="}",
        variable_start_string="\VAR{",
        variable_end_string="}",
        comment_start_string="\#{",
        comment_end_string="}",
        line_statement_prefix="%%",
        line_comment_prefix="%#",
        autoescape=False,
        loader=FileSystemLoader("templates"),
    )
    template = env.get_template(template_file)

    yaml_data = read_yaml(filename=yaml_file)
    latex_string = template.render(**yaml_data)

    with open(f"{filename}.tex", "wt") as stream:
        stream.write(latex_string)

    # convert to pdf and clean up temp files
    jobname = os.path.join(dirname, "latexmk_temp")
    cmd = subprocess.run(
        [
            "latexmk",
            "-pdf",
            f"-jobname={jobname}",
            f"{filename}.tex",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # rename pdf
    if os.path.isfile(f"{jobname}.pdf"):
        os.rename(f"{jobname}.pdf", f"{filename}.pdf")
        cmd = subprocess.run(
            [
                "latexmk",
                "-c",
                f"-jobname={jobname}",
                f"{filename}.tex",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        message = f"PDF could not be generated. See latexmk logs {jobname}.log"
        logger.error(message)
        raise ValueError(message)
    return f"{filename}.pdf"
