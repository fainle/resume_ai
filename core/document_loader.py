from langchain_community.document_loaders import JSONLoader


class ResumeJsonLoader(JSONLoader):
    def __init__(self, file_path):
        super().__init__(
            file_path=file_path,
            jq_schema='.',
            text_content=False
        )