from app.services import keywords
from app.utils import catch_exceptions,download_from_s3
import logging
from constants import keywords_model_mapping
import os
from .keywords_generator import KeywordGenerationUsingExtractionAlgo
logger = logging.getLogger(name=__name__)

class GetKeywordsNew:

    def __init__(self) -> None:
        pass
    
    @classmethod
    def check_file_exists(self,abs_file_path):
        try:
            s3_path = abs_file_path
            if not os.path.exists(abs_file_path):
                print('===> abs_path :',abs_file_path)
                os.makedirs("/".join(abs_file_path.split('/')[:-1]), exist_ok=True)
                download_from_s3(s3_path,abs_file_path)

            return abs_file_path
        except Exception as e:
            logger.error(e,exc_info=True)

    @classmethod
    def collector(self,product_title,category):
        annoy_file_path = self.check_file_exists('keywords_models/' + category + '/' + category + '_' + 'annoy.ann')
        csv_file_path = self.check_file_exists('keywords_models/' + category + '/' + category + '_' + 'product_titles.csv')
        scalar_file_path = self.check_file_exists('keywords_models/' + category + '/' + category + '_' + 'scalar.pkl')
        cluster_file_path = self.check_file_exists('keywords_models/' + category + '/' + category + '_' + 'cluster.pkl')

        keygen = KeywordGenerationUsingExtractionAlgo(csv_file_path,scalar_file_path,cluster_file_path,annoy_file_path)

        keywords = keygen.Keyword_Generation_Calling_Function(product_title)

        return keywords

    @classmethod
    def get_keywords(self,request_data):
        try:

            response_data = {}
            missing_fields = []
            mandatory_fields = ["category_name","product_title"]
            for field in mandatory_fields:
                if not field in request_data["data"]:
                    missing_fields.append(field)
            response_data = {
                "status":False,
                "message":"Required field is missing",
                "error_obj":{
                    "description":"{} is/are missing".format(','.join(missing_fields)),
                    "error_code":"REQUIRED_FIELD_IS_MISSING"
                }
            }
            if len(missing_fields) == 0:
                product_title = request_data["data"]["product_title"]
                category = keywords_model_mapping.get(request_data["data"]["category_name"],"beauty")
                if product_title:
                    keywords = self.collector(product_title,category)
                    response_data = {
                        "status":True,
                        "data":{
                        "keywords":keywords
                        }
                    }
                else:
                    response_data = {
                        "status":False,
                        "message":"Product title is not valid",
                        "error_obj":{
                            "description":"Title is not valid",
                            "error_code":"INVALID_TEXT"
                    }
                }

            return response_data
        except Exception as e:
            logger.error(e,exc_info=True)