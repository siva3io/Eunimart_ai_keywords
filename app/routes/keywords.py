import logging
from flask import Blueprint, jsonify, request
from app.services.keywords import GetKeywords
from app.services.keywords import GetKeywordsNew
from app.core import limiter
from flask_limiter.util import get_remote_address
import flask_limiter

keywords = Blueprint('keywords', __name__)

logger = logging.getLogger(__name__)    


@keywords.route('/get_keywords', methods=['POST'])
def get_keywords():
    request_data = request.get_json()
    
    data = GetKeywords.get_keywords(request_data)
    if not data:
        data = {}
    return jsonify(data)

@keywords.route('/dummy', methods=['GET'])
def dummy():
    request_data = request.get_json()
    print("request_data==>",request_data)
    return jsonify({"test":"Working"})


@keywords.route('/get_keywords_new', methods=['POST'])
@limiter.limit('5/day',key_func = flask_limiter.util.get_ipaddr)
def get_keywords_new():
    request_data = request.get_json()
    
    data = GetKeywordsNew.get_keywords(request_data)
    if not data:
        data = {}
    return jsonify(data)