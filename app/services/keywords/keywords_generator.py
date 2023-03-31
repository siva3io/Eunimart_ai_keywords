import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import pos_tag
import joblib
from tqdm.notebook import tqdm
import numpy as np
import re
import math
import scipy
import string
import constants

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

class KeyphraseSelectionPerSingleQueryTitle:
    
    def __init__(self):
        pass
        
    def Replace_Digit_With_Pipe(self, query: str) -> str:
        return re.sub(r'\s\d+\s','|',query)
    
    def Split_Query_With_Given_Token(self, query: str, token: chr) -> list:
        return query.split(token)
    
    def Word_Tokenization_Using_POS_Tags(self, split_query: str) -> list:   # POS implies parts of speech
        return [pos_tag([each_word]) for each_word in word_tokenize(split_query)]
    
    def Validating_Hyphenating_phrases(self, hyphen_phrases: str) -> bool:
        word_hyphen_split = hyphen_phrases.split('-')
        if word_hyphen_split[0].isdigit() or len(word_hyphen_split[0])<3 or word_hyphen_split[1].isdigit() or len(word_hyphen_split[1])<=3:
            return False
        else: return True
    
    def Hyphenating_phrases_Extraction(self, query: str) -> list:
        hyphenating_phrases = re.findall(r'\w+-\w+\s', query)
        valid_hyphenating_phrases = []
        for each_hyphen_phrases in hyphenating_phrases:
            if self.Validating_Hyphenating_phrases(each_hyphen_phrases):
                valid_hyphenating_phrases.append(each_hyphen_phrases)
        return valid_hyphenating_phrases
        
    def Lemmatize_Tokenized_Words(self, tokenize: list) -> str:
        return ' '.join([constants.lemmatizer.lemmatize(tokenized_word[0][0]) for tokenized_word in tokenize])
    
    
    def Remove_Text_In_Brackets(self, query: str) -> list:
        extracted_phrases, query = re.findall(r'[\(\[][\w+\-\s%?#\+=,.]+[\)\]]',query), re.sub(r'[\(\[][\w+\-\s%?#\+=,.]+[\)\]]','',query)
        for loop_iterator in extracted_phrases:
            query+='| '+loop_iterator[1:-1]+' '
        return query
    
    def Remove_Abbrevations_In_Text(self, query: str) -> str:
        return re.sub(r'[A-Z_]+?\d+','',query) 
        
    def Parts_Of_Speech_Tagging(self, query: str) -> list:
        return [pos_tag([word]) for word in query.split()]
    
    def Extract_Words_Length_gtreq_12(self, query: str) -> list:
        extract_words = []
        for each_word in query.split(' '): 
            if len(each_word)>=11: extract_words.append(each_word)
        return extract_words
    
    def POS_Tagging_Parsing(self, pos_words_list: list) -> list:
        parsed_keyphrases = []; keyphrase_extraction = ''; iterate_upto_length_lessthan_3 = 0
        for iterating_each_pos_tags in pos_words_list:
            word = iterating_each_pos_tags[0][0]; pos_tag_of_word = iterating_each_pos_tags[0][1]
            if pos_tag_of_word in constants.tags and word not in constants.stopwords and len(word)>=3:  # if number of phrases greater than 3 we can stop and take that as a keyphrase.
                keyphrase_extraction+=word+' '
                if len(word)>=12: parsed_keyphrases.extend([word])
            else:
                if word == 'and':    # if last but not least is 'and' keyword then we can take the whole as a keyphrase
                    if iterate_upto_length_lessthan_3==len(pos_words_list)-2:
                        keyphrase_extraction+='&'+' '+pos_words_list[iterate_upto_length_lessthan_3+1][0][0]
                        if len(keyphrase_extraction)>1: parsed_keyphrases.extend([keyphrase_extraction.strip()]); break
                else:       # if length of words is less than 3 we need to update the keyphrase with current word
                    if len(keyphrase_extraction)>1: parsed_keyphrases.extend([keyphrase_extraction.strip()]); keyphrase_extraction = '';
            iterate_upto_length_lessthan_3+=1
        if len(keyphrase_extraction)>1: parsed_keyphrases.extend([keyphrase_extraction.strip()]); keyphrase_extraction = '';  
        return parsed_keyphrases
    
    def Extracting_Adjectives_FollowedBy_Noun(self, current_iterator, pos_words_list: list):
        iterate_upto_length_lessthan_2 = 0
        extracted_keyphrase = pos_words_list[current_iterator+1][0]
        adjective_combo_noun_keyphrase = pos_words_list[current_iterator][0]+' '
        for loop_iterator in range(current_iterator+1, len(pos_words_list)):
            if pos_words_list[loop_iterator][0]==extracted_keyphrase and iterate_upto_length_lessthan_2<2:
                adjective_combo_noun_keyphrase+=pos_words_list[loop_iterator][0]+' '
                iterate_upto_length_lessthan_2+=1
            else: break
        return adjective_combo_noun_keyphrase, iterate_upto_length_lessthan_2
    
    def Extracting_Noun_FollowedBy_Adjective(self, current_iterator, pos_words_list: list) -> list:
        noun_combo_adjective_keyphrase = []
        if len(pos_words_list)!=1: 
            noun_combo_adjective_keyphrase.append(pos_words_list[current_iterator-1][0]+' '+pos_words_list[current_iterator][0])
        else: noun_combo_adjective_keyphrase.append(pos_words_list[current_iterator-1][0])
        return noun_combo_adjective_keyphrase
    
    def Adjective_Tag_Parsing(self, pos_words_list: list) -> list:
        """
        For keyphrases greater than 3 words in it will take adjectives followed by one or more nouns
        Thumb Rule: 
             NP:{<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)
        """
        loop_iterator = 0; keyphrase_extraction = ''; parsed_keyphrases = []
        while(loop_iterator<len(pos_words_list)):
            word = pos_words_list[loop_iterator][0]; pos_tag_of_word = pos_words_list[loop_iterator][1]
            if loop_iterator!=len(pos_words_list)-1 and pos_tag_of_word in constants.adj_tags and pos_words_list[loop_iterator+1][1] in constants.Noun_tags:
                adjective_combo_noun_keyphrase, increment_iterator = self.Extracting_Adjectives_FollowedBy_Noun(loop_iterator, pos_words_list)
                loop_iterator += increment_iterator
                parsed_keyphrases.extend([adjective_combo_noun_keyphrase])
            if loop_iterator==len(pos_words_list)-1 and pos_tag_of_word in constants.adj_tags:
                noun_combo_adjective_keyphrase = self.Extracting_Noun_FollowedBy_Adjective(loop_iterator, pos_words_list)
                parsed_keyphrases.extend([noun_combo_adjective_keyphrase])
                loop_iterator+=1; continue
            if loop_iterator!=len(pos_words_list)-1 and pos_tag_of_word in constants.adj_tags and pos_words_list[loop_iterator+1][1]=='VBG':
                parsed_keyphrases.extend([pos_words_list[loop_iterator][0]+' '+pos_words_list[loop_iterator+1][0]])
                loop_iterator+=2; continue
            loop_iterator+=1
        return parsed_keyphrases
    
    def Noun_Tag_Parsing(self, pos_words_list: list) -> list:
        parsed_keyphrases = []; loop_iterator = 0
        while(loop_iterator<len(pos_words_list)-1):
            if pos_words_list[loop_iterator][1] in constants.Noun_tags and pos_words_list[loop_iterator+1][1] in constants.Noun_tags:
                parsed_keyphrases.extend([pos_words_list[loop_iterator][0]+" "+pos_words_list[loop_iterator+1][0]])
            if loop_iterator+3<=len(pos_words_list)-1 and pos_words_list[loop_iterator][1] in constants.Noun_tags and (pos_words_list[loop_iterator+1][0]=='and' or pos_words_list[loop_iterator+1][0]=='&') and (pos_words_list[loop_iterator+2][1] in constants.Noun_tags and pos_words_list[loop_iterator+3][1] in constants.Noun_tags):
                parsed_keyphrases.extend([pos_words_list[loop_iterator][0]+' '+pos_words_list[loop_iterator+1][0]+' '+pos_words_list[loop_iterator+2][0]+' '+pos_words_list[loop_iterator+3][0]])
                loop_iterator+=4; continue
            elif pos_words_list[loop_iterator][1]=='VBG' and pos_words_list[loop_iterator+1][1]=='NN':
                parsed_keyphrases.extend([pos_words_list[loop_iterator][0]+' '+pos_words_list[loop_iterator+1][0]])
            elif loop_iterator!=len(pos_words_list)-2 and pos_words_list[loop_iterator][1]=='NNS' and pos_words_list[loop_iterator+2][1]=='NNS' and (pos_words_list[loop_iterator+1][0]=='and' or pos_words_list[loop_iterator+1][0]=='&'):
                print(pos_words_list[loop_iterator][0])
                parsed_keyphrases.extend([pos_words_list[loop_iterator][0]+" "+pos_words_list[loop_iterator+1][0]+" "+pos_words_list[loop_iterator+2][0]])
            loop_iterator+=1
        return parsed_keyphrases
    
    def Adjective_Or_Noun_Tag_Parsing_Calling(self, valid_keyphrase: str) -> list:
        pos_tagging = pos_tag(valid_keyphrase.split(' ')) 
        return self.Adjective_Tag_Parsing(pos_tagging), self.Noun_Tag_Parsing(pos_tagging)
    
    def Validating_Phrases_Else_Eliminating_Phrases(self, Extracted_keywords_list: list) -> list:
        validated_extracted_keyphrases = []
        for each_keyphrase in Extracted_keywords_list:
            if len(each_keyphrase.split())>3:
                valid_adj_keyphrases, valid_noun_keyphrases = self.Adjective_Or_Noun_Tag_Parsing_Calling(each_keyphrase)
                validated_extracted_keyphrases.extend(valid_adj_keyphrases)
                validated_extracted_keyphrases.extend(valid_noun_keyphrases)
            else:
                if len(each_keyphrase.split())>=1: validated_extracted_keyphrases.extend([each_keyphrase])
        validated_extracted_keyphrases_ = []
        for each_keyphrase in validated_extracted_keyphrases:
            if type(each_keyphrase)==str: validated_extracted_keyphrases_.append(each_keyphrase)
            else:validated_extracted_keyphrases_.extend(each_keyphrase)
        return list(set(validated_extracted_keyphrases_))
    
    def separate_punctuation_with_space(self, text: str) -> str:
        processed_text = ""
        for character in text:
            if character in string.punctuation and character!='-': processed_text+=" "+character+" "
            else: processed_text+=character
        return ' '.join(processed_text.split())

    def Preprocess_Query_Title(self, query_splitted_text: str) -> str:
        tokenized_text = self.Word_Tokenization_Using_POS_Tags(query_splitted_text)
        lemmatized_text = self.Lemmatize_Tokenized_Words(tokenized_text)
        removed_abbrevation_text = self.Remove_Abbrevations_In_Text(lemmatized_text).lower()
        processed_text = self.separate_punctuation_with_space(removed_abbrevation_text)
        return processed_text
    
    def Keyword_Selection_Calling_Function(self, query_title: str):
        query_title = self.Remove_Text_In_Brackets(query_title)
        Extracted_keywords_list = []
        for each_pipe_split in self.Split_Query_With_Given_Token(self.Replace_Digit_With_Pipe(query_title),'|'):
            for each_comma_split in self.Split_Query_With_Given_Token(each_pipe_split,','):
                preprocessed_query_title = self.Preprocess_Query_Title(each_comma_split)
                Extracted_keywords_list.extend(self.Extract_Words_Length_gtreq_12(preprocessed_query_title))
                parts_of_speech_tagging = self.Parts_Of_Speech_Tagging(preprocessed_query_title)
                Extracted_keywords_list.extend(self.POS_Tagging_Parsing(parts_of_speech_tagging)) 
                
        Extracted_keywords_list.extend(self.Hyphenating_phrases_Extraction(query_title))
        return self.Validating_Phrases_Else_Eliminating_Phrases(Extracted_keywords_list) 

class KeywordGenerationUsingExtractionAlgo:

    def __init__(self, product_titles_csv_path: str, standardscalar_pkl_file_path: str, kmeans_algo_pkl_file_path: str, annoy_algo_ann_file_path):
        self.product_titles_csv_data = pd.read_csv(product_titles_csv_path)
        self.standardscalar_pkl_file = joblib.load(standardscalar_pkl_file_path)
        self.kmeans_algo_pkl_file = joblib.load(kmeans_algo_pkl_file_path)
        self.annoy_algo_ann_file_path = annoy_algo_ann_file_path
        self.keyphrase_selection = KeyphraseSelectionPerSingleQueryTitle()
        
    
    def preprocessing_test_query(self, test_query: str) -> str:
        remove_unwant_tokens = re.sub(r'[^a-zA-Z]',' ',test_query.lower())
        tokenized_text = [pos_tag(each_word.split(' ')) for each_word in word_tokenize(remove_unwant_tokens)]
        lemmatized_text = [constants.lemmatizer.lemmatize(loop_iterator[0][0]) for loop_iterator in tokenized_text if loop_iterator[0][1] in constants.tags]
        processed_test_query = [loop_iterator for loop_iterator in lemmatized_text if len(loop_iterator)>=3 and loop_iterator not in constants.stopwords]
        return ' '.join(processed_test_query)
    
    def finding_title_embeddings(self, titles_in_cluster: list) -> np.array:
        bert_embeddings_of_titles_in_cluster = np.empty(shape=(len(titles_in_cluster),768))
        for loop_iterator in range(len(titles_in_cluster['Unnamed: 0'])):
            product_title = titles_in_cluster.iloc[loop_iterator]['product_title']
            bert_embeddings_of_titles_in_cluster[loop_iterator] = constants.bert_model.encode([self.preprocessing_test_query(product_title)])
        return bert_embeddings_of_titles_in_cluster
    
    def bert_embeddings(self, word_dictionary: list) -> np.array:
        embedding = np.zeros(shape=(len(word_dictionary),768),dtype='float64')
        for dict_ in range(len(word_dictionary)):
            embedding[dict_] = constants.bert_model.encode(word_dictionary[dict_])
        return embedding
    
    def similarity_search_algorithm(self, test_query: str) -> list: 
        processed_test_query = self.preprocessing_test_query(test_query)
        bert_embedding_of_test_query = constants.bert_model.encode(processed_test_query) 
        standarlized_vector = self.standardscalar_pkl_file.transform(bert_embedding_of_test_query.reshape(1,-1))
        constants.Annoy.load(self.annoy_algo_ann_file_path)
        indexes = constants.Annoy.get_nns_by_vector(bert_embedding_of_test_query,9)
        fetched_similar_product_titles = [] 
        for each_index_in_annoy in indexes: 
            try:
                fetched_similar_product_titles.append(self.product_titles_csv_data.iloc[each_index_in_annoy]['product_title'])
            except:
                pass 
        return fetched_similar_product_titles
    
    def maximum_marginal_relevance_algorithm(self, bert_embeddings_of_titles_in_cluster, bert_embedding_of_test_query, extracted_keyphrases, n_iterator, diversity_in_relevance = 0.5) -> list:
        test_similarity_with_other_titles = cosine_similarity(bert_embedding_of_test_query, bert_embeddings_of_titles_in_cluster)
        extracted_keyphrases_similarity_with_testquery = cosine_similarity(bert_embedding_of_test_query)
        similar_titles_indexes = [np.argmax(test_similarity_with_other_titles)]
        similar_keyphrases_indexes = [loop_iterator for loop_iterator in range(len(extracted_keyphrases)) if loop_iterator!=similar_titles_indexes[0]]
        for loop_iterator in range(n_iterator -1):
            keyphrases_similarity = test_similarity_with_other_titles[similar_keyphrases_indexes,:]
            identified_similarities = np.max(extracted_keyphrases_similarity_with_testquery[similar_keyphrases_indexes][:,similar_titles_indexes],axis=1)
            maximum_marginal_relevance = (1-diversity_in_relevance) * (keyphrases_similarity) - diversity_in_relevance * identified_similarities.reshape(-1,1)
            try:
                maximum_marginal_relevance_indexes = similar_keyphrases_indexes[np.argmax(maximum_marginal_relevance)] 
            except ValueError: 
                return extracted_keyphrases 
            similar_titles_indexes.append(maximum_marginal_relevance_indexes) 
            similar_keyphrases_indexes.remove(maximum_marginal_relevance_indexes) 
        return [extracted_keyphrases[loop_iterator] for loop_iterator in similar_titles_indexes] 
    
    def duplicate_keyphrase_removal(self, generated_keyphrases:list) -> list: 
        for loop_iterator in range(len(generated_keyphrases)):
            keyphrase = generated_keyphrases[loop_iterator].split(' ')
            if len(keyphrase)<=2:
                for sub_loop_iterator in range(len(generated_keyphrases)):
                    if (generated_keyphrases[loop_iterator] in generated_keyphrases[sub_loop_iterator] and loop_iterator!=sub_loop_iterator) or len(generated_keyphrases[loop_iterator].split())<2:
                        generated_keyphrases[loop_iterator] = ''
        keyphrase_generated = []
        for keyphrase in generated_keyphrases:
            if keyphrase!='': keyphrase_generated.append(keyphrase)
        return keyphrase_generated[:7]
    
    def vocabulary_extraction_from_similar_titles(self, similar_titles: list) -> list:
        extracted_vocabulary_from_titles = set()
        for each_title in similar_titles:
            extracted_vocabulary_from_titles.update(self.keyphrase_selection.Keyword_Selection_Calling_Function(each_title))
        return list(extracted_vocabulary_from_titles)
    
    def Keyword_Generation_Calling_Function(self, test_query: str) -> list:
        extracted_keywords_from_test_query = self.keyphrase_selection.Keyword_Selection_Calling_Function(test_query)
        preprocessed_test_query = self.preprocessing_test_query(test_query)
        bert_embedding_of_test_query = constants.bert_model.encode([test_query])
        fetched_similar_titles = self.similarity_search_algorithm(test_query)
        
        processed_keyphrases_from_titles = []
        n_iterator_relevance = math.ceil((40/100)*len(test_query.split(' ')))
        embeddings_of_extracted_keywords_from_test = self.bert_embeddings(extracted_keywords_from_test_query)
        
        maximum_relevant_extraction = self.maximum_marginal_relevance_algorithm(bert_embedding_of_test_query, embeddings_of_extracted_keywords_from_test, extracted_keywords_from_test_query, n_iterator_relevance)
        processed_keyphrases_from_titles.extend(maximum_relevant_extraction)
        
        vocabulary_from_similar_titles = self.vocabulary_extraction_from_similar_titles(fetched_similar_titles)
        vocabulary_keyphrase_embeddings = self.bert_embeddings(vocabulary_from_similar_titles)
        maximum_relevant_vocabulary_extraction = self.maximum_marginal_relevance_algorithm(bert_embedding_of_test_query, vocabulary_keyphrase_embeddings, vocabulary_from_similar_titles, n_iterator_relevance)
        
        generated_keywords = self.duplicate_keyphrase_removal(maximum_relevant_extraction[:8]) + self.duplicate_keyphrase_removal(maximum_relevant_vocabulary_extraction[:5])
        return self.duplicate_keyphrase_removal(generated_keywords)
