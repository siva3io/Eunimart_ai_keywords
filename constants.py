channel_id_name_mapping = {
    "1"    : "Amazon_USA",
    "12"   : "Amazon_India",
    "14"   : "Flipkart",
    "13"   : "Amazon_UAE",
    "15"   : "Bonanza",
    "3"    : "eBay",
    "24"   : "Etsy"
}
from sentence_transformers import SentenceTransformer 
from nltk.stem import WordNetLemmatizer 
from annoy import AnnoyIndex 
from nltk.corpus import stopwords

keywords_model_mapping = { 
    "Beauty"                    :"beauty",
    "Camera_photo"              :"Cellphones_and_Cameras",
    "Computers"                 :"Computers",
    "Consumer_electronics"      :"Consumer_Electronics",
    "Fashion"                   :"Fashion_and_Clothing",
    "health"                    :"Health",
    "Home"                      :"Home_and_Kitchen",
    "Industrial"                :"Industrial_and_Scientific",
    "Jewellery"                 :"Jewellery",
    "Office"                    :"Office",
    "Outdoors"                  :"Sports_and_Outdoors",
    "Pet_supplies"              :"pet_supplies",
    "Shoes"                     :"Shoes_and_Footwear",
    "Sports"                    :"Sports_and_Outdoors",
    "Toys"                      :"Toys_and_Games"
} 

tags = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW']
stopwords = {'poorly', 'what', 'therein', 'become', 'z', 'hid', 'wasnt', 'significant', "what'll", 'ending', 'yes', 'nos', 'throughout', 'tends', 'eg', 'id', 'invention', "can't", 'himself', 'hes', 'ups', 'mainly', 'section', 'almost', 'couldnt', 'onto', 'index', 'the', 'accordance', 'if', 'others', 'anything', 'o', 'proud', 'rd', 'like', 'past', 'believe', 'downwards', 'auth', 'specified', 'vs', 'fifth', 'follows', 'throug', 'former', 'needs', 'available', 'comes', "they've", 'last', 'somehow', 'formerly', 'said', 'j', 'ml', 'keep', 'tries', 'w', 'use', 'nonetheless', 'inc', 'somethan', 'an', 'tip', 'it', 'yet', 'never', 'stop', 'itd', 'overall', 'ones', 'probably', 'went', 'many', 'nowhere', 'mean', 'nine', 'whenever', 'line', 'really', 'substantially', 'get', 'sent', 'able', 'look', 'usefulness', 'r', 'though', 'twice', 'without', 'arent', 'que', 'during', 'this', 'must', 'in', 'being', 'predominantly', 'following', "didn't", 'obtained', 'whod', 'd', 'na', 'latterly', 'nor', 'myself', 'zero', 'nearly', 'youre', 'toward', 'few', 'per', 'ord', 'along', 'willing', 'anyways', 'gotten', "there've", "i'll", 'see', 'please', 'wont', 'therefore', 'doing', 'hardly', "hasn't", 'begin', 'important', 'welcome', 'either', 'f', 'v', "shouldn't", 'with', 'ever', 'own', 'although', 'anybody', 'beginnings', 'thus', 'done', 'we', 'among', 'otherwise', 'each', 'then', 'there', 'regarding', 'hundred', 'normally', 'begins', 'seeing', 'specify', 'elsewhere', 'thered', 'such', 'www', 'anymore', 'sometimes', 'results', 'is', 'obtain', 'asking', 'down', 'information', 'took', 'whatever', 'words', 'giving', 'g', 'themselves', 'but', 'previously', 'ok', 'present', 'one', 'too', 'liked', 'had', 'on', 'potentially', 'put', 'both', 'tried', 'actually', 'something', 'whereas', 'your', 'are', 'anyone', 'contains', 'towards', 'upon', 'co', 'anyhow', 'affects', 'happens', 'q', 'moreover', 'owing', 'quite', 'his', 'yourself', 'little', 'were', 'm', 'plus', 'take', 'later', 'thanx', 'and', "it'll", 'where', 'am', 'soon', 'whim', 'mug', 'hed', 'largely', 'wherein', 'apparently', 'else', 'first', 'for', 'seeming', 'certain', 'regardless', 'saying', 'five', 'heres', 'became', 'instead', 'thousand', 'unlike', 'next', 'whomever', 'resulted', 'ah', 'around', 'merely', 'thereof', 'knows', 'just', 'against', 'all', 'who', 'across', 'two', 'whither', 'everyone', "that'll", 'immediate', 'as', 'vols', 'want', 'came', 'containing', 'sub', 'indeed', 'back', 'whoever', 'made', 'same', 'affecting', 'e', 'through', 'thence', 'strongly', 'necessarily', 'unfortunately', 'together', 'likely', 'k', 'mrs', 'regards', 'why', 'hereupon', 'refs', 'briefly', 'ie', 'let', 'very', 'than', 'seem', 'does', 'using', 'n', 'aren', 'used', 'non', 'brief', 'necessary', 'related', 'tell', 'date', 'has', 'u', 'usually', 'been', 'werent', 'alone', 'nevertheless', 'keeps', 'whereby', 'ought', 'have', 'primarily', 'ninety', 'go', 'itself', 'beforehand', "they'll", 'particularly', 'miss', 'none', 'still', 'whereupon', 'becoming', 'often', 'within', 'nothing', 'b', 'sup', 'she', 'relatively', 'herein', 'mr', 'aside', 'me', 'namely', 'whom', 'somewhat', 'somebody', 'afterwards', 'c', 'until', "'ll", 'every', 'nay', 'that', 'i', 'several', 'gives', 'looking', 'page', 'useful', 'l', 'shows', 'wherever', 'usefully', 'behind', 'gave', 'importance', 'far', 'added', 'wish', 'pages', "who'll", 'into', 'approximately', 'got', 'also', "i've", 'once', 'promptly', 'nobody', 'cannot', 'was', 'hereby', 'whereafter', 'more', 'those', 'successfully', 'most', 'some', 'six', 'hereafter', 'not', 'say', "we've", 'oh', 'another', 'less', 'sec', 'taken', 'whos', 'thats', 'thereupon', 'however', 'thru', 'world', 'after', 'about', 'h', 'saw', "she'll", "there'll", 'youd', 'did', 'selves', 'getting', 'need', 'awfully', 'noted', 'latter', 'everything', 'howbeit', 'gets', 'over', 'thoughh', 'again', 'makes', 'meantime', 'specifying', 'old', 's', 'hi', 'ran', 'ed', 'now', 'thereby', 'yourselves', 'kg', 'contain', 'wheres', 'much', 'viz', 'show', 'okay', 'thereto', 'here', 'always', 'sorry', 'between', 'hence', 'un', 'may', 'up', 'means', 'various', 'out', 'looks', 'anywhere', 'or', 'vol', 'even', 'widely', 'wouldnt', 'except', "haven't", 'having', 'four', 'so', 'th', 'shall', 'truly', 'ex', 'come', 'value', 'whence', 'whether', 'theyd', 'mostly', 'different', 'possible', 'unless', 'particular', 'gone', 'name', 'p', 'et-al', 'seemed', 'possibly', 'these', 'thou', 'from', 'ask', 'he', 'etc', 'hither', 'edu', 'whats', 'furthermore', "you'll", 'before', 'accordingly', 'him', 'im', 'everywhere', 'found', 'you', 'part', 'whose', 'abst', 'y', 'noone', 'their', 'theirs', 'placed', 'eighty', 'lets', 'seen', 'biol', 'ours', 'lately', 'recent', 'quickly', 'beside', 'resulting', 'near', 'ref', 'shes', 'suggest', 'adj', 'qv', 'anyway', 'arise', 'beginning', 'unto', 'no', 'says', 'somewhere', 'immediately', 'they', 'ca', 'inward', 'respectively', 'enough', 'can', 'million', 'known', 'above', 'home', 'below', 'mg', 'showns', 'trying', 'beyond', 'other', 'similar', 'announce', 'hers', 'of', 'them', 'certainly', 'would', 'similarly', 'thank', 'ltd', 'ourselves', 'yours', 'her', 'when', 'neither', 'due', 'fix', 'theyre', 'provides', 'sufficiently', 'make', 'com', 'know', 'showed', 'its', 'my', 'because', 'lest', 'omitted', 'outside', 'affected', 'could', 'way', 'effect', 'km', 'uses', 'think', 'run', 'goes', 'ff', 'specifically', "doesn't", 'while', 'least', "you've", 'seven', 'sometime', 'eight', "isn't", 'maybe', 'might', 'at', 'away', 'given', 'do', 'nd', 'us', 'our', 'perhaps', 't', 'act', "we'll", 'becomes', 'theres', 'whole', 'slightly', 'taking', 'a', 'shown', 'since', 'x', 'off', 'shed', 'right', 'any', 'by', 'self', 'forth', 'causes', 'readily', 'herself', 'til', 'which', 'research', 'significantly', 'under', 'to', 'should', 'et', "'ve", 'only', 'ts', 'cause', 'kept', 'meanwhile', 'end', 'therere', 'followed', 'thanks', 'thereafter', 'besides', 're', 'wants', 'rather', 'everybody', 'obviously', "don't", 'recently', 'be', 'further', 'according', 'how', "that've", 'already', 'someone', 'especially', 'give', 'amongst', 'unlikely', 'wed', 'try', 'pp', 'sure', 'via', 'seems', 'new'}
stopwords.update(['generic','pack','gram','grams','combo','fl',"count",'set','free','pcs','false','true','type','packaging','vary','bonus','twin','thrice','double','cases','case','ounce'])
Noun_tags = ['NN','NNS','NNP','NNPS']; adj_tags = ['JJ','JJR','JJS'] 
stopwords.difference_update(['and','non']) 
punctuation = ['!','\"','#','$','%','&',"'",'(',')','*','+','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']
stopwords.update(punctuation)
lemmatizer = WordNetLemmatizer()

bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

Annoy = AnnoyIndex(768,'angular')