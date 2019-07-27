# coding:utf-8
from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import re

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings(action = 'ignore') #忽略警告

import seaborn as sns
import scipy.stats as sts
from sklearn.metrics import classification_report,f1_score
from sklearn.metrics import confusion_matrix



path = "kaggle\\COTR\\"
datas = pd.read_csv(path+'train.csv', iterator=True)
tests =  pd.read_csv(path+'test.csv', iterator=True)
trains = datas.get_chunk(30000)
#trains.dropna(inplace=True)
tests = tests.get_chunk(5000)
features = trains.columns[3:]
features_int = ['identity_annotator_count', 'toxicity_annotator_count', 'disagree', 'likes', 'sad',
                'wow', 'funny', 'article_id', 'publication_id']
feature_object = ['rating', 'created_date']
for f_i in features_int:
    if f_i != 'article_id':
        trains[f_i] = trains[f_i].astype(np.int16)
for f_f in features:
    if f_f not in features_int and f_f not in feature_object:
        trains[f_f] = trains[f_f].astype(np.float)

# 根据特征属性划分不同的特征组
features_religion = ['hindu', 'buddhist', 'christian',
                     'muslim', 'atheist', 'other_religion'] # 宗教特征
features_ethnicity = ['white', 'black', 'asian', 'latino',
                      'jewish', 'other_race_or_ethnicity'] # 种族特征
features_sexual = ['heterosexual', 'homosexual_gay_or_lesbian',
                   'other_sexual_orientation', 'bisexual'] #  取向特征
features_gender = ['male', 'female', 'transgender', 'other_gender'] # 性别特征
features_disability = ['intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
                       'physical_disability', 'other_disability'] # 健康疾病特征
X = ['asian', 'atheist',
       'bisexual', 'black', 'buddhist', 'christian', 'female',
       'heterosexual', 'hindu', 'homosexual_gay_or_lesbian',
       'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
       'muslim', 'other_disability', 'other_gender',
       'other_race_or_ethnicity', 'other_religion',
       'other_sexual_orientation', 'physical_disability',
       'psychiatric_or_mental_illness', 'transgender', 'white']
features_comment = ['rating', 'likes', 'sad', 'wow', 'funny', 'disagree'] # 关于评论本身的属性和点赞人数
features_ids = ['article_id', 'parent_id', 'publication_id',
                'identity_annotator_count', 'toxicity_annotator_count'] # 关于受评论的内容本身
times = ['created_date'] # 对于test数据同样无用

# 简单来说这两部分特征对于test数据都是没有价值因为其产生并不会体现在评论本身里面而是受外部因素影响
features_s = features_disability + features_gender + \
             features_sexual + features_ethnicity + features_religion
print(len(X), len(features_s))

others_features = ['severe_toxicity', 'obscene', 'identity_attack', 'insult',
                   'threat', 'sexual_explicit']
def perform_preprocessing(train, test):
    contract = {
        "'SB91':'senate bill','tRump':'trump','utmterm':'utm term','FakeNews':'fake news','Gʀᴇat':'great','ʙᴏᴛtoᴍ':'bottom','washingtontimes':'washington times','garycrum':'gary crum','htmlutmterm':'html utm term','RangerMC':'car','TFWs':'tuition fee waiver','SJWs':'social justice warrior','Koncerned':'concerned','Vinis':'vinys','Yᴏᴜ':'you','Trumpsters':'trump','Trumpian':'trump','bigly':'big league','Trumpism':'trump','Yoyou':'you','Auwe':'wonder','Drumpf':'trump','utmterm':'utm term','Brexit':'british exit','utilitas':'utilities','ᴀ':'a', '😉':'wink','😂':'joy','😀':'stuck out tongue', 'theguardian':'the guardian','deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation','doctrne':'doctrine','fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers','negotiatiations': 'negotiations','dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists',Trump's": 'trump is',
        "'cause": 'because', ',cause': 'because', ';cause': 'because', "ain't": 'am not', 'ain,t': 'am not',
        'ain;t': 'am not', 'ain´t': 'am not', 'ain’t': 'am not', "aren't": 'are not',
        'aren,t': 'are not', 'aren;t': 'are not', 'aren´t': 'are not', 'aren’t': 'are not', "can't": 'cannot',
        "can't've": 'cannot have', 'can,t': 'cannot', 'can,t,ve': 'cannot have',
        'can;t': 'cannot', 'can;t;ve': 'cannot have', 'can´t': 'cannot', 'can´t´ve': 'cannot have', 'can’t': 'cannot',
        'can’t’ve': 'cannot have',
        "could've": 'could have', 'could,ve': 'could have', 'could;ve': 'could have', "couldn't": 'could not',
        "couldn't've": 'could not have', 'couldn,t': 'could not', 'couldn,t,ve': 'could not have',
        'couldn;t': 'could not',
        'couldn;t;ve': 'could not have', 'couldn´t': 'could not',
        'couldn´t´ve': 'could not have', 'couldn’t': 'could not', 'couldn’t’ve': 'could not have',
        'could´ve': 'could have',
        'could’ve': 'could have', "didn't": 'did not', 'didn,t': 'did not', 'didn;t': 'did not', 'didn´t': 'did not',
        'didn’t': 'did not', "doesn't": 'does not', 'doesn,t': 'does not', 'doesn;t': 'does not', 'doesn´t': 'does not',
        'doesn’t': 'does not', "don't": 'do not', 'don,t': 'do not', 'don;t': 'do not', 'don´t': 'do not',
        'don’t': 'do not',
        "hadn't": 'had not', "hadn't've": 'had not have', 'hadn,t': 'had not', 'hadn,t,ve': 'had not have',
        'hadn;t': 'had not',
        'hadn;t;ve': 'had not have', 'hadn´t': 'had not', 'hadn´t´ve': 'had not have', 'hadn’t': 'had not',
        'hadn’t’ve': 'had not have', "hasn't": 'has not', 'hasn,t': 'has not', 'hasn;t': 'has not', 'hasn´t': 'has not',
        'hasn’t': 'has not',
        "haven't": 'have not', 'haven,t': 'have not', 'haven;t': 'have not', 'haven´t': 'have not',
        'haven’t': 'have not', "he'd": 'he would',
        "he'd've": 'he would have', "he'll": 'he will',
        "he's": 'he is', 'he,d': 'he would', 'he,d,ve': 'he would have', 'he,ll': 'he will', 'he,s': 'he is',
        'he;d': 'he would',
        'he;d;ve': 'he would have', 'he;ll': 'he will', 'he;s': 'he is', 'he´d': 'he would', 'he´d´ve': 'he would have',
        'he´ll': 'he will',
        'he´s': 'he is', 'he’d': 'he would', 'he’d’ve': 'he would have', 'he’ll': 'he will', 'he’s': 'he is',
        "how'd": 'how did', "how'll": 'how will',
        "how's": 'how is', 'how,d': 'how did', 'how,ll': 'how will', 'how,s': 'how is', 'how;d': 'how did',
        'how;ll': 'how will',
        'how;s': 'how is', 'how´d': 'how did', 'how´ll': 'how will', 'how´s': 'how is', 'how’d': 'how did',
        'how’ll': 'how will',
        'how’s': 'how is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', 'i,d': 'i would',
        'i,ll': 'i will',
        'i,m': 'i am', 'i,ve': 'i have', 'i;d': 'i would', 'i;ll': 'i will', 'i;m': 'i am', 'i;ve': 'i have',
        "isn't": 'is not',
        'isn,t': 'is not', 'isn;t': 'is not', 'isn´t': 'is not', 'isn’t': 'is not', "it'd": 'it would',
        "it'll": 'it will', "It's": 'it is',
        "it's": 'it is', 'it,d': 'it would', 'it,ll': 'it will', 'it,s': 'it is', 'it;d': 'it would',
        'it;ll': 'it will', 'it;s': 'it is', 'it´d': 'it would', 'it´ll': 'it will', 'it´s': 'it is',
        'it’d': 'it would', 'it’ll': 'it will', 'it’s': 'it is',
        'i´d': 'i would', 'i´ll': 'i will', 'i´m': 'i am', 'i´ve': 'i have', 'i’d': 'i would', 'i’ll': 'i will',
        'i’m': 'i am',
        'i’ve': 'i have', "let's": 'let us', 'let,s': 'let us', 'let;s': 'let us', 'let´s': 'let us',
        'let’s': 'let us', "ma'am": 'madam', 'ma,am': 'madam', 'ma;am': 'madam', "mayn't": 'may not',
        'mayn,t': 'may not', 'mayn;t': 'may not',
        'mayn´t': 'may not', 'mayn’t': 'may not', 'ma´am': 'madam', 'ma’am': 'madam', "might've": 'might have',
        'might,ve': 'might have', 'might;ve': 'might have', "mightn't": 'might not', 'mightn,t': 'might not',
        'mightn;t': 'might not', 'mightn´t': 'might not',
        'mightn’t': 'might not', 'might´ve': 'might have', 'might’ve': 'might have', "must've": 'must have',
        'must,ve': 'must have', 'must;ve': 'must have',
        "mustn't": 'must not', 'mustn,t': 'must not', 'mustn;t': 'must not', 'mustn´t': 'must not',
        'mustn’t': 'must not', 'must´ve': 'must have',
        'must’ve': 'must have', "needn't": 'need not', 'needn,t': 'need not', 'needn;t': 'need not',
        'needn´t': 'need not', 'needn’t': 'need not', "oughtn't": 'ought not', 'oughtn,t': 'ought not',
        'oughtn;t': 'ought not',
        'oughtn´t': 'ought not', 'oughtn’t': 'ought not', "sha'n't": 'shall not', 'sha,n,t': 'shall not',
        'sha;n;t': 'shall not', "shan't": 'shall not',
        'shan,t': 'shall not', 'shan;t': 'shall not', 'shan´t': 'shall not', 'shan’t': 'shall not',
        'sha´n´t': 'shall not', 'sha’n’t': 'shall not',
        "she'd": 'she would', "she'll": 'she will', "she's": 'she is', 'she,d': 'she would', 'she,ll': 'she will',
        'she,s': 'she is', 'she;d': 'she would', 'she;ll': 'she will', 'she;s': 'she is', 'she´d': 'she would',
        'she´ll': 'she will',
        'she´s': 'she is', 'she’d': 'she would', 'she’ll': 'she will', 'she’s': 'she is', "should've": 'should have',
        'should,ve': 'should have', 'should;ve': 'should have',
        "shouldn't": 'should not', 'shouldn,t': 'should not', 'shouldn;t': 'should not', 'shouldn´t': 'should not',
        'shouldn’t': 'should not', 'should´ve': 'should have',
        'should’ve': 'should have', "that'd": 'that would', "that's": 'that is', 'that,d': 'that would',
        'that,s': 'that is', 'that;d': 'that would',
        'that;s': 'that is', 'that´d': 'that would', 'that´s': 'that is', 'that’d': 'that would', 'that’s': 'that is',
        "there'd": 'there had',
        "there's": 'there is', 'there,d': 'there had', 'there,s': 'there is', 'there;d': 'there had',
        'there;s': 'there is',
        'there´d': 'there had', 'there´s': 'there is', 'there’d': 'there had', 'there’s': 'there is',
        "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have',
        'they,d': 'they would', 'they,ll': 'they will', 'they,re': 'they are', 'they,ve': 'they have',
        'they;d': 'they would', 'they;ll': 'they will', 'they;re': 'they are',
        'they;ve': 'they have', 'they´d': 'they would', 'they´ll': 'they will', 'they´re': 'they are',
        'they´ve': 'they have', 'they’d': 'they would', 'they’ll': 'they will',
        'they’re': 'they are', 'they’ve': 'they have', "wasn't": 'was not', 'wasn,t': 'was not', 'wasn;t': 'was not',
        'wasn´t': 'was not',
        'wasn’t': 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have',
        'we,d': 'we would', 'we,ll': 'we will',
        'we,re': 'we are', 'we,ve': 'we have', 'we;d': 'we would', 'we;ll': 'we will', 'we;re': 'we are',
        'we;ve': 'we have',
        "weren't": 'were not', 'weren,t': 'were not', 'weren;t': 'were not', 'weren´t': 'were not',
        'weren’t': 'were not', 'we´d': 'we would', 'we´ll': 'we will',
        'we´re': 'we are', 'we´ve': 'we have', 'we’d': 'we would', 'we’ll': 'we will', 'we’re': 'we are',
        'we’ve': 'we have', "what'll": 'what will', "what're": 'what are', "what's": 'what is',
        "what've": 'what have', 'what,ll': 'what will', 'what,re': 'what are', 'what,s': 'what is',
        'what,ve': 'what have', 'what;ll': 'what will', 'what;re': 'what are',
        'what;s': 'what is', 'what;ve': 'what have', 'what´ll': 'what will',
        'what´re': 'what are', 'what´s': 'what is', 'what´ve': 'what have', 'what’ll': 'what will',
        'what’re': 'what are', 'what’s': 'what is',
        'what’ve': 'what have', "where'd": 'where did', "where's": 'where is', 'where,d': 'where did',
        'where,s': 'where is', 'where;d': 'where did',
        'where;s': 'where is', 'where´d': 'where did', 'where´s': 'where is', 'where’d': 'where did',
        'where’s': 'where is',
        "who'll": 'who will', "who's": 'who is', 'who,ll': 'who will', 'who,s': 'who is', 'who;ll': 'who will',
        'who;s': 'who is',
        'who´ll': 'who will', 'who´s': 'who is', 'who’ll': 'who will', 'who’s': 'who is', "won't": 'will not',
        'won,t': 'will not', 'won;t': 'will not',
        'won´t': 'will not', 'won’t': 'will not', "wouldn't": 'would not', 'wouldn,t': 'would not',
        'wouldn;t': 'would not', 'wouldn´t': 'would not',
        'wouldn’t': 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', 'you,d': 'you would',
        'you,ll': 'you will',
        'you,re': 'you are', 'you;d': 'you would', 'you;ll': 'you will',
        'you;re': 'you are', 'you´d': 'you would', 'you´ll': 'you will', 'you´re': 'you are', 'you’d': 'you would',
        'you’ll': 'you will', 'you’re': 'you are',
        '´cause': 'because', '’cause': 'because', "you've": "you have", "could'nt": 'could not',
        "havn't": 'have not', "here’s": "here is", 'i""m': 'i am', "i'am": 'i am', "i'l": "i will", "i'v": 'i have',
        "wan't": 'want', "was'nt": "was not", "who'd": "who would",
        "who're": "who are", "who've": "who have", "why'd": "why would", "would've": "would have", "y'all": "you all",
        "y'know": "you know", "you.i": "you i",
        "your'e": "you are", "arn't": "are not", "agains't": "against", "c'mon": "common", "doens't": "does not",
        'don""t': "do not", "dosen't": "does not",
        "dosn't": "does not", "shoudn't": "should not", "that'll": "that will", "there'll": "there will",
        "there're": "there are",
        "this'll": "this all", "u're": "you are", "ya'll": "you all", "you'r": "you are", "you’ve": "you have",
        "d'int": "did not", "did'nt": "did not", "din't": "did not", "dont't": "do not", "gov't": "government",
        "i'ma": "i am", "is'nt": "is not", "‘I": 'I',
        'ᴀɴᴅ': 'and', 'ᴛʜᴇ': 'the', 'ʜᴏᴍᴇ': 'home', 'ᴜᴘ': 'up', 'ʙʏ': 'by', 'ᴀᴛ': 'at', '…and': 'and',
        'civilbeat': 'civil beat', \
        'TrumpCare': 'Trump care', 'Trumpcare': 'Trump care', 'OBAMAcare': 'Obama care', 'ᴄʜᴇᴄᴋ': 'check', 'ғᴏʀ': 'for',
        'ᴛʜɪs': 'this', 'ᴄᴏᴍᴘᴜᴛᴇʀ': 'computer', \
        'ᴍᴏɴᴛʜ': 'month', 'ᴡᴏʀᴋɪɴɢ': 'working', 'ᴊᴏʙ': 'job', 'ғʀᴏᴍ': 'from', 'Sᴛᴀʀᴛ': 'start', 'gubmit': 'submit',
        'CO₂': 'carbon dioxide', 'ғɪʀsᴛ': 'first', \
        'ᴇɴᴅ': 'end', 'ᴄᴀɴ': 'can', 'ʜᴀᴠᴇ': 'have', 'ᴛᴏ': 'to', 'ʟɪɴᴋ': 'link', 'ᴏғ': 'of', 'ʜᴏᴜʀʟʏ': 'hourly',
        'ᴡᴇᴇᴋ': 'week', 'ᴇɴᴅ': 'end', 'ᴇxᴛʀᴀ': 'extra', \
        'Gʀᴇᴀᴛ': 'great', 'sᴛᴜᴅᴇɴᴛs': 'student', 'sᴛᴀʏ': 'stay', 'ᴍᴏᴍs': 'mother', 'ᴏʀ': 'or', 'ᴀɴʏᴏɴᴇ': 'anyone',
        'ɴᴇᴇᴅɪɴɢ': 'needing', 'ᴀɴ': 'an', 'ɪɴᴄᴏᴍᴇ': 'income', \
        'ʀᴇʟɪᴀʙʟᴇ': 'reliable', 'ғɪʀsᴛ': 'first', 'ʏᴏᴜʀ': 'your', 'sɪɢɴɪɴɢ': 'signing', 'ʙᴏᴛᴛᴏᴍ': 'bottom',
        'ғᴏʟʟᴏᴡɪɴɢ': 'following', 'Mᴀᴋᴇ': 'make', \
        'ᴄᴏɴɴᴇᴄᴛɪᴏɴ': 'connection', 'ɪɴᴛᴇʀɴᴇᴛ': 'internet', 'financialpost': 'financial post', 'ʜaᴠᴇ': ' have ',
        'ᴄaɴ': ' can ', 'Maᴋᴇ': ' make ', 'ʀᴇʟɪaʙʟᴇ': ' reliable ', 'ɴᴇᴇᴅ': ' need ',
        'ᴏɴʟʏ': ' only ', 'ᴇxᴛʀa': ' extra ', 'aɴ': ' an ', 'aɴʏᴏɴᴇ': ' anyone ', 'sᴛaʏ': ' stay ', 'Sᴛaʀᴛ': ' start',
        'SHOPO': 'shop',
        }
    punct = [',', '.', '"', ':', '🐢', ')', '🐴', '🐵', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^','\n'
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
    def clean_special_chars(text, punct, mapping):
        for p in mapping:
            text = text.replace(p, mapping[p].lower())
        for p in punct:
            text = text.replace(p, ' ')
        return text

    for df in [train, test]:
        df['comment_text'] = df['comment_text'].astype(str)
        df['comment_text'] = df['comment_text'].apply(lambda x: clean_special_chars(x, punct, contract))

    return train, test

trains, tests = perform_preprocessing(trains[['id', 'target', 'comment_text']], tests)
def get_new_features(datas):
    features_religion = ['hindu', 'buddhist', 'christian',
                         'muslim', 'atheist', 'other_religion']  # 宗教特征
    features_ethnicity = ['white', 'black', 'asian', 'latino',
                          'jewish', 'other_race_or_ethnicity']  # 种族特征
    features_sexual = ['heterosexual', 'homosexual_gay_or_lesbian',
                       'other_sexual_orientation', 'bisexual']  # 取向特征
    features_gender = ['male', 'female', 'transgender', 'other_gender']  # 性别特征
    features_disability = ['intellectual_or_learning_disability', 'psychiatric_or_mental_illness',
                           'physical_disability', 'other_disability']  # 健康疾病特征
    pattern = r'\s'
    datas['comment_text'] = datas['comment_text'].apply(
        lambda x: list(re.split(pattern, x))
    ).astype(str)
    for i in features_religion:
        datas[i+'_count_text'] = datas['comment_text'].apply(lambda x: x.lower().count(i))
    ethnicity_word = ['latinos', 'hispanic', 'hispanics', 'negger', 'negro', 'mongoloid', 'caucasian', 'jewess', 'jew']
    features_ethnicity = features_ethnicity + ethnicity_word
    for i in features_ethnicity:
        datas[i+'_count_text'] = datas['comment_text'].apply(lambda x: x.lower().count(i))
    new_features_sexual = ['gay', 'bisexual', 'heterosexual',
                           'homosexual', 'lesbian']
    # 关于这一部分的言论，说实话很难去定义是否有恶，之后可能需要定义权重给这些特征
    for i in new_features_sexual:
        datas[i+'_count_text'] = datas['comment_text'].apply(lambda x: x.lower().count(i))
    new_features_gender_word = ['sexism', 'ageism', 'sexist']
    features_gender = features_gender + new_features_gender_word
    for i in features_gender:
        datas[i+'_count_text'] = datas['comment_text'].apply(lambda x: x.lower().count(i))
    # 同样关于这部分的特征除了明确的单词外类似于male,fmeal都是十分中性的词，同样需要特殊处理
    new_features_disability = ['mentally', 'ill', 'physical',
                               'disability', 'mental', 'disable',
                               'psychiatric']
    for i in new_features_disability:
        datas[i+'_count_text'] = datas['comment_text'].apply(lambda x: x.lower().count(i))
    datas['new_disability_count_text'] = 0
    for i in new_features_disability:
        datas['new_disability_count_text'] += datas[i+'_count_text']
        datas.drop(i+'_count_text', axis=1, inplace=True)
    datas['new_disability_count_text'] = datas['new_disability_count_text'].apply(lambda x: 1 if x>3 else 0)
    datas.drop('comment_text', axis=1, inplace=True)
    datas['special_value_sum'] = datas.apply(lambda x: x.sum(), axis=1)
    datas['special_value_std'] = datas.apply(lambda x: x.std(), axis=1)
    datas['special_value_mean'] = datas['special_value_sum'] / len(datas.columns)
    return datas

other_trains = get_new_features(trains[['comment_text']])
other_tests = get_new_features(tests[['comment_text']])



# 使用tf-Idf处理失败
"""
trains_poison = trains.loc[trains.target > 0.5]
trains_nopoison = trains.loc[trains.target == 0]
others_features_ids = []
others_features_ids_n = []
for i in others_features:
    others_features_ids = others_features_ids + list(trains_poison.loc[trains_poison[i] > 0.5].id.values)
    others_features_ids_n = others_features_ids_n + list(trains_nopoison[trains_nopoison[i]==0].id.values)

others_features_comments = []
X = ['0 ', 'z']
for i in set(others_features_ids):
    #others_features_comments.append(trains_poison.loc[trains_poison.id==i].comment_text.values[0])
    X[0] = X[0] + trains_poison.loc[trains_poison.id==i].comment_text.values[0]
others_features_comments_n = []
print("zxcasd")
print(X)
Y = ['0', 'z']
import random
for i in set(random.sample(others_features_ids_n, 100)):
    others_features_comments_n.append(trains_nopoison.loc[trains_nopoison.id==i].comment_text.values[0])
    Y[0] = Y[0] + trains_nopoison.loc[trains_nopoison.id==i].comment_text.values[0]
print(Y)
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
tf = TfidfTransformer()
ct = CountVectorizer()
others_features_comments = ct.fit_transform(X)
others_features_comments_tfidf = tf.fit_transform(others_features_comments)
words = ct.get_feature_names()  # 所有文本的关键字
weight = others_features_comments_tfidf.toarray()
new_others_features_words = []
sort_weight = np.argsort(weight[0])
for i in range(100):
    new_others_features_words.append(words[sort_weight[i]])
print(set(new_others_features_words))

'''
for w in weight:
    loc = np.argsort(-w)
    for i in range(1):
        new_others_features_words.append(words[loc[i]])
new_others_features_words = set(new_others_features_words)
'''

#others_features_comments_n = random.sample(others_features_comments_n, 100)
others_features_comments_n = ct.fit_transform(Y)
others_features_comments_tfidf_n = tf.fit_transform(others_features_comments_n)
words_n = ct.get_feature_names()  # 所有文本的关键字
weight_n = others_features_comments_tfidf.toarray()
new_others_features_words_n = []
sort_weight_n = np.argsort(-weight_n[0])
for i in range(100):
    new_others_features_words_n.append(words_n[sort_weight_n[i]])
print(new_others_features_words_n)
'''
for w in weight_n:
    loc = np.argsort(-w)
    for i in range(1):
        new_others_features_words_n.append(words_n[loc[i]])
new_others_features_words_n = set(new_others_features_words_n)
'''
remove_words = []
for i in new_others_features_words:
    if i in new_others_features_words_n:
        remove_words.append(i)
for i in remove_words:
    new_others_features_words.remove(i)
print(len(new_others_features_words))
print(new_others_features_words)
"""
trains['target'] = trains['target'].apply(lambda x: 1 if x>0.5 else 0)
train = trains[:25000]
label_train = train.pop('target')
other_trains_1 = other_trains[:25000]
tests = trains[25000:]
other_trains_2 = other_trains[25000:]
label_test  = tests.pop('target')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=200)
tokenizer.fit_on_texts(list(trains['comment_text'])+list(tests['comment_text']))
word_index = tokenizer.word_index
train_X = tokenizer.texts_to_sequences(train['comment_text'])
test_X = tokenizer.texts_to_sequences(tests['comment_text'])
train_X = pad_sequences(train_X, maxlen=220)
test_X = pad_sequences(test_X, maxlen=220)

train_X = np.hstack([train_X, other_trains_1])
test_X = np.hstack([test_X, other_trains_2])


from sklearn.model_selection import StratifiedKFold

params = {
    'max_depth':-1,
    'n_estimators':1000,
    'learning_rate':0.05,
    'num_leaves':2**9-1,
    'colsample_bytree':0.28,
    'objective':'binary',
    'n_jobs':-1,
    'eval_metric':'auc'
}
import lightgbm as lgb
xtrain = lgb.Dataset(train_X, label_train)
num_round = 10000
lgb = lgb.train(params, xtrain, num_round)
yp = lgb.predict(test_X)
from sklearn.metrics import roc_auc_score,f1_score

print(roc_auc_score(list(label_test.values), list(yp)))

