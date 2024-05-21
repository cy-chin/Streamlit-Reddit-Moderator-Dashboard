'''
Streamlit app - Reddit Moderator Dashboard

Note: perform the following before able to access the app from web browser
1. Update the path to load the model.pkl and count_vectorizer.pkl pickle code

'''


import streamlit as st
import pandas as pd
import numpy as np
import pickle

import string
import requests 
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def sentence_lemmatizer(text):
    if text.strip() == '':
        return np.NaN

    lemmatizer = WordNetLemmatizer()
    char_list = word_tokenize(text.lower())

    # Lemmatize list of words and join
    return ' '.join([lemmatizer.lemmatize(w.lower(), get_wordnet_pos(w.lower())) for w in char_list])

def text_cleaning(text):

    url_regex = re.compile(
    r'((http|https)://'     # Start with http:// or https://
    r'([a-zA-Z0-9.-]+)'     # Match the domain name (alphanumeric characters, dots, and dashes)
    r'(\.[a-zA-Z]{2,})'     # Match the top-level domain (e.g., .com, .net) with at least 2 characters
    r'(:\d+)?'              # Match an optional port number
    r'(/\S*)?'              # Match an optional path (any non-whitespace characters)
    r'(\?[^"\s]*)?)',        # Match an optional query string (attribute-value pairs)
    re.IGNORECASE        # Ignore case sensitivity
    )

    # remove URL
    text = url_regex.sub("", text)

    # Mark the comment with [deleted] or [removed] with pseudo marker "pseudodeleted" and "pseudoremoved"
    # After lemmatization, [deleted] become [ delete ], [removed] become [ remove ]
    # After stemming, [deleted] become [ delet ], [removed] become [ remov ]
    text = text.replace("[deleted]","pseudodeleted").replace("[removed]","pseudoremoved").replace("[ delete ]","pseudodeleted").replace("[ remove ]","pseudoremoved").replace("[ delet ]","pseudodeleted").replace("[ remov ]","pseudoremoved")

    # remove newline 
    text = text.replace("\n", " ").replace("\r", " ").replace("\r\n"," ").replace("_x000D_", " ")

    # remove  "'s"
    text = re.sub(r"(\'s)","", text)
    
    # remove stopword 
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

   # remove punctuation and special character
    text = ''.join([char for char in text if char not in special_char_list])

    # return the cleaned text
    return text.strip()

def text_cleaning_post_lemmatized(text):

    # pattern of a standard footer in a submission template
    pattern = r'pm\sexclude\sme\sexclude\sfrom\ssubreddit\sfaq\sinformation\ssource\s.*downvote\sto\sremove\sv028'
    text = re.sub(pattern, '', text)

    # pattern2 of a standard footer in a submission template
    pattern2 = r'nonmobile\slink\shelperbot\sv11\srhelperbot\si\sbe\sa\sbot\splease\smessage\suswim1929\swith\sany\sfeedback\sandor\shate\scounter\s\d{6}'
    text = re.sub(pattern, '', text)

    # pattern to find any words that are repeating more than 2 times, replace with only 1 occurence of the word
    text = re.sub(r'\b(\w+)(?: \1\b)+', r'\1', text.lower())

    text = re.sub(r'gon\sna', "gonna", text.lower())

    # return the cleaned text
    return text.strip()


def calculate_total(row):
    row['Total'] = row[[0,1]].sum()
    return row

def calculate_percentage(row):
    threshold_1 = 45.0
    threshold_2 = 50.0
    row['Misinformation%'] = 100* row[1]/row['Total']
    if row['Misinformation%'] >= threshold_2:
        row['Misinformation Alert'] = "ALERT"
    elif threshold_1 < row['Misinformation%'] < threshold_2:
        row['Misinformation Alert'] = "WARNING"
    else:
        row['Misinformation Alert'] = "N"
    return row

# trained model
with open(r"./model.pkl", 'rb') as rf_model:
    model = pickle.load(rf_model)

# fitted transformer - CountVectorizer, to transform the data before calling to model (for prediction)
with open(r"./count_vectorizer.pkl", 'rb') as rf_cv:
    cvec = pickle.load(rf_cv)


st.set_page_config(page_title='Reddit Moderator - DASHBOARD', 
                   page_icon='', 
                   layout='centered', 
                   initial_sidebar_state='collapsed',
                   menu_items= {
                       'Get Help':'http://localhost:8501',
                       'Report a bug':'http://localhost:8501',
                       'About':'http://localhost:8501'                                              
                   })
st.subheader("Welcome, Daniel!")

# Set title of the app
st.title("Reddit Moderator DASHBOARD .. What's hot")
subreddit = st.selectbox("Subreddits you are moderating. Which one to monitor? ",['politics','news',]) 
hot_num = st.slider('How many hot topic?', 0, 10, 5)
progress_text = "Gathering the comments. Please wait..."
progress_text_done = "All comments are ready for analyzing"
progress_text_2 = "Analyzing the comments. This may take some time ..."
# my_bar = st.progress(0, text=progress_text)
submission_ids = []
submission_urls = []
all_comments_list = []

button1 = st.button('Start Analyzing') 

if button1:
    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
    response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
    if response.status_code == 200:
        my_bar = st.progress(0.1, text=progress_text)
        json_data = response.json()
        submission_count = 0 
        for post in json_data['data']['children']:
            if submission_count > hot_num:
                break
            submission_ids.append(post['data']['id'])
            submission_count += 1 

        submission_urls = [ f"https://www.reddit.com/r/{subreddit}/comments/{id}.json" for id in submission_ids ]
        # st.write(submission_urls)

        def extract_comments(comment, submission_title, comments_list):
            if 'body' in comment['data']:  # Check if the comment has a body
                comments_list.append({
                    'comment id': comment['data']['id'],
                    'title': submission_title,
                    'body': comment['data']['body']
                })
        
        # Recursively extract child comments
            if 'replies' in comment['data'] and comment['data']['replies'] != '':
                replies_data = comment['data']['replies']['data']['children']
                for reply in replies_data:
                    extract_comments(reply, submission_title, comments_list)


        # Initialize an empty list to store comments


        # Loop through each submission URL
        for n, url in enumerate(submission_urls):
            # Send a GET request to retrieve the JSON data
            response = requests.get(url, headers={'User-agent': 'Mozilla/5.0'})
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON data
                progress = (1.0/hot_num)*(n) if n > 0 else 0.1
                my_bar.empty()
                my_bar = st.progress(progress, text=progress_text)
                json_data = response.json()
                
                # Extract submission information
                submission_data = json_data[0]['data']['children'][0]['data']
                submission_title = submission_data['title']
                num_comments = submission_data['num_comments']
                
                # st.write("Submission Title:", submission_title)
                # st.write("Number of Comments:", num_comments)
                
                # Extract comments and child comments recursively
                comments_data = json_data[1]['data']['children']
                for comment in comments_data:
                    extract_comments(comment, submission_title, all_comments_list)
                
            else:
                st.write("Failed to retrieve data from URL:", url)

            my_bar.empty()
            my_bar = st.progress(1.0, text=progress_text_done)

        # Create DataFrame from all comments
    comments_df = pd.DataFrame(all_comments_list)

    my_bar2 = st.progress(0.1, text=progress_text_2)

    #st.write(f"All comments from multiple submissions (including comment forests) download into dataframe successfully.")

    stop_words = set(stopwords.words('english'))
    special_char_list = list(string.punctuation)
    special_char_list+=["’","'s","’s","...","$","@$$.","like", "it", "would", "im","“", "”", "u"]



    comments_df['comments_cleaned'] = comments_df['body'].map(text_cleaning)
    my_bar2.empty()
    my_bar2 = st.progress(0.4, text=progress_text_2)
    comments_df['comments_cleaned_lemmatized'] = comments_df['comments_cleaned'].map(sentence_lemmatizer)
    my_bar2.empty()
    my_bar2 = st.progress(0.7, text=progress_text_2)
    comments_df = comments_df.dropna(subset=['comments_cleaned_lemmatized'])
    my_bar2.empty()
    my_bar2 = st.progress(0.9, text=progress_text_2)
    comments_df['comments_cleaned_lemmatized'].isnull().sum() 
    comments_df.reset_index(drop=True, inplace=True)

    comments_df['comments_cleaned_lemmatized_cleaned'] = comments_df['comments_cleaned_lemmatized'].map(text_cleaning_post_lemmatized)
    my_bar2.empty()
    my_bar2 = st.progress(1.0, text=progress_text_2)


    X = cvec.transform(comments_df['comments_cleaned_lemmatized_cleaned'])

    y_predict = model.predict(X)

    comments_df['label'] = pd.DataFrame(y_predict)

    # groupby 'Pclass' and 'Survived'
    t = comments_df.groupby(['title','label']).size()

    # unstack 'Survived' to the columns
    dt = t.unstack(['label'])
 
    # first apply the calculate_total function to calculate the total of each row, and create a new column 'Total',for each pclass
    dt2 = dt.apply(calculate_total, axis=1)

    # then, apply the calculate_percentage function to calculate the percentage for each pclass
    dt3 = dt2.apply(calculate_percentage, axis=1)

    st.write(dt3)
    st.success("Please pay attention to the submissions with ALERT")