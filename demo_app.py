import base64
import pickle
import re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import requests

# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from annotated_text import annotated_text
from keybert import KeyBERT
from load_css import local_css

# from nltk.corpus import stopwords
from PIL import Image
from streamlit_extras.colored_header import colored_header
from streamlit_extras.tags import tagger_component
from wordcloud import WordCloud

# Initial page config
st.set_page_config(
    page_title='Topic Modeling Demo',
    layout='wide',
    initial_sidebar_state='expanded',
)

# nlp = spacy.load('en_core_web_sm')
# stop_words = spacy.lang.en.stop_words.STOP_WORDS
# stop_words = set(stopwords.words('english'))
with open('./DATA/stopwords.pkl', 'rb') as f:
    stop_words = pickle.load(f)


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    local_css('./styles.css')
    df_source, matching_topic, model_get_topics, model_topic_df = setup_data()

    input_option = cs_sidebar()
    cs_body(df_source, model_topic_df, model_get_topics, matching_topic, input_option=input_option)


def img_to_bytes(img_path: str) -> str:
    """
    Convert an image file to base64 encoded bytes.

    :param img_path: The path to the image file
    :type img_path: str
    :return: The base64 encoded bytes of the image
    :rtype: str
    """
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def create_wordcloud(model_get_topics, topic_idx: int) -> WordCloud:
    """
    Create a word cloud from a given topic using a model.

    :param model_get_topics: The dataframe contains the topics and their scores from BERTopic model
    :type model_get_topics: pd.DataFrame
    :param topic_idx: The topic index to generate the word cloud
    :type topic_idx: int
    :return: The generated word cloud
    :rtype: WordCloud
    """
    topic_idx = topic_idx
    filtered_df = model_get_topics[model_get_topics['topic_idx'] == topic_idx]
    text = dict(zip(filtered_df['topic'], filtered_df['score']))

    wc = WordCloud(width=400, height=200, background_color='#000814', colormap='Blues')
    wc.generate_from_frequencies(text)
    return wc


def get_top_important_words(post: str, top_n: int = 15) -> Tuple[List[str], List[float]]:
    """
    Extract the top important words from a post using the KeyBERT model.

    :param post: The post to extract keywords from
    :type post: str
    :param top_n: The number of top keywords to extract, defaults to 15
    :type top_n: int, optional
    :return: A tuple containing a list of the top words and their corresponding scores
    :rtype: Tuple[List[str], List[float]]
    """

    kw_model = KeyBERT()
    keywords_with_scores = kw_model.extract_keywords(
        post, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n
    )

    top_words = [kw[0] for kw in keywords_with_scores]
    top_scores = [kw[1] for kw in keywords_with_scores]

    return top_words, top_scores


def plot_wordcloud_with_keybert_scores(
    top_words: List[str], top_scores: List[float], background_color: str = '#000814'
) -> WordCloud:
    """
    Plot a word cloud with KeyBert scores.

    :param top_words: The top words to include in the word cloud
    :type top_words: List[str]
    :param top_scores: The KeyBert scores corresponding to the top words
    :type top_scores: List[float]
    :param background_color: The background color for the word cloud, defaults to '#000814'
    :type background_color: str, optional
    :return: The generated word cloud
    :rtype: WordCloud
    """
    word_scores = {word: score for word, score in zip(top_words, top_scores)}
    wordcloud = WordCloud(
        width=400, height=200, background_color=background_color, colormap='Blues'
    ).generate_from_frequencies(word_scores)

    return wordcloud


def annotate_user_input_with_scores(
    user_input_default: str, top_words: List[str], top_scores: List[float]
) -> List[Union[str, Tuple[str, str]]]:
    """
    Annotate user input with scores.

    :param user_input_default: The user input to annotate
    :type user_input_default: str
    :param top_words: The top words to consider for annotation
    :type top_words: List[str]
    :param top_scores: The scores corresponding to the top words
    :type top_scores: List[float]
    :return: The annotated post as a list of words and scores
    :rtype: List[Union[str, Tuple[str, str]]]
    """
    annotated_post = []
    lst_of_words = re.findall(r'\w+|[^\w\s]', user_input_default)
    for word in lst_of_words:
        word_score_pair = None
        for top_word, score in zip(top_words[:5], top_scores[:5]):
            if top_word.lower() == word.lower():
                word_score_pair = (word + ' ', str(round(score, 2)))
                break
        if word_score_pair:
            annotated_post.append(word_score_pair)
        else:
            annotated_post.append(word + ' ')
    return annotated_post


def plot_bar_chart(top_words: List[str], top_scores: List[float]) -> None:
    """
    Plot a bar chart of top words and their corresponding scores.

    :param top_words: The top words to plot
    :type top_words: List[str]
    :param top_scores: The scores corresponding to the top words
    :type top_scores: List[float]
    """
    fig = go.Figure(data=[go.Bar(x=top_words, y=top_scores)])
    fig.update_layout(
        xaxis_title='Topics',
        yaxis_title='Scores',
        yaxis=dict(tickformat='.2f'),
        xaxis_tickangle=0,
        coloraxis=dict(colorscale='blues'),
    )

    st.plotly_chart(fig, use_container_width=True)


# Function to make API request for prediction
@st.cache_data(show_spinner=False)
def predict(text):
    try:
        input_data = {'post': text}
        API_URL = 'https://bertopic-demo-api-vc4itkehka-uc.a.run.app/predict'

        # Send POST request to the API endpoint
        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            prediction = response.json()
            return prediction
        else:
            st.write('Error:', response.text)
            return None
    except Exception as e:
        st.write('Error:', str(e))
        return None


@st.cache_data(show_spinner=False)
def setup_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Set up the data by loading the necessary dataframes.

    :return: A tuple containing the source dataframe, the 2 dataframes extracted from BERTopic model, and the matching topic dataframe
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """

    df_source = pd.read_pickle('./DATA/cleaned_data_5Apr.pkl')
    matching_topic = pd.read_pickle('./DATA/matching_topic.pkl')
    model_get_topics = pd.read_pickle('./DATA/model_get_topics.pkl')
    model_topic_df = pd.read_pickle('./DATA/model_get_topic_info.pkl')

    return df_source, matching_topic, model_get_topics, model_topic_df


def cs_sidebar() -> str:
    """
    Create a sidebar for the Streamlit app.

    :return: The selected input option from the sidebar
    :rtype: str
    """
    st.sidebar.markdown(
        """[<img src='data:image/png;base64,{}' class='img-fluid' width=280 height=280>](https://streamlit.io/)""".format(
            img_to_bytes('./asset/logo_krom.png')
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
                        <small>This demo showcases the utilization of Bertopic for topic modeling on social media posts.<br>
                        - Paper : [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794)  
                        <br>
                        - Document :  
                            - [Quick Start](https://maartengr.github.io/BERTopic/index.html)  
                            - [Best Practices](https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html)
                        </small>
                        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("""<hr>""", unsafe_allow_html=True)
    input_option = st.sidebar.radio(
        'To get start, select input option:', ['Manual Input', 'Choose from List']
    )

    st.sidebar.markdown("""<hr>""", unsafe_allow_html=True)
    st.sidebar.markdown("""<small> KromLab | Apr 2024 </small>""", unsafe_allow_html=True)
    return input_option


def cs_body(
    df_source: pd.DataFrame,
    model_topic_df: pd.DataFrame,
    model_get_topics: pd.DataFrame,
    matching_topic: pd.DataFrame,
    input_option: str = 'Manual Input',
) -> None:
    """
    Create the body of the Streamlit app based on the selected input option.

    :param df_source: The source dataframe
    :type df_source: pd.DataFrame
    :param model_topic_df: The model.get_topic_info() dataframe
    :type model_topic_df: pd.DataFrame
    :param model_get_topics: The model.get_topics(topic_idx) dataframe
    :type model_get_topics: pd.DataFrame
    :param matching_topic: The matching topic dataframe
    :type matching_topic: pd.DataFrame
    :param input_option: The selected input option, defaults to 'Manual Input'
    :type input_option: str, optional
    """
    st.write(
        """
        <p style="text-align: left; font-size: 45px;">
            <span class="border-highlight"><strong>
            <span class="title-word title-word-1">Topic</span>
            <span class="title-word title-word-2">Modeling</span>
            <span class="title-word title-word-3">From</span>
            <span class="title-word title-word-4">Posts -</span>
            <span class="title-word title-word-5">Demo</span>
            </strong></span>
        </p> 
        """,
        unsafe_allow_html=True,
    )

    if input_option == 'Manual Input':
        default_text = 'A shelf without cream cheese is a holiday without cheesecake. So if you can’t make your favorite dessert this year, buy any other one. And we’ll pay for it. #SpreadTheFeeling'
        user_input = st.text_area('Enter Post:', default_text)

    else:
        selected_post = st.selectbox('Select Post:', df_source['body'])

        selected_row = df_source[df_source['body'] == selected_post].iloc[0]
        user_input = selected_row['clean_body']

        col1, col2 = st.columns([1, 2])
        with col1:
            media_url = selected_row['media']
            if media_url:
                try:
                    response = requests.get(media_url[0])
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption='Post Media', use_column_width=True, width=400)
                except Exception:
                    # Handle the exception
                    # st.error(f"Error: {e}")
                    st.image(
                        'https://storage.googleapis.com/nextspaceflight/media/no_photo_availableSB.jpg',
                        caption='No media available',
                        use_column_width=True,
                        width=400,
                    )

        with col2:
            colored_header(
                label='Post Information',
                description='',
                color_name='blue-70',
            )
            st.write(selected_row['body'])
            if selected_row['tags']:
                tagger_component('Hashtags:', selected_row['tags'])
            post_info_df = pd.DataFrame(
                {
                    'Link': [selected_row['link']],
                    'Published Date': [selected_row['published_at']],
                    'Username': [selected_row['username']],
                }
            )
            st.dataframe(post_info_df, use_container_width=True, hide_index=True)

    if st.button('Generate Topics'):
        response = predict(user_input)
        topic_index = response['topic_idx']

        new_data_df = pd.DataFrame(columns=['Post', 'Topic', 'Representation'])
        for d, t in zip([user_input], [topic_index]):
            topic = model_topic_df[model_topic_df['Topic'] == t]['Name'].item()
            rep = model_topic_df[model_topic_df['Topic'] == t]['Representation']
            df = pd.DataFrame({'Post': d, 'Topic': topic, 'Representation': rep})
            new_data_df = pd.concat([new_data_df, df], ignore_index=True)

        st.divider()
        st.subheader('Generated Topics')

        model_topic = new_data_df['Topic'].values[0]
        xomad_topic = matching_topic[matching_topic['Name'] == model_topic]['xomad_topic'].values[0]
        topic_index = matching_topic[matching_topic['Name'] == model_topic]['model_topic'].values[0]

        #####################################
        # Show the topic result - Start here #
        #####################################
        with st.container():
            col1, col2 = st.columns(2)
            col1.subheader('Model Topic')
            col1.markdown('**' + model_topic + '**')
            col2.subheader('Xomad Topic')
            col2.markdown('**' + xomad_topic + '**')

        st.divider()
        #####################################
        # Show the topic result - End here #
        #####################################

        with st.container():
            #####################################
            # Show the c-TFIDF - Start here #
            #####################################
            with st.container():
                st.subheader(f'c-TF-IDF Score for {model_topic}')
                st.write("""
                        c-TF-IDF score measures how uniquely important a word is within a specific cluster of posts compared to its importance across all clusters. 
                        - High c-TF-IDF Score: If a word has a high c-TF-IDF score, it means that word is very special for a particular type of cluster. It's like a word that you mostly find in one type of cluster and not in others.
                        - Low c-TF-IDF Score: If a word has a low c-TF-IDF score, it means that word is not very special for any particular type of cluster. It's like a word that you find almost everywhere, in many different types of clusters.
                        """)
                filtered_df = model_get_topics[model_get_topics['topic_idx'] == topic_index]
                words = [word + '  ' for word in filtered_df['topic']][:5]
                scores = [round(score, 2) for score in filtered_df['score']][:5]
                st.markdown(
                    "<p style='color: #ffc300;'><b>Top Topics and Their Scores</b></p>",
                    unsafe_allow_html=True,
                )
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label=words[0], value=scores[0], delta=None)
                col1.progress(100)
                col2.metric(label=words[1], value=scores[1], delta=None)
                col2.progress(100)
                col3.metric(label=words[2], value=scores[2], delta=None)
                col3.progress(100)
                col4.metric(label=words[3], value=scores[3], delta=None)
                col4.progress(100)
                col5.metric(label=words[4], value=scores[4], delta=None)
                col5.progress(100)
                plot_bar_chart(
                    [word + '  ' for word in filtered_df['topic']][:10],
                    [round(score, 2) for score in filtered_df['score']][:10],
                )
            #####################################
            # Show the c-TFIDF - End here #
            #####################################
            st.divider()
            #####################################
            # Show the KeyBert - Start here #
            #####################################
            with st.container():
                st.subheader('KeyBERT Score for Post')

                st.write("""
                        KeyBERT is a keyword extraction technique that leverages BERT embeddings to create keywords and keyphrases that are most similar to a post.
                        - The most similar words could then be identified as the words that best describe the entire post.
                        """)
                top_words, top_scores = get_top_important_words(user_input, top_n=15)
                st.markdown(
                    "<p style='color: #ffc300;'><b>Top Topics and Their Scores</b></p>",
                    unsafe_allow_html=True,
                )
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric(label=top_words[0], value=round(top_scores[0], 2), delta=None)
                col1.progress(100)
                col2.metric(label=top_words[1], value=round(top_scores[1], 2), delta=None)
                col2.progress(100)
                col3.metric(label=top_words[2], value=round(top_scores[2], 2), delta=None)
                col3.progress(100)
                col4.metric(label=top_words[3], value=round(top_scores[3], 2), delta=None)
                col4.progress(100)
                col5.metric(label=top_words[4], value=round(top_scores[4], 2), delta=None)
                col5.progress(100)

                st.markdown('<u><b>Post</b></u>:', unsafe_allow_html=True)

                user_input_default = (
                    selected_row['body'] if input_option != 'Manual Input' else user_input
                )
                annotated_post = annotate_user_input_with_scores(
                    user_input_default, top_words, top_scores
                )
                annotated_text(*annotated_post)
                plot_bar_chart(top_words[:10], top_scores[:10])
            #####################################
            # Show the KeyBert - End here #
            #####################################
        st.divider()

        #####################################
        # Plot the word cloud - Start here #
        #####################################
        with st.container():
            col3, col4 = st.columns(2)

            with col3:
                col3.subheader('Model Topic Representation')
                if topic_index:
                    wc_model = create_wordcloud(model_get_topics, topic_index)
                    fig, ax = plt.subplots()
                    ax.imshow(wc_model, interpolation='bilinear')
                    ax.axis('off')
                    fig.patch.set_facecolor('#000814')
                    st.pyplot(fig)

            with col4:
                col4.subheader('Posts Representation')
                wc_post = plot_wordcloud_with_keybert_scores(top_words, top_scores)
                fig, ax = plt.subplots()
                ax.imshow(wc_post, interpolation='bilinear')
                ax.axis('off')
                fig.patch.set_facecolor('#000814')
                st.pyplot(fig)
        #####################################
        # Plot the word cloud - End here #
        #####################################


if __name__ == '__main__':
    main()
