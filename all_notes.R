library(tidyverse)
library(tidytext)

theme_set(theme_bw())

data_full <- read_csv('./data/CORONA_TWEETS.csv')

data <- data_full |>
  select(doc_id, text) |>
  filter(doc_id < 50)

#################### Pre-processing ############################################

# count characters, words, sentences, lexical diversity in each doc
data <- data |>
  mutate(text = str_squish(text), # removes leading/trailing whitespace and single whitespace all internal
         lower_text = str_to_lower(text),
         nchar = nchar(text),
         word_count = str_count(text, "\\S+"), # regex is for non-whitespace substrings
         eos_count = str_count(text, "[.!?][^.!?]"), # counts sentence ends (although will also pick up abbrevs)
         lexical_diversity = sapply(str_split(lower_text, "\\s+"), 
                                    function(words) { length(unique(words)) / length(words) })
         )


# data with stop words removed and put back together
expanded_stop_words <- tibble(word = c("coronavirus", "covid"),
                            lexicon = "custom") |>
  bind_rows(stop_words)

data <- data |>
  select(doc_id, text) |>
  unnest_tokens(output = word, input= text, strip_punct=T)|>
  anti_join(expanded_stop_words) |>
  group_by(doc_id) |>
  mutate(nostop = str_flatten(word, collapse = " ")) |>
  slice_head(n=1) |>
  ungroup() |>
  select(-word) |>
  left_join(data, by = "doc_id") |>
  select(doc_id, text, lower_text, nostop, nchar, word_count,
         eos_count, lexical_diversity)

# a different way to count words
word_count2 <- data %>% 
  unnest_tokens(output = word, input= text) |>
  count(word, sort = TRUE) |>
  top_n(15)

# tokenised, words
tokenised_words <- data |>
  unnest_tokens(output = word, input= text) |>
  select(doc_id, word)

# tokenised ngrams (stop words before)
tokenised_ngrams1 <- data |>
  select(nostop) |>
  unnest_tokens(output = ngram, input = nostop, token = "ngrams", n = 2)

# tokenised ngrams (stop words after)
tokenised_ngrams2 <- data |>
  select(text) |>
  unnest_tokens(output = ngram, input = text, token = "ngrams", n = 2) |>
  separate(ngram, into = c("first","second"),
           sep = " ", remove = FALSE) |>
  anti_join(stop_words,
            by = c("first" = "word")) |>
  anti_join(stop_words,
            by = c("second" = "word"))

# tf-idf calculation

tf_idf <- tokenised_words |>
  count(doc_id, word, sort=T) |>
  bind_tf_idf(word, doc_id, n) |>
  arrange(doc_id, desc(tf_idf))


#####################Simple Frequency Charting ##################################

### simple frequency bar chart
freq_chart <- tokenised_words |>
  #anti_join(stop_words) |>
  group_by(doc_id) |>
  count(word, sort = TRUE) |>
  top_n(10) |>
  ungroup() |>
  ggplot(aes(reorder(word, n), n, fill = as_factor(doc_id))) +
  geom_bar(stat = "identity") +
  facet_wrap(~doc_id, scales = "free_y") +
  labs(x = NULL, y = "Word Frequency by Doc ID") +
  scale_fill_discrete(name = "Doc ID") +
  coord_flip()

freq_chart

### tf-idf bar chart
tfidf_chart <- tf_idf |>
  filter(doc_id < 4) |>
  anti_join(stop_words) |>
  ggplot(aes(reorder(word, tf_idf), tf_idf, fill = as_factor(doc_id))) +
  geom_bar(stat = "identity") +
  facet_wrap(~doc_id, scales = "free_y") +
  labs(x = NULL, y = "Word Frequency by TF_IDF") +
  scale_fill_discrete(name = "Doc ID") +
  coord_flip()

tfidf_chart

### word cloud 2
library(wordcloud2)

cloud_words <- tokenised_words |>
  anti_join(stop_words) |>
  count(word, sort=TRUE) |>
  filter(n > 1)

set.seed(1234) # word cloud the same every time

# expects a dataframe with words or ngrams in one column and counts in another
wordcloud2(
  data = cloud_words,
  shape = 'circle'
)

### network graph
library(ggraph)
library(igraph)

set.seed(2020)

graph_words2 <- tokenised_ngrams2 |>
  count(first, second, sort = TRUE) |>
  filter(n > 1) |>
  graph_from_data_frame()

graph_words2

arrows_info <- grid::arrow(type = "closed", length = unit(.2, "cm"))

#Second Graph option
ggraph(graph_words2, layout = "fr") + 
  geom_edge_link(aes(edge_alpha = n), #vary the edge color by n
                 arrow = arrows_info, #add the arrow information
  ) +
  geom_node_point(size = 2,color = "lightblue") + #makes the geom slightly larger
  geom_node_text(aes(label = name), #adds text labels for the nodes
                 repel = TRUE)


############ Stemming and Lemmatisation ########################################
library(tidyverse)
library(textstem)

data <- data |>
  mutate(stems = stem_strings(nostop),
         lemmas = lemmatize_strings(nostop))

# to stem a vector of individual words
stemmed2 <- tokenised_words |>
  anti_join(stop_words) |>
  mutate(stem = stem_words(word))

# to lemmatise a vector of individual words
lemma2 <- tokenised_words |>
  anti_join(stop_words) |>
  mutate(lemma = lemmatize_words(word))

########### POS Tagging ########################################################
library(udpipe)

model_eng_ewt   <- udpipe_download_model(language = "english-ewt")
model_eng_ewt_path <- model_eng_ewt$file_model
model_eng_ewt_loaded <- udpipe_load_model(file = model_eng_ewt_path)

# Create tibble that has every word and punc mark annotated - also shows, sentences, paragraphs, lemmas, words etc
text_annotated <- udpipe_annotate(model_eng_ewt_loaded, x = data$text) |>
  as_tibble() |>
  mutate(lower_token = str_to_lower(token))

# Frequency of different parts of speech - xpos are lang specific tags
xpos_freq <- txt_freq(text_annotated$xpos) |>
  rename(xpos_tag = key)

# Frequency of different parts of speech - upos are universal tags
upos_freq <- txt_freq(text_annotated$upos) |>
  rename(upos_tag = key)

# Most occurring adjectives
adj_chart <- text_annotated |>
  filter(upos == "ADJ") |>
  count(lower_token, sort = T) |>
  filter(n > 1) |>
  ggplot(aes(x = reorder(lower_token, n), y = n, fill = n)) +
  geom_bar(stat = "identity") +
  labs(x = "Adjective", y = "Frequency") +
  coord_flip() +
  scale_fill_gradient(trans = "reverse")

adj_chart

# All types chart
pos_tags_chart <- text_annotated |>
  ggplot(aes(x = fct_rev(fct_infreq(upos)))) +
  geom_bar(aes(fill = ..count..)) +
  coord_flip() +
  scale_fill_gradient(trans = "reverse") +
  labs(x = "POS", y = "Frequency")

pos_tags_chart

############### Chapter 7: Document Term Matrix #########################
### TM method
library(tm)

# bow
data_vector <- data$text
corpus <- VCorpus(VectorSource(data_vector)) # character vector must be converted to corpus
corpus <- tm_map(corpus, removeWords, stopwords("en")) # these transformations could be done beforehand
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation) 
dtm_tm <- DocumentTermMatrix(corpus) # normal bag of words
tm::inspect(dtm_tm) # difficult to view the whole thing?

# with tf-idf scores instead
dtm_tm_tfidf <- weightTfIdf(dtm_tm)
tm::inspect(dtm_tm_tfidf)

### Tidyverse method
bow_df <- tokenised_words |>
  count(doc_id, word, sort=T)

dtm_tidy_bow <- bow_df |> cast_dtm(document = doc_id, term = word, value = n)
tm::inspect(dtm_tidy_bow)

tfidf_df <- bow_df
  bind_tf_idf(word, doc_id, n)

dtm_tidy_tfidf <- tf_idf |> cast_dtm(document = doc_id, term = word, value = tf_idf)
tm::inspect(dtm_tidy_tfidf)

########### Chapter 7: Word embeddings with Glove ############
library(data.table) # more efficient than tibble

# download glove pre-trained model - can take a while
options(timeout = 1200)
filename_url <- "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip"
filename_local <- "./data/glove.2024.wikigiga.50d.zip"
if (!file.exists(filename_local)){
  download.file(filename_url, filename_local) 
  unzip(filename_local, exdir="./data/") 
}

glove_file_name <- "./data/wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"

# Load GloVe embeddings (I find sometimes trying to read as a csv can cause memory issues depending on your set-up)
IF_MEMORY_IS_SPARESE_LOAD_JUST_THIS_MANY_WORDS <- -1 # 100000to load everything, change this value to -1
glove_embeddings <- fread(glove_file_name, header = FALSE, quote = " ", nrows = IF_MEMORY_IS_SPARESE_LOAD_JUST_THIS_MANY_WORDS)
setnames(glove_embeddings, c("word", paste0("V", 1:50)))

# Convert the embeddings to a matrix and assign the words as teh row names
glove_matrix <- as.matrix(glove_embeddings[, -1, with = FALSE])
rownames(glove_matrix) <- glove_embeddings$word

#### Compute word similarity
get_word_similarity <- function(word1, word2, glove_words_matrix) {
  time_start <- Sys.time()
  print(paste("word similarity started at", time_start))
  
  vec1 <- glove_words_matrix[word1, ]
  vec2 <- glove_words_matrix[word2, ]
  
  # Compute norms
  norm1 <- sqrt(sum(vec1 * vec1))
  norm2 <- sqrt(sum(vec2 * vec2))
  
  # Compute cosine similarity (same as before)
  similarity <-   sum(vec1 * vec2) / (norm1 * norm2)
  
  timediff <- Sys.time() - time_start
  print(paste("word similarity complete taking", timediff))
  
  return(similarity)
}

get_word_similarity("hospital", "clinic", glove_matrix)

#### Return list of most similar words
get_connected_words <- function(target_word, glove_words_matrix, how_many = 10){
  time_start <- Sys.time()
  print(paste("get connected words started at", time_start))
  
  target_vec <- glove_words_matrix[target_word, ]
  
  glove_norms <- sqrt(rowSums(glove_words_matrix * glove_words_matrix))
  target_norm <- sqrt(sum(target_vec * target_vec))
  
  similarities <- (glove_words_matrix %*% target_vec) / (glove_norms * target_norm)
  
  similar_words_indexes <- order(similarities, decreasing = TRUE)[1:how_many]  # Indexes of Top X similar words
  similar_words <- rownames(glove_words_matrix)[similar_words_indexes] # actual Top X similar words
  
  timediff <- Sys.time() - time_start
  print(paste("get connected words complete taking", timediff))
  
  return(similar_words)
}

hospital_connected <- get_connected_words("hospital",glove_matrix)

some_words <- data.frame(word = c("hospital", "clinic", "practice"))
some_words_connections <- apply(some_words, 1, get_connected_words, glove_matrix) 
some_words_connections

############# Chapter 8: Named Entity Recognition ####################
library(gapminder)

### Count how many times different countries occur in the tweets
# squash all the tweets into a single string
squashed_text <- tokenised_words$word |>
  str_flatten(collapse = " ")

country_mentions <- country_codes |>
  mutate(country = str_to_lower(country),
         times_mentioned = str_count(squashed_text, country)) |>
  filter(times_mentioned > 0)

### replacing names using lexicon package - note that this is extremely ineffective
library(lexicon)

data("freq_first_names", package = "lexicon")
freq_first_names <- freq_first_names |>
  rename(name = Name) |>
  select(name) |>
  mutate(type = "[FIRST_NAME]")

data("freq_last_names", package = "lexicon")
freq_last_names <- freq_last_names |>
  rename(name = Surname) |>
  select(name) |>
  mutate(type = "[LAST_NAME]")

all_names <- bind_rows(freq_first_names, freq_last_names) |>
  mutate(name = str_to_lower(name)) |>
  distinct(name, .keep_all = TRUE)

# Anonymise the text using the names list
# loads of words identified as names... that are not names.. also lots of stop words
data_anon <- tokenised_words |>
  anti_join(expanded_stop_words) |>
  left_join(all_names, by = c("word" = "name")) |>
  mutate(word = if_else(!is.na(type), type, word)) |>
  #filter(!is.na(type))
  group_by(doc_id) |>
  mutate(text_anon = str_flatten(word, collapse = " ")) |>
  slice_head(n=1) |>
  ungroup() |>
  select(doc_id, text_anon) |> 
  left_join(data, by = "doc_id") |>
  select(doc_id, text_anon, nostop)

######### Function to return all names in a text for review - most are not names
find_names <- function(text, lookup) {
  
  words <-unlist(strsplit(text, "\\s+"))
  
  matched_names <- words[words %in% lookup]
  
  matched_names_tbl <- tibble(matched_names)
  
  # Return matched names or NA if no match
  if (length(matched_names) > 0) {
    return(matched_names_tbl)
  } else {
    return(NA)
  }
}

# Apply to data
found_names <- find_names(squashed_text, all_names$name)

####### Pre trained models for entity recognition ######
library(nametagger)

## nametagger seems rubbish at identifying female names
nametagger_model <- nametagger_download_model("english-conll-140408", model_dir = "models")


#we create a function to run the entity tagger
extract_entities <- function(text) {
  result <- predict(nametagger_model, text, split = '\\s+')  
  # Use mutate to create the merged column and select to keep only that column
  result
}

show_only_recognised <- function(entities){
  entities %>%
    filter(entity != 'O') %>%
    mutate(full_name = paste(term, entity, sep = "/")) %>%
    summarize(all_names = paste(full_name, collapse = ", "))  # Combine into a single string
}

text ="Hello Mary This is Albert who lives in Scotland. We work as teachers.  We have a cat Meilo and she is a tabby. Scotland is cold and rainy in the winter but I like Scotland"
entities_in_word <- extract_entities(text)
entities_in_word
show_only_recognised(entities_in_word)

text ="Hello I am Paul and this is Fred we live in Scotland. We work as school teachers.  We have a cat called Bob and he is a tabby. Scotland is cold and rainy in the winter but I like Scotland."
entities_in_word <- extract_entities(text)
entities_in_word
show_only_recognised(entities_in_word)

######## Entity package - will not work unless java is installed
# if (!require("pacman")) install.packages("pacman")
# pacman::p_load_gh("trinker/entity")
# library(entity)
# 
# data(wiki)
# entitiy_wiki <- wiki
# person_entity(wiki)
# 
# #person_entity
# #location_entity
# #organisation_entity
# #date_entity
# #money_entity
# #percent_entity
# 
# data$location <- location_entity(data$text)

############## Chapter 9: Sentiment Analysis ###################################
library(textdata)
nrc_joy <- get_sentiments("nrc") %>% 
  filter(sentiment == "joy")

# count joy words in each tweet
joy_count <- tokenised_words |>
  inner_join(nrc_joy) |>
  count(doc_id, sort = TRUE)

# classify each tweet as positive or negative
tokenised_sentiment <- tokenised_words |>
  left_join(get_sentiments("bing")) |>
  mutate(sentiment = if_else(is.na(sentiment), "neutral", sentiment)) |>
  group_by(doc_id) |>
  summarise(
    positive = sum(sentiment == "positive"),
    negative = sum(sentiment == "negative")
  ) |>
  mutate(overall_score = positive - negative,
         overall_class = case_when(overall_score > 0 ~ "positive",
                                   overall_score < 0 ~ "negative",
                                   overall_score == 0 ~ "neutral"))
  
  #arrange(doc_id)

# visualise the most common positive words and most common negative words
negative_words <- tokenised_words |>
  anti_join(stop_words) |>
  left_join(get_sentiments("bing")) |>
  filter(sentiment == "negative") |>
  count(word, sort=TRUE)

negative_cloud <- wordcloud2(
  data = negative_words,
  shape = 'circle'
)

library(wordcloud)
library(reshape)
library(reshape2)
tokenised_words %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>%
  comparison.cloud(colors = c("gray20", "gray80"),
                   max.words = 100)

# visualise the most common words associated with negative and positive tweets


############ Chapter 9: Topic Modelling ######################################
library(topicmodels)

### The models group the corpus of documents into topics in an unsupervised fashion by looking at
#co-existence of terms
# could use for triage notes/reasons for referral etc

# once you download the dataset with this line, you will have access to the variable AssociatedPress
data("AssociatedPress")

# try it, get a few terms
Terms(AssociatedPress) %>% head(10)

# To inspect it a bit (get the terms ordered from most popular), we could turn it into a dataframe, the arrange
ap_topics <- tidy(AssociatedPress) |>
  arrange(desc(count))

# k sets the number of topics to split the data into - like cluster analysis
ap_lda <- LDA(AssociatedPress, k = 8, control = list(seed = 1234))

lda_terms <- as_tibble(terms(ap_lda,10))
topic_dist <- posterior(ap_lda)$topics
print(head(topic_dist))

ap_topics <- tidy(ap_lda)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)

ap_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
