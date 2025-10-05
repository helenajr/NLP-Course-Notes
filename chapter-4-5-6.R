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
  #anti_join(stop_words) |>
  count(word, sort=TRUE) |>
  filter(n > 0)

set.seed(1234) # word cloud the same every time

# expects a dataframe with words or ngrams in one column and counts in another
wordcloud2(
  data = cloud_words,
  shape = 'circle'
)

### network graph
library(ggraph)

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

