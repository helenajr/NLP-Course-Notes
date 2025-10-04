library(tidyverse)
library(tidytext)

theme_set(theme_bw())

data_full <- read_csv('./data/CORONA_TWEETS.csv')

data <- data_full |>
  select(doc_id, text) |>
  filter(doc_id < 10)

#################### Pre-processing ############################################

# count characters, words, sentences, lexical diversity in each doc
data <- data |>
  mutate(lower_text = str_to_lower(text),
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
