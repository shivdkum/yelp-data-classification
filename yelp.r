#including libraries
require(tm)
require(SnowballC)
require(wordcloud)
require(dplyr)
require(e1071)
require(caret)
require(RTextTools)

#Input file from the yelp dataset
input <- read.csv(file ="sample.csv", nrows = 10000, header = TRUE, stringsAsFactors = FALSE)

#Wordcloud for one star reviews
onestar <- filter(input, review_stars == 1)
corp <- Corpus(VectorSource(onestar$review_text))
corp <- tm_map(corp,removePunctuation)
corp <- tm_map(corp, tolower)
corp <- tm_map(corp, removeWords, stopwords('english'))
corp <- tm_map(corp, stripWhitespace)
corp <- tm_map(corp, removeWords, c("the","this","that","with","its","are","their","just","get","will"))
corp <- tm_map(corp,PlainTextDocument)
corp <- tm_map(corp, stemDocument)
wordcloud(corp, max.words = 80, random.order = FALSE)

#Finding high frequency words in one star reviews
dtm <- DocumentTermMatrix(corp)
termFreq <- colSums(as.matrix(dtm))
tf <- data.frame(term = names(termFreq), freq = termFreq)
tf <- tf[order(-tf[,2]),]
head(tf)

#wordcloud for five star reviews
fivestar <- filter(input, review_stars == 5)
corp <- Corpus(VectorSource(fivestar$review_text))
corp <- tm_map(corp,removePunctuation)
corp <- tm_map(corp, tolower)
corp <- tm_map(corp, removeWords, stopwords('english'))
corp <- tm_map(corp, stripWhitespace)
corp <- tm_map(corp, removeWords, c("the","this","that","with","its","are","their","just","get","will"))
corp <- tm_map(corp,PlainTextDocument)
corp <- tm_map(corp, stemDocument)
wordcloud(corp, max.words = 80, random.order = FALSE)

#Finding high frequency words in five star reviews
dtm <- DocumentTermMatrix(corp)
termFreq <- colSums(as.matrix(dtm))
tf <- data.frame(term = names(termFreq), freq = termFreq)
tf <- tf[order(-tf[,2]),]
head(tf)

#Predicting reviews raing from review text using different classifers
input <- read.csv(file ="sample.csv", nrows = 100, header = TRUE, stringsAsFactors = FALSE)
input$class <- ifelse(input$review_stars >3, input$class <- c("Yes"),input$class <- c("No"))
input$class <- as.factor(input$class)
textmat <- create_matrix(input$review_text, language="english", removeNumbers=TRUE, stemWords=TRUE, removeSparseTerms=.998)
container <- create_container(textmat, input$review_stars, trainSize=1:80, testSize=81:100, virgin=FALSE)
svm_model <- train_model(container,"SVM")
boosting_model <- train_model(container,"BOOSTING")
bagging_model <- train_model(container,"BAGGING")
randomforest_model <- train_model(container,"RF")
neuralnet_model <- train_model(container,"NNET")
svm_pred <- classify_model(container, svm_model)
boosting_pred <- classify_model(container, boosting_model)
bagging_pred <- classify_model(container, bagging_model)
randomforest_pred <- classify_model(container, randomforest_model)
neuralnet_pred <- classify_model(container, neuralnet_model)
abstract <- create_analytics(container, cbind(svm_pred,boosting_pred,bagging_pred,randomforest_pred,neuralnet_pred))
summary(abstract)

# summary
topic_summary <- analytics@label_summary
alg_summary <- analytics@algorithm_summary
ens_summary <-analytics@ensemble_summary
doc_summary <- analytics@document_summary
create_ensembleSummary(analytics@document_summary)
write.csv(analytics@document_summary, "DocumentSummary.csv")

#Running valdiation and calculating accuracy of the classifiers
svm <- cross_validate(container, 5, "SVM")
bagging <- cross_validate(container, 5, "BAGGING")
boosting <- cross_validate(container, 5, "BOOSTING")
randomforest <- cross_validate(container, 5, "RF")
neuralnet <- cross_validate(container, 5, "NNET")
