require(tm)
require(SnowballC)
require(wordcloud)
require(dplyr)
require(e1071)
input <- read.csv(file ="sample.csv", nrows = 10000, header = TRUE, stringsAsFactors = FALSE)
#Filtering for review stars = 1
onestar <- filter(input, review_stars == 1)

corp <- Corpus(VectorSource(onestar$review_text))
corp <- tm_map(corp,removePunctuation)
corp <- tm_map(corp, tolower)
corp <- tm_map(corp, removeWords, stopwords('english'))
corp <- tm_map(corp, stripWhitespace)
corp <- tm_map(corp, removeWords, c("the","this","that","with","its","are","their","just","get","will"))
corp <- tm_map(corp,PlainTextDocument)
corp <- tm_map(corp, stemDocument)
wordcloud(corp, max.words = 50, random.order = FALSE)
dtm <- DocumentTermMatrix(corp)
termFreq <- colSums(as.matrix(dtm))
tf <- data.frame(term = names(termFreq), freq = termFreq)
tf <- tf[order(-tf[,2]),]
head(tf)


# need changes to be done on below code to make it work on this dataset.

inspect(dtm)
onestar.train <- onestar[1:100,]
onestar.test <- onestar[101:200,]
dtm.train <- dtm[1:100,]
dtm.test <- dtm[101:200,]
corpus.clean.train <- corp[1:100]
corpus.clean.test <- corp[101:200]
dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)
system.time( classifier <- naiveBayes(trainNB, onestar.train$class, laplace = 1) )
system.time( pred <- predict(classifier, newdata=testNB) )
table("Predictions"= pred,  "Actual" = df.test$class )
conf.mat <- confusionMatrix(pred, df.test$class)
conf.mat$overall['Accuracy']
