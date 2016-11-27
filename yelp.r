require(tm)
require(SnowballC)
require(wordcloud)
require(dplyr)
require(e1071)
require(caret)
input <- read.csv(file ="sample.csv", nrows = 1000, header = TRUE, stringsAsFactors = FALSE)
input$class <- ifelse(input$review_stars >3, input$class <- c("Yes"),input$class <- c("No"))
input$class <- as.factor(input$class)
input$class
corp <- Corpus(VectorSource(input$review_text))
corp
inspect(corp[1:3])
corp <- tm_map(corp,removePunctuation)
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, tolower)
corp <- tm_map(corp, removeWords, stopwords('english'))
corp <- tm_map(corp, stripWhitespace)
corp <- tm_map(corp, removeWords, c("the","this","that","with","its","are","their","just","get","will"))
corp <- tm_map(corp,PlainTextDocument)
corp <- tm_map(corp, stemDocument)
dtm <- DocumentTermMatrix(corp)
inspect(dtm)
input.train <- sample_n(input, 200)
input.test <- sample_n(input, 200)
dtm.train <- dtm[1:200,]
dtm.test <- dtm[201:400,]
corp.train <- corp[1:200]
corp.test <- corp[201:400]
dim(dtm.train)
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))
dtm.train.nb <- DocumentTermMatrix(corp.train, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corp.test, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)
system.time( classifier <- naiveBayes(trainNB, input.train$class, laplace = 1) )
system.time( pred <- predict(classifier, newdata=testNB) )
table("Predictions"= pred,  "Actual" = input.test$class )
conf.mat <- confusionMatrix(pred, input.test$class)
conf.mat$overall['Accuracy']

#forming word cloud using high frequency words
onestar <- filter(input, review_stars == 5)
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
