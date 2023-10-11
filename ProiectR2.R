library(tidyverse)
library(modelr)
library(scatterplot3d)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)
library(rsample) #pentru initial.split()

Streamuri <- read_csv("Piese.csv")

Streamuri %>%
  ggplot(aes(Danceability , Stream)) + geom_point() + geom_smooth()

Streamuri %>%
  ggplot(aes(Energy , Stream)) + geom_point() + geom_smooth()

Streamuri %>%
  ggplot(aes(Speechiness , Stream)) + geom_point() + geom_smooth()

Streamuri %>%
  ggplot(aes(Duration, Stream)) + geom_point() + geom_smooth()

Streamuri %>%
  ggplot(aes(Liveness , Stream)) + geom_point() + geom_smooth()

Streamuri %>%
  ggplot(aes(Tempo, Stream)) + geom_point() + geom_smooth()

#regresia liniara a stream-urilor in functie de danceability
streamuri_danceability <- lm(data = Streamuri, Stream ~ Danceability)
summary(streamuri_danceability)

#plot the data against the learned regresion line
grid_Danceability <- Streamuri %>% 
  data_grid(Danceability = seq_range(Danceability, 100)) %>%
  add_predictions(streamuri_danceability, "Stream")

ggplot(Streamuri, aes(Danceability, Stream)) +
  geom_point() +
  geom_line(data = grid_Danceability, color = "green", size=1)

confint(streamuri_danceability)

#regresia liniara a stream-urilor in functie de energy

streamuri_energy <- lm(data = Streamuri, Stream ~ Energy)
summary(streamuri_energy)

#plot the data against the learned regresion line
grid_Energy <- Streamuri %>% 
  data_grid(Energy = seq_range(Energy, 100)) %>%
  add_predictions(streamuri_energy, "Stream")

ggplot(Streamuri, aes(Energy, Stream)) +
  geom_point() +
  geom_line(data = grid_Energy, color = "green", size=1)

confint(streamuri_energy)

#regresia liniara a stream-urilor in functie de speechiness

streamuri_speechiness <- lm(data = Streamuri, Stream ~ Speechiness)
summary(streamuri_speechiness)

#plot the data against the learned regresion line
grid_Speechiness <- Streamuri %>% 
  data_grid(Speechiness = seq_range(Speechiness, 100)) %>%
  add_predictions(streamuri_speechiness, "Stream")

ggplot(Streamuri, aes(Speechiness, Stream)) +
  geom_point() +
  geom_line(data = grid_Speechiness, color = "green", size=1)

confint(streamuri_speechiness)

#regresia liniara a stream-urilor in functie de duration

streamuri_duration <- lm(data = Streamuri, Stream ~ Duration)
summary(streamuri_duration)

#plot the data against the learned regresion line
grid_Duration <- Streamuri %>%
  data_grid(Duration = seq_range(Duration, 100)) %>%
  add_predictions(streamuri_duration, "Stream")

ggplot(Streamuri, aes(Duration, Stream)) +
  geom_point() +
  geom_line(data = grid_Duration, color = "green", size=1)

confint(streamuri_duration)

#regresia liniara a stream-urilor in functie de liveness

streamuri_liveness <- lm(data = Streamuri, Stream ~ Liveness)
summary(streamuri_liveness)

#plot the data against the learned regresion line
grid_Liveness <- Streamuri %>%
  data_grid(Liveness = seq_range(Liveness, 100)) %>%
  add_predictions(streamuri_liveness, "Stream")

ggplot(Streamuri, aes(Liveness, Stream)) +
  geom_point() +
  geom_line(data = grid_Liveness, color = "green", size=1)

confint(streamuri_liveness)


#regresia liniara a stream-urilor in functie de tempo

streamuri_tempo <- lm(data = Streamuri, Stream ~ Tempo)
summary(streamuri_tempo)

#plot the data against the learned regresion line
grid_Tempo <- Streamuri %>%
  data_grid(Tempo = seq_range(Tempo, 100)) %>%
  add_predictions(streamuri_tempo, "Stream")

ggplot(Streamuri, aes(Tempo, Stream)) +
  geom_point() +
  geom_line(data = grid_Tempo, color = "green", size=1)

confint(streamuri_tempo)

# Formula regresiei liniare cu mai mulți parametrii
reg_toate <- lm(Stream ~ Danceability + Tempo + Speechiness + Duration + Liveness + Energy, data = Streamuri)

# Sumarul modelului de regresie liniară
summary(reg_toate)

# Formula regresiei liniare cu cei patru parametrii ramasi
reg_final <- lm(Stream ~ Danceability  + Speechiness + Liveness + Energy, data = Streamuri)

# Sumarul modelului de regresie liniară
summary(reg_final)


# Scalarea variabilei "Stream"
scaled_Stream <- Streamuri$Stream / 1000000  # Împărțim valorile la 1.000.000 pentru a le aduce la o scară mai mică

# Generarea graficului 3D
s3d <- scatterplot3d(Streamuri$Danceability, Streamuri$Energy, Streamuri$Speechiness, color="blue", pch=16, xlab="Danceability", ylab="Energy", zlab="Speechiness")
s3d$points3d(Streamuri$Danceability, Streamuri$Energy, Streamuri$Speechiness, col="blue", pch=16)  # Adăugăm punctele pentru variabilele inițiale
s3d$plane3d(reg_final)  # Adăugăm planul de regresie

# Etichetarea axei "Stream" cu valorile scalate
s3d$zlim <- range(scaled_Stream)  # Setăm limitele axei "Stream" la valorile scalate
s3d$zlab <- "Stream (scaled)"  # Setăm eticheta axei "Stream" la "Stream (scaled)"
s3d$points3d(Streamuri$Danceability, Streamuri$Energy, Streamuri$Speechiness, col="blue", pch=16)  # Adăugăm punctele pentru variabilele inițiale


#predictia

pred_stream <- tibble(
  Danceability = 0.4,
  Liveness = 0.3,
  Energy = 0.9,
  Speechiness= 0.5
)
predict(reg_final, newdata = pred_stream, interval="confidence")
predict(reg_final, newdata = pred_stream, interval="prediction")



#arborii de decizie

Streamuri %>%
  ggplot(aes(Stream)) + geom_density()

#impartirea setului de date (90-10)

set.seed(100)
piese_split <- initial_split(Streamuri, prop = 0.70)
piese_train <- training(piese_split)
piese_test <- testing(piese_split)

m1 <- rpart(
  formula = Stream ~ ., # Comparare streamuri contra restul variabilelor
  data = piese_train,
  method = "anova",
  control = rpart.control(cp = 0.001) # Setăm parametrul complexity (cp) mai mic pentru a permite crearea de ramuri suplimentare în arbore
)

print(m1) # Afișarea arborelui rezultat (în mod text)
rpart.plot(m1) # Afișarea grafică a arborelui rezultat
plotcp(m1)
print(m1$cptable) # Afișarea parametrilor alpha


#cautam cele mai bune valori pentru parametri minsplit si maxdepth
hyper_grid <- expand.grid(
  minsplit = seq(5, 20, 1),
  maxdepth = seq(8, 15, 1)
)
head(hyper_grid)
models <- list()
for (i in 1:nrow(hyper_grid)) {
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  models[[i]] <- rpart(
    formula = Stream ~. ,
    data = piese_train,
    method = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}
get_cp <- function(x) {
  min <- which.min(x$cptable[,"xerror"])
  cp <- x$cptable[min, "CP"]
}
get_min_error <- function(x) {
  min <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"]
}


mutated_grid <- hyper_grid %>%
  mutate(
    cp = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  )  
mutated_grid %>%
  arrange(error) %>%
  top_n(-5, wt=error)

optimal_tree <- rpart(
  formula = Stream ~ .,
  data = piese_test,
  method = "anova",
  control = list(minsplit = 5, maxdepth = 8, cp = 0.006551022 )
)

pred <- predict(m1, newdata = piese_test)
RMSE(pred = pred, obs = piese_test$Stream)
optimal_tree

#BAGGING
library(ipred)
set.seed(100)

