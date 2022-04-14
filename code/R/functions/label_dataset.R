suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))


#' label dataset using predefined labels 
#'
#' @param dataset a datafarme with the  dataset 
#' @param labels a dataframe  with labels
#' @return
#' @export
#'
#' @examples
#'
label_dataset <- function (dataset, labels) {
  nodes_infected <-
    labels %>% dplyr::filter(X3 == "infected") %>%
    select(X1) %>% unname() %>% unlist()
  dataset$label <- "normal"
  dataset_labeled <-
    dataset %>% 
    mutate(label = ifelse(node %in% nodes_infected, "infected", "normal"))
  dataset_labeled
}