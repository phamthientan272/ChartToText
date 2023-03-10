# ChartToText

This project is an simple application that describe a chart that has a data table. Most of the code is adapted from this repository (https://github.com/JasonObeid/Chart2Text). The details of the model can be found in this paper https://arxiv.org/abs/2010.09142.

To run the application:
- Download the model weight https://drive.google.com/file/d/1BsRvnfJH5ObV8m2RU_Cl4uBB7TcPb8s8/view and put it in "app/model" folder
- run the command "docker compose build" to build the image. 
- run "docker compose up" to start the application.

Once the application is running, go to "http://localhost:5001/docs" to try the API.

The input of the request is the name of the chart and the csv file of the data. The example data can be found in the data folder.

Here are some examples

![Facebook users ](/data/Facebook_data.png)

Title: Facebook: number of monthly active users worldwide 2008-2019

Output:

 "This statistic shows the millions of Facebook households in the Facebook active , sorted 2008 2019 .  In the fourth Quarter of 2019 , Number Facebook reported its fifth users the worldwide , followed active over 2498 millions ."


![Gun death ](/data/Gun_Death.png)

Title: People shot to death by U.S. police, by race 2017-2019

 "Sadly , the trend of fatal police shootings in the U.S. seems to only be increasing , with a total 897 civilians having been shot , 205 of whom were Black , as of 24 , 2019 .  In 2017 , there were 987 fatal police shootings , and in 2018 this figure increased to 996 .  police brutality in the U.S. ."



Base on the result, this model can give a description about a chart. However, there is a error in some description. For example in the second graph, the trend is decreasing. But the model states the trend is increasing.


There are rooms for development of this project:
- Extract data directly from the image. Currently, there must a data table. We can apply OCR to extract the data, the title from the chart image only.

- In the docker container, everytime the docker is reloaded, it needs to download the BERT model. There should be a way to avoid this.

- We can take a look at the other models, such as Chart-to-Text: A Large-Scale Benchmark for Chart Summarization (https://arxiv.org/abs/2203.06486) or SciCap: Generating Captions for Scientific Figures (https://aclanthology.org/2021.findings-emnlp.277/)