# Team Bosch Bisch: root cause analysis of a production line

## Description:
This project uses artficial intelligence to analyse a data set containing 5000 training samples in order to determine the causation relationship betweens stages of a production pipeline. Using statistical methods we produce a normalised causation matrix which we display graphically on our front end built in React with MUI CSS framewrok.


## Where things are 
`Claudio/Matrices` directory contains some different matrices that were obtained through various training methods

## Install instructions 
Install all npm dependancies by running `npm install`. the rest is your problem... Just joking - run `pip install -r requirements.txt`.

## Usage - How to run:
Run the script in `Claudio/Scripts/CausalDiscoery` to generate the causality matrix. `Claudio/Scripts/CausalInference` is used to generate causality weights.  Run the bosch Bisch bash file: `boschBisch.sh` from the command line to see our BEAUTIFUL (😙🤌) front end! by default we show the all the relationships causing the 85th sensor to give higher results which are indicative of a high scrap rate (faulty process). You can filter by only the sensors which are found to be causing this by using the threshold slider on the top right. sensors with a higher causation relationship are displayed in darker red for fast and easy interpretation. One can chose to find causation between other sensors by clicking on other values and can see the data and colours change accordingly. One can also click the causes/caused by switch to toggle the direction of the relationships if you wish to see which other sensors are likely influenced by this value.

## Note
The weights defining the causal relationships in the frontend part are initialized randomly. This was due to the fact that we didn't have te weights ready to perform the demo. this allowed us to work on the frontend independently from the backend.
