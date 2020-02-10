### SaP500 Volume prediction using wavenet
![S&P500 Volume](https://github.com/Darthholi/Sap500/blob/master/Volume.png?raw=true)
  
  
Detailed technical report in https://github.com/Darthholi/Sap500/blob/master/VolumePredictionWithWavenet.pdf  
Includes pretrained models.
  
First look at the data was done using colab notebook: https://colab.research.google.com/drive/1yuOsEa0zRQkE4p-qIeYYWsm7Ov8yBmH9
  
#### A note on reproducting the results
The experiments did run on Intel® Core™ i7-6700K CPU @ 4.00GHz × 8 with 15,6 GiB memory and
 GeForce GTX 1080 Ti/PCIe/SSE2 graphic card (and environment from
  https://lambdalabs.com/lambda-stack-deep-learning-software.  
 Since the results were dependent on the initial random training starting point,
  there is present a saved model`wavenet-startpoint-suggest.h5`, which can be used to continue the training 
  (the model is different only in some layers, so most of the weights can be reused).
    
  Running the code in google colab is shown in this section of the notebook: https://colab.research.google.com/drive/1yuOsEa0zRQkE4p-qIeYYWsm7Ov8yBmH9#scrollTo=ea2QjosCbNc6

#### Commands:  
Setting up the environment:
```  
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
pip install --upgrade -r requirements.txt
```

Download the data not present here: `./download.sh`  
  
Train from scratch:
`python cmd-train --model_name="wavenet/wavenet.h5"`  
Use a pretrained optimum:   
`cmd-train --model_name="wavenet/wavenet.h5" --start_weights_from="wavenet-startpoint-suggest.h5"`  
Evaluate a model and plot graphs:  
`cmd-eval --model_name="wavenet1452/wavenet.h5" --cache_name="wavenet1452/cached/"`  
