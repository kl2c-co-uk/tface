

datapoints.FacePatch will need a class parmeter
- i'm going to add this as a string

def train() will need to write the new classes
- got it tooo, seemed like i needed a 

def i_cartoon_datapoints() creates teh datapoionts so that's where it can be flagged as cartoon
- that worked

the yolo5wider() method creates the datapoints - set the class there
- that worked

def process_datapoint() has a line where it writes that out
- that's fine

tried to setup a home-training machine. i'm going to fork the project and send that tot he training m achine

YoloPipe.cs will need to be updated to handle the extra dimension. i should also fix the bubbles and allow "default accept" for this.


i should clamp exported height and width of images
