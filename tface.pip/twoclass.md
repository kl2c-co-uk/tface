

datapoints.FacePatch will need a class parmeter
- i'm going to add this as a string

def train() will need to write the new classes

def i_cartoon_datapoints() creates teh datapoionts so that's where it can be flagged as cartoon

the yolo5wider() method creates the datapoints - set the class there

def process_datapoint() has a line where it writes that out

YoloPipe.cs will need to be updated to handle the extra dimension. i should also fix the bubbles and allow "default accept" for this.



