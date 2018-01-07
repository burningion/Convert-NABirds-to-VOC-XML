# Convert NABirds Dataset to VOC XML

Convert the NABirds Dataset to VOC XML for training in Darknet / Darkflow 

## Usage

Put this in the downloaded and extracted NABirds directory. It should be in the same directory as the `nabirds.py` file.

Set the `numberExamples` variable to -1 to convert the entire set to the labeled XML files. Otherwise, set it to the number of examples you'd like to train on.

Next, run the `generateVOCannotations.py` file, and it will create two directories. One will be called `Annotations`, and the other will be called `VOCImages`. You can use both of these to train from with Darkflow like this:

```bash
$ flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation nabirds/Annotations --dataset nabirds/VOCImages/
```
