**This repository contains a complete pipeline that training a seq2seq model needs, including training a model, and doing the test.**
## what you need to run the code
tensorflow and [opennmt(tensorflow version)](https://github.com/OpenNMT/OpenNMT-tf)
## A summary of this code
### data
data/generate_toy_data.py can create train, eval and test data. A data sample is like this:
> ("a b c z", "d e f c")

i.e, the target sentence is a character-level-3-right-shift of the source sentence.

### train and test
each toy_main_\[num\].py has similar content, but implements different seq2seq model.

Train: `python toy_main_[num].py --is_train True`

Test: `python toy_main_[num].py --is_train False`


### Result comparison
see record.txt