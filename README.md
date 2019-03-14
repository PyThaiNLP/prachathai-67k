# prachathai-67k
News Article Corpus from Prachathai.com

The `prachathai-67k` dataset was scraped from the news site [Prachathai](prachathai.com). We filtered out those articles with less than 500 characters of body text, mostly images and cartoons. It contains 67,889 articles wtih 51,797 tags from August 24, 2004 to November 15, 2018. The dataset was originally scraped by [@lukkiddd](https://github.com/lukkiddd) and cleaned by [@cstorm125](https://github.com/cstorm125). Download the dataset [here](https://www.dropbox.com/s/fsxepdka4l2pr45/prachathai-67k.zip?dl=1). You can also see preliminary exploration in `exploration.ipynb`.

This dataset is a part of [pyThaiNLP](https://github.com/PyThaiNLP/) Thai text [classification-benchmarks](https://github.com/PyThaiNLP/classification-benchmarks). For the benchmark, we selected the following tags with substantial volume that resemble **classifying types of articles**※:

* `การเมือง` - politics
* `สิทธิมนุษยชน` - human rights
* `คุณภาพชีวิต` - quality of life
* `ต่างประเทศ` - international
* `สังคม` - social
* `สิ่งแวดล้อม` - environment
* `เศรษฐกิจ` - economics
* `วัฒนธรรม` - culture
* `แรงงาน` - labor
* `ความมั่นคง` - national security
* `ไอซีที` - ICT
* `การศึกษา` - education

We provide 3 benchmarks for 12-topic multi-label classification of `prachathai-67k`: [fastText](https://github.com/facebookresearch/fastText), LinearSVC and [ULMFit](https://github.com/cstorm125/thai2fit). In all cases, we first finetune the embeddings using all data. The data is then split into train, validation and test sets at 70/10/20 split. The benchmark numbers are based on the test set. Performance metrics are macro-averaged accuracy and F1 score. See [classification.ipynb](https://github.com/PyThaiNLP/prachathai-67k/blob/master/classification.ipynb) for more information.

| model     | macro-accuracy | macro-F1 |
|-----------|----------------|----------|
| fastText  | 0.9302         | 0.5529   |
| LinearSVC | 0.513277       | 0.552801 |
| **ULMFit**    | **0.948737**       | **0.744875**	 |

※ Note that Prachathai.com is a left-leaning, human-right-focused news site, and thus unusual news labels such as human rights and quality of life.
