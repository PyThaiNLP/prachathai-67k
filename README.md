# prachathai-67k
News Article Corpus from Prachathai.com

The `prachathai-67k` dataset was scraped from the news site [Prachathai](prachathai.com). We filtered out those articles with less than 500 characters of body text, mostly images and cartoons. It contains 67,889 articles wtih 51,797 tags from August 24, 2004 to November 15, 2018. The dataset was originally scraped by [@lukkiddd](https://github.com/lukkiddd) and cleaned by [@cstorm125](https://github.com/cstorm125). Download the dataset [here](https://www.dropbox.com/s/nfcrbbsd605ag6c/prachathai-67k.csv?dl=1). You can also see preliminary exploration in `exploration.ipynb`.

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

This is a **multi-label classification task** and we use **F1-score** for each label as the benchmark. The benchmark include both **title classification** and **body classification**.

※ Note that Prachathai.com is a left-leaning, human-right-focused news site, and thus unusual news labels such as human rights and quality of life.
