

# Preprocessing

The format of ***MRCONSO.csv*** of the NMLS database.

e.g.

```
C0000005|ENG|P|L0000005|PF|S0007492|Y|A26634265||M0019694|D012711|MSH|PEP|D012711|(131)I-Macroaggregated Albumin|0|N|256|
```

We use ***extract.py*** to get the features needed for scoring.

- SAB (**Source Abbreviation**)
- TTY (**Term Type**)
- SUPPRESS (**SUPPRESS Description**)

And also some other information

- CUI (**Unique identifier for concept**)
- STR (**String**)

***examples***

![NMLS-SPS/1.png at main · Hawkaoaoa/NMLS-SPS (github.com)](https://github.com/Hawkaoaoa/NMLS-SPS/blob/main/figures/1.png)

Then we score each string based on the following rules:

1. Rankings in the NMLS recommended preference table

   (linear affinity)

   > [Source and Term Types: Default Order of Precedence and Suppressibility - 2022AA Release (nih.gov)](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/precedence_suppressibility.html)

2. The length of the string

   - Generally, we'd like the string to be brief in order to integrated into the knowledge graph.

   Considering the input length restricts of conventional bert-based model, we first remove all the string with length longer than 510 and take a glimpse on the length distribution.

3. ...

> All the score is limited to the range from 0 to 1.

***Length < 510***

![NMLS-SPS/2.png at main · Hawkaoaoa/NMLS-SPS (github.com)](https://github.com/Hawkaoaoa/NMLS-SPS/blob/main/figures/2.png)

​	Most of the data has a shorter length than 100, so we primarily set it as the threshold for eliminating the strings (Of course, strings with length long than this are not preferred, and should be labeled with very low score even without modeling.)

​	As for the data left, we adopt a linear penalty for strings with length longer than 50(empirically for now).

> Some strings are the same with  a single **CUI**. Considering we are scoring merely on the preference of the strings, we simply keep only one string with the highest score.

​	By running the ***score.py***, we could get the raw score data file from the NMLS corpus.

***More modification***

​	We notice that, even though some strings belong to different **CUI**, they may also be the same literally. And also, some non-English characters exist in the strings, which are inappropriate for the model and the knowledge graph. 

​	As for these problems, we clean the raw data file with ***filter.py*** and get the data needed for the model training.

***Preprocessed dataset***

- 6219671 strings
- 9:1 for train num : test num

# Model

​	The model we use is ***CODER***, which is provided by Zheng Yuan *et al.*

> [GanjinZero/CODER: CODER: Knowledge infused cross-lingual medical term embedding for term normalization. [JBI, ACL-BioNLP 2022\] (github.com)](https://github.com/GanjinZero/CODER)

Additionally, when we check the scores distribution of the data

***scores distribution***

![NMLS-SPS/3.png at main · Hawkaoaoa/NMLS-SPS (github.com)](https://github.com/Hawkaoaoa/NMLS-SPS/blob/main/figures/3.png)

we could see the data is clearly very imbalanced. 

​	As for the problem of imbalanced data distribution for the regression task, we adopt the newly proposed loss function ***Balanced MSE**, which is provided by Jiawei Ren *et al*.

> [jiawei-ren/BalancedMSE: [CVPR 2022 Oral\] Balanced MSE for Imbalanced Visual Regression https://arxiv.org/abs/2203.16427 (github.com)](https://github.com/jiawei-ren/BalancedMSE)

- preliminary experiments indicate that this loss function is much helpful for the imbalanced problem for regression.

The basic functions essential for training are defined in ***utils.py***

And  the main function is ***main.py***

```python
python main.py --num_epochs 30
```

> ***Available Args***
>
> - pretrained_model_name
>
>   (pretrained bert model to be used, the format is the same with **huggingface transformers style**)
>
> - init_sigma
>
>   (init parameters for ***BMC loss***)
>
> - sigma_lr
>
>   (learning rate for the parameters of ***BMC loss***)
>
> - ....
>

# Results

Some analyzing functions are provided in the ***stat.py***

# Case-study

1.Generally, the strings with more precedence in the NMLS recommended table and also shorter length will get higher scores.

e.g.

```
Dipalmitoyllecithin	0.9862183020948181
Droxidopa	0.9878721058434399
Pymadine	0.9862183020948181

immunoglobulin transcytosis in epithelial cells mediated by polymeric immunoglobulin receptor	0.11607497243660418
directional guidance of interneurons involved in migration from the subpallium to the cortex	0.13265711135611907
```

2.Strings with direct expressions (**for the same concepts**) are more preferred.

e.g.

```
Branching Enzyme	0.9862183020948181
Enzyme, Branching	0.7535832414553473 # unecessary format!

Starch Branching Enzyme	0.9862183020948181
Branching Enzyme, Starch	0.7535832414553473
Enzyme, Starch Branching	0.7535832414553473

15S RNA	0.9867695700110254
RNA, 15S	0.7535832414553473
```

3.Strings for the concepts of metabolites with only uppercase characters are less preferred, while for the concepts of specific terms with only lowercase characters are less preferred.

- This is very meaningful for the related application scenes.

e.g.

```
# "metabolites"
Meglutol	0.9878721058434399
MEGLUTOL	0.896361631753032

Tenamfetamine	0.964167585446527
TENAMFETAMINE	0.896361631753032

# "specific terms"
MPTP	0.9862183020948181
mptp	0.7276736493936053

MDA	0.9636163175303197
mda	0.7276736493936053
```

4.Some strings seem to be brief enough but have the score of 0, this is mainly because they are not matched to the NMLS recommended table. The possible reasons are 

- The sources are not reliable
- The terms are not updated to date

e.g.

```
rosy bitterling	0.0
golden snapper	0.0
Japanese rose	0.0
redstriped eartheater	0.0
```

5.Some strings include many **punctuations**. It may be reasonable for the specific substances such as chemicals, but may be inappropriate for those with other unecessary information.

e.g.

```
# substances, like chemicals
2,2' Dipyridyl	0.9862183020948181
24,25(OH)2 D3	0.8710033076074972
[2-(2,6-dichloroanilino)phenyl]acetic acid	0.9636163175303197

# unecessary punctuation for additional description
Capital Expenditures	0.9878721058434399
capital [economics]	0.7276736493936053

Capitation Fee	0.9878721058434399
Capitation fees (Medical care)	0.7585446527012129

Acholeplasma laidlawii	0.9878721058434399
Acholeplasma laidlawii (Freundt 1955 (sic)) Edward and Freundt 1970	0.5392061742006615

# tedious punctuations
movimento:Osservazione:Pt:Cuore.Ventricolo.sinistro.apice:Ord:Angiogramma	0.0871444321940463
2,2',2'',2''''-[(4,8-Dipiperidinopyrimido-[5,4-d]pyrimidine-2,6-diyl)dinitrilo]tetraethanol	0.1576736493936053
```

6.Short strings with separate single or few characters may be the transformed version of some standard  terms, so they may have lower scores.

e.g.

```
D-D	0.9614112458654906
d d	0.7276736493936053

INK4A Protein	0.9862183020948181
P A 016 INK 004 PROTEIN	0.6334068357221609
```

7.Some strings represent the same meaning but with slight difference for the quantity of certain substances.

e.g.

```
URINARY DRAINAGE SHEATH,UROCARE #4409	0.9696802646085998
URINARY DRAINAGE SHEATH,UROCARE #4410	0.9696802646085998

Trophamine 10 % Injectable Product	0.9966923925027563
Trophamine 6 % Injectable Product	0.9966923925027563
```

- whether only one should be kept needs further consultation for the practical considerations.

Specifically, the ***predict.py*** can be used to evaluate the the sting you input 

```
python predict.py --string="Medical service"
```

>  Some other Args
>
> - pretrained_model_name
>
>   (load basic bert model structure)
>
> - pretrained_model_path
>
>   (the path to the model fine-tuned for the string scoring)
>
> - device

***Demonstrations***

```
>> python predict.py --string="AASMv2.6"
The score for "AASMv2.6" is 0.8253
>> python predict.py --string="a a s m v2.6"
The score for "a a s m v2.6" is 0.5661

>> python predict.py --string="aspirin"
The score for "aspirin" is 0.9739
>> python predict.py --string="%aspirin#"
The score for "%aspirin#" is 0.6599

>> python predict.py --string="drug [AS23]"
The score for "drug [AS23]" is 0.9981
>> python predict.py --string="Give me another drug [AS23] to treat the patient P2314"
The score for "Give me another drug [AS23] to treat the patient P2314" is 0.6328
```

# Requirements

- transformers 4.18.0
- scikit-learn 0.24.2
- scipy 1.5.4
- seaborn 0.11.2
- sentencepiece 0.1.96
- torch 1.10.2+cu113
- torchaudio 0.10.2+cu113
- torchvision 0.11.3+cu113