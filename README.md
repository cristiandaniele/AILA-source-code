# AILA-source-code
Methodology for Automated and Intelligent Likelihood Assignment

The repository is divided into three folders:
1) Summariser folder: contains a script that summarises a text using n-grams, POS-tagging and entropy measurement.
   - Run:
    python3 Summariser_tool.py -h 
     
2) Entity Extractor folder: contains a script that extracts the named entities from a summarised text and then gathers, per each entity, all the sentences of the original text which contain such entity, including its synonyms, as well as a verb.
   - Edit "entity_extractor.py" file:
    original_text=[path of yours original policy]
    summarised_text=[path of yours summarised policy]
   - Run:
    python3 entity_extractor.py
   
3) Machine Learning Model: contains all files you need to train the model, in detail:
   1) a script that creates and trains a CNN [fairnessAnalysis.py]
   2) the dataset used to train the model [dataset.txt]
   3) three Json file [entities_toyota.json, entities_mercedes.json, and entities_tesla.json] containing the entity extracted using entity_extractor.py 
   - Edit "fairnessAnalysis.py" file:
    policyName=[policy you want to review]
   - Run:
    python3 fairnessAnalysis.py

   
