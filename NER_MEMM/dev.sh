python3 code/feature_builder.py dev maxent/dev.pos-chunk maxent/dev.feat;
cd maxent/;
java -cp maxent-3.0.0.jar:trove.jar:. MEtag dev.feat NER-MEMM.model dev.txt;
cd ..;
python3 code/score.name.py maxent/dev.name maxent/dev.txt;