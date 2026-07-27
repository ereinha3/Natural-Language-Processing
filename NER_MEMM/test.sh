python3 code/feature_builder.py test maxent/test.pos-chunk maxent/test.feat;
cd maxent/;
java -cp maxent-3.0.0.jar:trove.jar:. MEtag test.feat NER-MEMM.model test.txt;
cd ..;