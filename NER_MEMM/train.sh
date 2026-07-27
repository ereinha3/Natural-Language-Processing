python3 code/feature_builder.py train maxent/train.pos-chunk-name maxent/train.feat;
cd maxent/;
java -cp maxent-3.0.0.jar:trove.jar:. MEtrain train.feat NER-MEMM.model;
cd ..;
