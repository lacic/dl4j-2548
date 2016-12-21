package com.issue2548;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import java.io.File;

public class TrainExample {

    public static final int BATCH_SIZE = 100;
    public static final int N_EPOCHS = 10;


    public static void main(String... args) {
        // define some paths where you would like to store the model
        String modelPath = "data" + File.separator + "models" + File.separator;
        String normalizerPath = "data" + File.separator + "normalizers" + File.separator;

        String modelName = "train.model";
        String configName = "found_config.json";

        // define some paths to the files which will be used for training
        String trainFeaturePath = "data" + File.separator + "trainTestSplit" + File.separator +
                "train" + File.separator + "features" + File.separator;
        String trainLabelPath = "data" + File.separator + "trainTestSplit" + File.separator +
                "train" + File.separator + "labels" + File.separator;

        // there are 31 files/examples
        // for the sake of the example, maybe generate some random time series example files with integer values?
        Integer maxFieldId = 5;

        // setup helper for serialization
        SerializationUtils serializationHelper = new SerializationUtils(modelPath, modelName, normalizerPath, configName);

        // init network from config
        MultiLayerConfiguration config = serializationHelper.loadNetworkConfig();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new ScoreIterationListener(1000));


        // load data for training
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();

        try {
            File featuresDirTrain = new File(trainFeaturePath);
            File labelsDirTrain = new File(trainLabelPath);

            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, maxFieldId));
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, maxFieldId));
        } catch (Exception e) {
            // log error
        }

        boolean regression = true;
        int numClasses = -1; //not used for regression

        DataSetIterator trainingData = new SequenceRecordReaderDataSetIterator(
                trainFeatures,
                trainLabels,
                BATCH_SIZE,
                numClasses,
                regression,
                SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        // setup normalizer
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fitLabel(true);
        normalizer.fit(trainingData);

        trainingData.reset();
        trainingData.setPreProcessor(normalizer);
        // store it
        serializationHelper.storeNormalizer(normalizer);

        // train
        for (int j = 0; j < N_EPOCHS; j++) {
            trainingData.reset();
            net.fit(trainingData);
            System.out.println("Epoch: " + j);
        }

        // store network model
        serializationHelper.storeNetworkModel(net);

        // Successfully created and trained the NN model
    }

}