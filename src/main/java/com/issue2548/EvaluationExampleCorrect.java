package com.issue2548;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class EvaluationExampleCorrect {

    public static void main(String... args) {
        Nd4j.create(1);

        // where are the trained model and the corresponding normalizers stored ?
        String modelPath = "data" + File.separator + "models" + File.separator;
        String normalizerPath = "data" + File.separator + "normalizers" + File.separator;

        String modelName = "train.model";

        // define the paths to the train features, test features and test labels
        // train test split can be 80/20
        String trainFeaturePath = "data" + File.separator + "trainTestSplit" + File.separator +
                "train" + File.separator + "features" + File.separator;

        String testFeaturePath = "data" + File.separator + "trainTestSplit" + File.separator +
                "test" + File.separator + "features" + File.separator;
        String testLabelPath = "data" + File.separator + "trainTestSplit" + File.separator +
                "test" + File.separator + "labels" + File.separator;

        // there are 31 files/examples used for training
        Integer maxFieldId = 5;

        // setup helper for serialization
        SerializationUtils serializationHelper = new SerializationUtils(modelPath, modelName, normalizerPath);

        try {
            MultiLayerNetwork net = serializationHelper.loadNetwork();
            NormalizerStandardize normalizer = serializationHelper.loadNormalizer();

            Map<String, List<String>> trainFeaturesMap = extractRows(trainFeaturePath, 0, maxFieldId);
            Map<String, List<String>> testFeaturesMap = extractRows(testFeaturePath, 0, maxFieldId);
            Map<String, List<String>> testLabelsMap = extractRows(testLabelPath, 0, maxFieldId);


            for (String fileName : trainFeaturesMap.keySet()) {
                // reset model for new file evaluation
                net.rnnClearPreviousState();

                List<String> trainFeatures = trainFeaturesMap.get(fileName);
                List<String> testFeatures = testFeaturesMap.get(fileName);
                List<String> testLabels = testLabelsMap.get(fileName);

                // init train history
                for (String trainFeature : trainFeatures) {
                    INDArray featureArray = createArray( Integer.parseInt(trainFeature) );

                    if (normalizer != null) {
                        normalizer.transform(featureArray);
                    }

                    // init with value
                    INDArray initOutput = net.rnnTimeStep(featureArray);
                }


                INDArray rnnOutput = null;
                Double predicted = null;
                // evaluate on test set
                for (int testIndex = 0; testIndex < testFeatures.size(); testIndex++) {
                    String inputValue = testFeatures.get(testIndex);

                    INDArray featureArray = createArray( Integer.parseInt(inputValue) );


                    if (normalizer != null) {
                        normalizer.transform(featureArray);
                    }

                    rnnOutput = net.rnnTimeStep(featureArray);
                    normalizer.revertLabels(rnnOutput);

                    // extract double value out of the output, check expected vs predicted difference (RMSE, MAPE, etc.)
                    Integer expected = Integer.parseInt( testLabels.get(testIndex) );
                    // ...
                }

            }

            System.out.println("Successfully tested the NN model");
        } catch (Exception e) {
            // log error
        }
    }


    /**
     * Extracts row values from the provided data
     * @param path Path to the files to read
     * @return a map for each file containing a list of the values (1 value per row)
     */
    public static Map<String, List<String>> extractRows(String path, Integer minFileId, Integer maxFileId) {
        Map<String, List<String>> fileMap = new HashMap<>();

        for (int i = minFileId; i <= maxFileId; i++) {
            List<String> list = new ArrayList<>();

            String fileName = i + ".csv";
            Path filePath = Paths.get(path, fileName);
            try (BufferedReader br = Files.newBufferedReader(filePath)) {
                //br returns as stream and convert it into a List
                list = br.lines().collect(Collectors.toList());
            } catch (IOException e) {
                e.printStackTrace();
            }

            fileMap.put(fileName, list);
        }

        return fileMap;
    }

    /**
     * Creates an INDArray using an integer value
     * @return INDArray to be used for RNN nets in dl4j
     */
    public static INDArray createArray(Integer value) {
        // number of time steps used for testing the next value
        INDArray data = Nd4j.ones(1, 1, 1);

        double[] independent = new double[1];
        independent[0] = value;

        INDArray ind = Nd4j.create(independent, new int[]{1, 1, 1});
        data.putScalar(0, 0, 0, ind.getDouble(0));

        return data;
    }

}