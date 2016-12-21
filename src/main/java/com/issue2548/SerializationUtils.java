package com.issue2548;

import com.google.common.io.Files;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;

public class SerializationUtils {

    private final Logger logger = LoggerFactory.getLogger(getClass());

    private String modelPath;
    private String modelName;
    private String normalizerPath;
    private String configName;

    public SerializationUtils(String modelPath, String modelName, String normalizerPath) {
        this.modelName = modelName;
        this.modelPath = modelPath;
        this.normalizerPath = normalizerPath;
    }

    public SerializationUtils(String modelPath, String modelName, String normalizerPath, String configName) {
        this.modelName = modelName;
        this.modelPath = modelPath;
        this.normalizerPath = normalizerPath;
        this.configName = configName;
    }


    /**
     * Stores the provided network model.
     * Uses the previously assigned <i>modelPath</i> and <i>modelName</i> attributes to store the data.
     *
     * @param net model to save
     */
    public void storeNetworkModel(MultiLayerNetwork net) {
        if (net == null)
            return;

        try {
            File storedModel = new File(modelPath, modelName);
            // deletes previously created model
            storedModel.delete();
            FileOutputStream fos = new FileOutputStream(storedModel);
            ModelSerializer.writeModel(net, fos, true);
        } catch (IOException e) {
            logger.error("Failed storing model [" + modelName + "].", e);
        }
    }



    /**
     * Stores the provided normalizer as mean and std files.
     * Uses the previously assigned <i>normalizerPath</i> and <i>modelName</i> attributes to store the data.
     *
     * @param normalizer normalizer to store
     */
    public void storeNormalizer(NormalizerStandardize normalizer) {
        if (normalizer == null)
            return;

        File featuresMeanFile = new File(normalizerPath, "normalizer-stand-feature-mean-" + modelName);
        File featuresStdFile = new File(normalizerPath, "normalizer-stand-feature-std-" + modelName);

        File labelsMeanFile = new File(normalizerPath, "normalizer-stand-labels-mean-" + modelName);
        File labelsStdFile = new File(normalizerPath, "normalizer-stand-labels-std-" + modelName);


        try {

            if(!featuresMeanFile.exists()){
                featuresMeanFile.getParentFile().mkdirs();
            }

            normalizer.save(featuresMeanFile, featuresStdFile, labelsMeanFile, labelsStdFile);
        } catch (Exception e) {
            logger.error("Failed storing normalizer for paths : [" + featuresMeanFile.getPath() + "] and [" + featuresStdFile.getPath() + "]", e);
        }
    }

    /**
     * Loads the a network model.
     * Uses the previously assigned <i>modelPath</i> and <i>modelName</i> attributes to store the data.
     *
     * @return previously stored net model
     */
    public MultiLayerNetwork loadNetwork() throws IOException {
        Nd4j.getRandom().setSeed(12345);

        File netFile = new File(modelPath, modelName);
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(netFile);
        return net;
    }

    /**
     * Loads the a network configuration.
     * Uses the previously assigned <i>modelPath</i> and <i>modelName</i> attributes to store the data.
     *
     * @return previously json content
     */
    public MultiLayerConfiguration loadNetworkConfig() {
        String jsonConfig = loadNetworkConfigJson();
        MultiLayerConfiguration config = MultiLayerConfiguration.fromJson(jsonConfig);
        return config;
    }


    /**
     * Loads the a network configuration.
     * Uses the previously assigned <i>modelPath</i> and <i>modelName</i> attributes to store the data.
     *
     * @return previously stored net configuration
     */
    public String loadNetworkConfigJson()  {
        String fileName = configName;

        try {
            File storedJsonConfiguration = new File(modelPath, fileName);
            List<String> readLines = Files.readLines(storedJsonConfiguration, Charset.forName("UTF-8"));

            StringBuilder jsonConfigurationBuilder = new StringBuilder();
            for (String line : readLines) {
                jsonConfigurationBuilder.append(line).append("\n");
            }

            return jsonConfigurationBuilder.toString();
        } catch (IOException e) {
            logger.error("Failed storing model [" + modelName + "].", e);
            return null;
        }
    }

    /**
     * Loads the stored normalizer as mean and std files.
     * Uses the previously assigned <i>normalizerPath</i> and <i>modelName</i> attributes to store the data.
     *
     * @return normalizer the stored normalizer
     */
    public NormalizerStandardize loadNormalizer() throws IOException {
        NormalizerStandardize normalizer = null;

        File featuresMeanFile = new File(normalizerPath, "normalizer-stand-feature-mean-" + modelName);
        File featuresStdFile = new File(normalizerPath, "normalizer-stand-feature-std-" + modelName);

        if (featuresMeanFile.exists() && featuresStdFile.exists()) {
            normalizer = new NormalizerStandardize();

            File labelsMeanFile = new File(normalizerPath, "normalizer-stand-labels-mean-" + modelName);
            File labelsStdFile = new File(normalizerPath, "normalizer-stand-labels-std-" + modelName);

            if (labelsMeanFile.exists() && labelsStdFile.exists()) {
                normalizer.fitLabel(true);
                normalizer.load(featuresMeanFile, featuresStdFile, labelsMeanFile, labelsStdFile);
            } else {
                normalizer.load(featuresMeanFile, featuresStdFile);
            }
        }
        return normalizer;
    }
}