import java.io.File;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Collections;
import java.io.FileNotFoundException;
//import java.util.stream.IntStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.Map;

/**
 * This is the main class for the CS 214 Project.
 * It handles the reading of image files and clustering them based on their
 * histograms.
 */
public class CS_214_Project {
    /**
     * The entry point of the application.
     *
     * @param args an array of command-line arguments, where args[0] is the input
     *             file
     *             and args[1] is the target number of clusters.
     */
    public static void main(String[] args) {
        System.exit(intMain(args));
    }

    /**
     * the int main that returns to main
     *
     * @return returnval for system exit
     */
    public static int intMain(String[] args) {
        if (args.length != 3) {
            System.err.println("Error: Please provide training set, test set, and number of clusters");
            return -1;
        }

        int k = Integer.parseInt(args[2]);
        makeClusteringMethod(args[0], args[1], k);
        return 0;
    }
    
    public static void makeClusteringMethod(String trainingSetFile, String testSetFile, int k ){
        imageReader fileReader1 = new imageReader(trainingSetFile);
        List<PGM> images1 = fileReader1.readImageFiles();
        imageReader fileReader2 = new imageReader(testSetFile);
        List<PGM> images2 = fileReader2.readImageFiles();

        Set<Integer> classnums = new HashSet<>();

        for (PGM image: images1){
            classnums.add(image.getGroundTruth());
        }
        if (!(classnums.size() >= 2)){
            System.out.println(classnums.size());
           // System.err.println("Error: training data has insufficient classes");
           // throw new IllegalArgumentException();
        }

        for (int i = 0; i < classnums.size(); i++){
            Perceptronize(trainingSetFile, i);
        }


        int numPerceptrons = classnums.size();
        Perceptron[] perceptrons = new Perceptron[numPerceptrons];

        double[][] perceptronScores = new double[numPerceptrons][images2.size()];
        for (int n = 0; n < numPerceptrons - 1; n++) {
            Perceptron perceptron = perceptrons[n];
            for (int i = 0; i < images2.size(); i++) {
                perceptronScores[n][i] = perceptron.classify(images2.get(i).getHistogram());
            }
        }

        double[][] similarityMatrix = new double[images2.size()][images2.size()];
for (int i = 0; i < images2.size(); i++) {
    for (int j = 0; j < images2.size(); j++) {
        if (i != j) {
            similarityMatrix[i][j] = computeSimilarity(perceptronScores, i, j, numPerceptrons);
        } else {
            similarityMatrix[i][j] = Double.POSITIVE_INFINITY; // Self-similarity
        }
    }
}
        Clusterize(testSetFile, k, 5, similarityMatrix);


        
    }

    public static double computeSimilarity(double[][] perceptronScores, int imageIndex1, int imageIndex2, int numPerceptrons) {
        double similarity = 0.0;
    
        for (int n = 0; n < numPerceptrons; n++) {
            double diff = perceptronScores[n][imageIndex1] - perceptronScores[n][imageIndex2];
            similarity += 1.0 / (diff * diff);
        }
    
        return similarity;
    }

    public static Perceptron Perceptronize(String trainingSetFile, int classNum){
        imageReader fileReader = new imageReader(trainingSetFile);
        List<PGM> images = fileReader.readImageFiles();

        Perceptron perceptron = new Perceptron();
        boolean[] isPositive = new boolean[images.size()];
        int i = 0;

        for (PGM image : images){
            isPositive[i] = (image.getGroundTruth() == classNum);
            i++;
        }
        perceptron.trainPerceptron(images, isPositive, 100);

        return perceptron;
    }
    public static void Clusterize(String testSetFile, int k, int simType, double[][] similarityMatrix) {
        if (simType < 0 && simType > 5) {
            System.err.println(("Error: Choose an actual cluster method"));
            throw new IllegalArgumentException();
        }
        imageReader fileReader = new imageReader(testSetFile);
        List<PGM> images = fileReader.readImageFiles();
        if (k > images.size()) {
            System.err.println(("Error: Cluster amount can't be more than number of images"));
            throw new IllegalArgumentException();
        }
        if (images.size() < 2) {
            System.err.println("Error: File must contain more than 1 file to compare");
            throw new IllegalArgumentException();
        }
        ClusterFuncs theclusters = new ClusterFuncs(images);
        if (simType == 1) {
            theclusters.compareAllClusters(k);
        }
        if (simType == 2) {
            theclusters.compareAllClusters4(k);
        }
        if (simType == 3) {
            theclusters.compareAllClustersInverse(k);
        }
        if (simType == 4) {
            theclusters.compareAllClusters9(k);
        }
        if (simType == 5){
            theclusters.performClustering(images, similarityMatrix, k);
        }
        //theclusters.printCluster();
        System.out.format("%.6f\n",theclusters.findOverallQuality());
    }

}

class PGM {
    private String fileName;
    private ArrayList<Integer> pixels;
    private Histogram histogram;
    private Norm4Histograms norm4Histograms;
    private Norm9Histograms norm9Histograms;
    private int groundTruth;

    public PGM(String fileName) {
        this.fileName = fileName;
        this.pixels = new ArrayList<>();
        try {
            readImageFile();
            this.histogram = new Histogram(pixels);
            this.norm4Histograms = new Norm4Histograms(pixels);
            this.norm9Histograms = new Norm9Histograms(pixels);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            throw new IllegalArgumentException("Error: File not found: " + fileName);
        }

    }

    private void readImageFile() {
        try (Scanner scanner = new Scanner(new File(fileName))) {
            if (!scanner.hasNext()) {
                throw new IllegalArgumentException("Error: Can't have Empty File");
            }
            String Pval = scanner.next();
            if (!Pval.equals("P2")) {
                throw new IllegalArgumentException("Error: Invalid Format");
            }
            int index = fileName.lastIndexOf("/");
            String realFileName = fileName.substring(index);
            groundTruth = Character.getNumericValue(realFileName.charAt(5));
            //System.out.println(fileName);
            int width = scanner.nextInt();
            int height = scanner.nextInt();
            int maxVal = scanner.nextInt();

            if (width < 0 || height < 0) {
                throw new IllegalArgumentException("Error: Invalid Dimensions");
            }

            while (scanner.hasNext()) {
                String token = scanner.next();
                try {
                    int num = Integer.parseInt(token);
                    if (num < 0 || num > 255) {
                        throw new IllegalArgumentException(
                                "Error: Number needs to be between 0 & 255, number was " + num);
                    }
                    pixels.add(num);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Error: Expected Integer type");
                }

            }
        } catch (FileNotFoundException e) {
            throw new IllegalArgumentException("Error: File not found1: " + fileName);
        }
        if (pixels.size() != 16384){
            throw new IllegalArgumentException("Error: Corrupted Image");
        }
    }

    public String getFileName() {
        return fileName;
    }

    public Histogram getHistogram() {
        return histogram;
    }

    public ArrayList<Integer> getPixels() {
        return pixels;
    }

    public Norm4Histograms getNorm4Histograms() {
        return this.norm4Histograms;
    }
    public Norm9Histograms getNorm9Histograms() {
        return this.norm9Histograms;
    }
    public int getGroundTruth() {
        return groundTruth;
    }
}

class Cluster {
    private ArrayList<PGM> images;
    private double[] clusterhisto;
    private double[][] norm4Histos;
    private double[][] norm9Histos;

    public Cluster(PGM initializer) {
        this.images = new ArrayList<>();
        this.images.add(initializer);
        this.clusterhisto = initializer.getHistogram().getNormalizedBuckets().clone();

        this.norm4Histos = new double[4][];
        this.norm4Histos[0] = initializer.getNorm4Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm4Histos[1] = initializer.getNorm4Histograms().getTopRight().getNormalizedBuckets().clone();
        this.norm4Histos[2] = initializer.getNorm4Histograms().getBottomLeft().getNormalizedBuckets().clone();
        this.norm4Histos[3] = initializer.getNorm4Histograms().getBottomRight().getNormalizedBuckets().clone();

        this.norm9Histos = new double[9][];
        this.norm9Histos[0] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm9Histos[1] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm9Histos[2] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm9Histos[3] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm9Histos[4] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm9Histos[5] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm9Histos[6] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm9Histos[7] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
        this.norm9Histos[8] = initializer.getNorm9Histograms().getTopLeft().getNormalizedBuckets().clone();
    
    }

    public void combine(Cluster addon) {
        for (int i = 0; i < this.clusterhisto.length; i++) {
            this.clusterhisto[i] = (this.clusterhisto[i] + addon.clusterhisto[i]) / 2.0;
        }
        for (int q = 0; q < 4; q++) {
            for (int i = 0; i < this.norm4Histos[q].length; i++) {
                this.norm4Histos[q][i] = (this.norm4Histos[q][i] + addon.norm4Histos[q][i]) / 2.0;
            }
        }
        for (int q = 0; q < 9; q++) {
            for (int i = 0; i < this.norm9Histos[q].length; i++) {
                this.norm9Histos[q][i] = (this.norm9Histos[q][i] + addon.norm9Histos[q][i]) / 2.0;
            }
        }
        this.images.addAll(addon.images);
    }

    public void printImageNames() {
        ArrayList<String> names = new ArrayList<>();
        for (PGM image : images) {
            String fullPath = image.getFileName();
            String fileName = new File(fullPath).getName();
            names.add(fileName);
        }
        Collections.sort(names);
        for (String name : names) {
            System.out.print(name);
            System.out.print(" ");
        }
    }
    public int findDominantCategory() {
        Map<Integer, Integer> frequencyMap = new HashMap<>();
        int dominantCategory = 0;
        int max = 0;
        for (PGM image : this.images) {
            int gt = image.getGroundTruth();
            frequencyMap.put(gt, frequencyMap.getOrDefault(gt, 0) + 1);
        }

        for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
            if (entry.getValue() > max) {
                dominantCategory = entry.getKey();
                max = entry.getValue();
            }
        }
        return dominantCategory;
    }
    public int getMatchingCount() {  // Changed method name and return type
        int numMatches = 0;
        int dominantCat = this.findDominantCategory();
        for (PGM image: this.images) {
            if (image.getGroundTruth() == dominantCat) {  // Count matching images
                numMatches++;
            }
        }
        return numMatches;  // Return int instead of double
    }

    public double[] getHistogram() {
        return this.clusterhisto;
    }

    public double[][] getNorm4Histos() {
        return this.norm4Histos;
    }

    public double[][] getNorm9Histos() {
        return this.norm9Histos;
    }

    public ArrayList<PGM> getImages() {
        return this.images;
    }
}

class ClusterFuncs {
    private ArrayList<Cluster> clusters;
    public int numClusters;
    

    public ClusterFuncs(List<PGM> images) {
        this.clusters = new ArrayList<>();
        for (PGM image : images) {
            this.clusters.add(new Cluster(image));
            numClusters++;
        }
    }

    public static int squareDiff(ArrayList<Integer> firstpixelSet, ArrayList<Integer> secondpixelSet) {
        int globalsum = 0;
        if (firstpixelSet.size() != secondpixelSet.size()) {
            throw new IllegalArgumentException("Error: can't sum square diff with different dimensions");
        }
        for (int i = 0; i < firstpixelSet.size(); i++) {
            int result = secondpixelSet.get(i) - firstpixelSet.get(i);
            result = result * result;
            globalsum += result;

        }
        return globalsum;
    }

    public double pairwiseMinimum(double[] histo1, double[] histo2) {
        double globalmin = 0;
        for (int i = 0; i < histo1.length; i++) {
            globalmin += Math.min(histo1[i], histo2[i]);
        }
        return globalmin;
    }
    public double findOverallQuality() {
        double totalMatches = 0;
        double totalImages = 0;
        
        for (Cluster cluster : this.clusters) {
            totalMatches += cluster.getMatchingCount();
            totalImages += cluster.getImages().size();
        }
        
        return totalMatches / totalImages;
    }

    private double computeClusterSimilarity(Cluster cluster1, Cluster cluster2, double[][] similarityMatrix, List<PGM> testImages) {
        double totalSimilarity = 0.0;
        int comparisons = 0;
    
        // Compare every image in cluster1 with every image in cluster2
        for (PGM image1 : cluster1.getImages()) {
            for (PGM image2 : cluster2.getImages()) {
                int index1 = testImages.indexOf(image1);
                int index2 = testImages.indexOf(image2);
                totalSimilarity += similarityMatrix[index1][index2];
                comparisons++;
            }
        }
    
        // Return the average similarity
        return totalSimilarity / comparisons;
    }
    
    public void compareAllClusters(int K, double[][] similarityMatrix, List<PGM> testImages) {
    // Step 1: Initialize each test image as its own cluster
    List<Cluster> clusters = new ArrayList<>();
    for (PGM image : testImages) {
        clusters.add(new Cluster(image));
    }

    // Step 2: Perform agglomerative clustering
    while (clusters.size() > K) {
        double maxSimilarity = Double.NEGATIVE_INFINITY;
        Cluster mergeCluster1 = null;
        Cluster mergeCluster2 = null;

        // Find the two most similar clusters
        for (int i = 0; i < clusters.size(); i++) {
            for (int j = i + 1; j < clusters.size(); j++) {
                double similarity = computeClusterSimilarity(clusters.get(i), clusters.get(j), similarityMatrix, testImages);
                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    mergeCluster1 = clusters.get(i);
                    mergeCluster2 = clusters.get(j);
                }
            }
        }

        // Merge the most similar clusters
        if (mergeCluster1 != null && mergeCluster2 != null) {
            mergeCluster1.combine(mergeCluster2);
            clusters.remove(mergeCluster2);
        }
    }

    // Step 3: Output the clusters
    for (Cluster cluster : clusters) {
        cluster.printImageNames();
    }
}


    public void performClustering(List<PGM> testImages, double[][] similarityMatrix, int K) {
        // Initialize each image as its own cluster
        List<Cluster> clusters = new ArrayList<>();
        for (PGM image : testImages) {
            clusters.add(new Cluster(image));
        }
    
        // Agglomerative clustering logic
        while (clusters.size() > K) {
            double maxSimilarity = Double.NEGATIVE_INFINITY;
            Cluster mergeCluster1 = null;
            Cluster mergeCluster2 = null;
    
            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    double similarity = computeClusterSimilarity(clusters.get(i), clusters.get(j), similarityMatrix, testImages);
                    if (similarity > maxSimilarity) {
                        maxSimilarity = similarity;
                        mergeCluster1 = clusters.get(i);
                        mergeCluster2 = clusters.get(j);
                    }
                }
            }
    
            // Merge the two most similar clusters
            if (mergeCluster1 != null && mergeCluster2 != null) {
                mergeCluster1.combine(mergeCluster2);
                clusters.remove(mergeCluster2);
            }
        }
    
        // Print the final clusters
        for (Cluster cluster : clusters) {
            cluster.printImageNames();
        }
    }
    
    public void compareAllClusters(int k) {
        while (numClusters > k) {
            double minDifference = 1;
            Cluster cluster1ToMerge = null;
            Cluster cluster2ToMerge = null;

            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    Cluster c1 = clusters.get(i);
                    Cluster c2 = clusters.get(j);
                    double difference = 1 - pairwiseMinimum(c1.getHistogram(), c2.getHistogram());
                    if (difference < minDifference) {
                        minDifference = difference;
                        cluster1ToMerge = c1;
                        cluster2ToMerge = c2;
                    }
                }
            }

            if (cluster1ToMerge != null && cluster2ToMerge != null) {
                cluster1ToMerge.combine(cluster2ToMerge);
                clusters.remove(cluster2ToMerge);
                numClusters--;
            }
        }
  
    }

    public void compareAllClusters4(int k) {
        while (numClusters > k) {
            double minDifference = 1;
            Cluster cluster1ToMerge = null;
            Cluster cluster2ToMerge = null;

            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    Cluster c1 = clusters.get(i);
                    Cluster c2 = clusters.get(j);

                    // Calculate average similarity across all quadrants
                    double totalSimilarity = 0;
                    for (int q = 0; q < 4; q++) {
                        totalSimilarity += pairwiseMinimum(c1.getNorm4Histos()[q],
                                c2.getNorm4Histos()[q]);
                    }
                    double averageSimilarity = totalSimilarity / 4.0;
                    double difference = 1 - averageSimilarity;

                    if (difference < minDifference) {
                        minDifference = difference;
                        cluster1ToMerge = c1;
                        cluster2ToMerge = c2;
                    }
                }
            }

            if (cluster1ToMerge != null && cluster2ToMerge != null) {
                cluster1ToMerge.combine(cluster2ToMerge);
                clusters.remove(cluster2ToMerge);
                numClusters--;
            }
        }

    }
    public void compareAllClusters9(int k) {
        while (numClusters > k) {
            double minDifference = 1;
            Cluster cluster1ToMerge = null;
            Cluster cluster2ToMerge = null;

            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    Cluster c1 = clusters.get(i);
                    Cluster c2 = clusters.get(j);

                    // Calculate average similarity across all 9 sections
                    double totalSimilarity = 0;
                    for (int q = 0; q < 9; q++) {
                        totalSimilarity += pairwiseMinimum(c1.getNorm9Histos()[q],
                                c2.getNorm9Histos()[q]);
                    }
                    double averageSimilarity = totalSimilarity / 9.0;
                    double difference = 1 - averageSimilarity;

                    if (difference < minDifference) {
                        minDifference = difference;
                        cluster1ToMerge = c1;
                        cluster2ToMerge = c2;
                    }
                }
            }

            if (cluster1ToMerge != null && cluster2ToMerge != null) {
                cluster1ToMerge.combine(cluster2ToMerge);
                clusters.remove(cluster2ToMerge);
                numClusters--;
            }
        }

   
    }
    public void compareAllClustersInverse(int k) {
        while (numClusters > k) {
            double minAverageSquareDiff = Double.MAX_VALUE;
            Cluster cluster1ToMerge = null;
            Cluster cluster2ToMerge = null;

            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    Cluster c1 = clusters.get(i);
                    Cluster c2 = clusters.get(j);

                    // Calculate average square difference between all image pairs in the clusters
                    double totalSquareDiff = 0;
                    int comparisons = 0;

                    for (PGM img1 : c1.getImages()) {
                        for (PGM img2 : c2.getImages()) {
                            totalSquareDiff += squareDiff(img1.getPixels(), img2.getPixels());
                            comparisons++;
                        }
                    }

                    double averageSquareDiff = totalSquareDiff / comparisons;

                    if (averageSquareDiff < minAverageSquareDiff) {
                        minAverageSquareDiff = averageSquareDiff;
                        cluster1ToMerge = c1;
                        cluster2ToMerge = c2;
                    }
                }
            }

            if (cluster1ToMerge != null && cluster2ToMerge != null) {
                cluster1ToMerge.combine(cluster2ToMerge);
                clusters.remove(cluster2ToMerge);
                numClusters--;
            }
        }

    }
    public void printCluster(){
        for (Cluster cluster : this.clusters) {
            cluster.printImageNames();
            System.out.println();
        }
    }
}

class Norm4Histograms {
    private Histogram topLeft;
    private Histogram topRight;
    private Histogram bottomLeft;
    private Histogram bottomRight;

    public Norm4Histograms(ArrayList<Integer> pixels) {
        ArrayList<Integer> topLeftPixels = new ArrayList<>();
        ArrayList<Integer> topRightPixels = new ArrayList<>();
        ArrayList<Integer> bottomLeftPixels = new ArrayList<>();
        ArrayList<Integer> bottomRightPixels = new ArrayList<>();

        int width = 128;
        int height = 128;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = pixels.get(y * width + x);

                if (x < width / 2) {
                    if (y < height / 2) {
                        topLeftPixels.add(pixel);
                    } else {
                        bottomLeftPixels.add(pixel);
                    }
                } else {
                    if (y < height / 2) {
                        topRightPixels.add(pixel);
                    } else {
                        bottomRightPixels.add(pixel);
                    }
                }
            }
        }

        topLeft = new Histogram(topLeftPixels);
        topRight = new Histogram(topRightPixels);
        bottomLeft = new Histogram(bottomLeftPixels);
        bottomRight = new Histogram(bottomRightPixels);
    }

    public Histogram getTopLeft() {
        return topLeft;
    }

    public Histogram getTopRight() {
        return topRight;
    }

    public Histogram getBottomLeft() {
        return bottomLeft;
    }

    public Histogram getBottomRight() {
        return bottomRight;
    }
}

class Norm9Histograms {
    private Histogram topLeft;
    private Histogram topCenter;
    private Histogram topRight;
    private Histogram middleLeft;
    private Histogram middleCenter;
    private Histogram middleRight;
    private Histogram bottomLeft;
    private Histogram bottomCenter;
    private Histogram bottomRight;

    public Norm9Histograms(ArrayList<Integer> pixels) {
        ArrayList<Integer> topLeftPixels = new ArrayList<>();
        ArrayList<Integer> topCenterPixels = new ArrayList<>();
        ArrayList<Integer> topRightPixels = new ArrayList<>();
        ArrayList<Integer> middleLeftPixels = new ArrayList<>();
        ArrayList<Integer> middleCenterPixels = new ArrayList<>();
        ArrayList<Integer> middleRightPixels = new ArrayList<>();
        ArrayList<Integer> bottomLeftPixels = new ArrayList<>();
        ArrayList<Integer> bottomCenterPixels = new ArrayList<>();
        ArrayList<Integer> bottomRightPixels = new ArrayList<>();

        int width = 128;
        int height = 128;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = pixels.get(y * width + x); // convert to 1d Array

                int xRegion;
                if (x <= 42) xRegion = 0;
                else if (x <= 85) xRegion = 1;
                else xRegion = 2;

                int yRegion;
                if (y <= 42) yRegion = 0;
                else if (y <= 85) yRegion = 1;
                else yRegion = 2;

                // Add pixel to appropriate section based on x and y regions
                if (yRegion == 0) {  
                    if (xRegion == 0) topLeftPixels.add(pixel);
                    else if (xRegion == 1) topCenterPixels.add(pixel);
                    else topRightPixels.add(pixel);
                }
                else if (yRegion == 1) {  // Middle row
                    if (xRegion == 0) middleLeftPixels.add(pixel);
                    else if (xRegion == 1) middleCenterPixels.add(pixel);
                    else middleRightPixels.add(pixel);
                }
                else {  // Bottom row
                    if (xRegion == 0) bottomLeftPixels.add(pixel);
                    else if (xRegion == 1) bottomCenterPixels.add(pixel);
                    else bottomRightPixels.add(pixel);
                }


            }
        }

        topLeft = new Histogram(topLeftPixels);
        topCenter = new Histogram(topCenterPixels);
        topRight = new Histogram(topRightPixels);
        middleLeft = new Histogram(middleLeftPixels);
        middleCenter = new Histogram(middleCenterPixels);
        middleRight = new Histogram(middleRightPixels);
        bottomLeft = new Histogram(bottomLeftPixels);
        bottomCenter = new Histogram(bottomCenterPixels);
        bottomRight = new Histogram(bottomRightPixels);
    }

    public Histogram getTopLeft() {
        return topLeft;
    }

    public Histogram getTopCenter() {
        return topCenter;
    }

    public Histogram getTopRight() {
        return topRight;
    }

    public Histogram getMiddleLeft() {
        return middleLeft;
    }

    public Histogram getMiddleCenter() {
        return middleCenter;
    }

    public Histogram getMiddleRight() {
        return middleRight;
    }

    public Histogram getBottomLeft() {
        return bottomLeft;
    }

    public Histogram getBottomCenter() {
        return bottomCenter;
    }

    public Histogram getBottomRight() {
        return bottomRight;
    }
}


class Histogram {
    private int[] buckets;
    private double[] normalizedBuckets;

    public Histogram(ArrayList<Integer> pixels) {
        buckets = new int[64];
        createHistogram(pixels);
        normalize();
    }

    private void createHistogram(ArrayList<Integer> pixels) {
        for (int pixel : pixels) {
            int bucket = (int) Math.floor(pixel / 4.0);
            buckets[bucket]++;
        }
    }

    private void normalize() {
        int sum = 0;
        for (int i = 0; i < buckets.length; i++) {
            sum += buckets[i];
        }

        // Create the normalized bucket array
        normalizedBuckets = new double[buckets.length];
        for (int i = 0; i < buckets.length; i++) {
            normalizedBuckets[i] = buckets[i] / (double) sum;
        }
        //System.out.println("Normalized Histogram: " + Arrays.toString(normalizedBuckets));
    }

    public double[] getNormalizedBuckets() {
        return normalizedBuckets;
    }

}

class imageComparator {
    private List<PGM> images;

    public imageComparator(List<PGM> images) {
        this.images = images;
    }

    public void compareAllImages() {
        for (PGM image : images) {
            PGM mostDifferent = findMostDifferentImage(image);
            double difference = calculateDifference(image, mostDifferent);
            System.out.printf("%s %s %.6f%n", image.getFileName(), mostDifferent.getFileName(), difference);
        }
    }

    public PGM findMostDifferentImage(PGM baseImage) {
        PGM mostDifferent = null;
        double maxDifference = -1;

        for (PGM otherImage : images) {
            if (otherImage != baseImage) {
                double difference = calculateDifference(baseImage, otherImage);
                if (difference > maxDifference) {
                    maxDifference = difference;
                    mostDifferent = otherImage;
                }
            }
        }

        return mostDifferent;
    }

    public double calculateDifference(PGM image1, PGM image2) {
        return pairwiseMinimum(image1.getHistogram().getNormalizedBuckets(),
                image2.getHistogram().getNormalizedBuckets());
    }

    public double pairwiseMinimum(double[] histo1, double[] histo2) {
        double globalmin = 0;
        for (int i = 0; i < histo1.length; i++) {
            globalmin += Math.min(histo1[i], histo2[i]);
        }
        return globalmin;
    }
}

class imageReader {
    private String inputFileName;

    public imageReader(String inputFileName) {
        this.inputFileName = inputFileName;
    }

    public List<PGM> readImageFiles() {
        List<PGM> images = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File(inputFileName))) {
            while (scanner.hasNext()) {
                String fileName = scanner.next();
                images.add(new PGM(fileName));
            }
        } catch (FileNotFoundException e) {
            System.err.println("Error: File not found2: " + inputFileName);
        }
        return images;
    }
}



class Perceptron {
    private double[] weights;
    private double bias;

    public Perceptron(){
        this.weights = new double[64]; //initialize weights
        this.bias = 0.0;
    }

    public double classify(Histogram histo){
        double[] normalizedBuckets = histo.getNormalizedBuckets();

        double sum = bias;

        for (int i = 0; i < 64; i++){
            sum += weights[i] * normalizedBuckets[i];
        }
        // sum indicates whether perceptron thinks it's in class or not
        return sum;

    }

    public void updateWeights(Histogram histo, boolean isClass){
        double[] normalizedBuckets = histo.getNormalizedBuckets();

        double d = 0;
        if (isClass){
            d = 1.0;
        }
        else {
            d = -1.0;
        }
        double y = classify(histo);
        double updateVal = d - y; //calculate update factor

        for (int i = 0; i < 64; i++){ //update weights
            weights[i] += updateVal * normalizedBuckets[i];
        }
        //bias = b + (d-y)
        bias += updateVal;
    }

    public void trainPerceptron(List<PGM> images, boolean[] isClass, int epochs){//epochs = num iterations
        for (int epoch = 0; epoch < epochs; epoch++){
            for (int i = 0; i < images.size(); i++){
                updateWeights(images.get(i).getHistogram(), isClass[i]);
            }

        }

    }
    public double getBias(){
        return this.bias;
    }
    public double[] getWeights(){
        return this.weights;
    }

    public boolean determineInClass(PGM image) {
        return classify(image.getHistogram()) > 0;
    }
}
