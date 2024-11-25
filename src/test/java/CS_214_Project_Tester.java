import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.io.IOException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterAll;

public class CS_214_Project_Tester {

    @Test
    public void ZeroInputFile() {
        String[] args = {};

        int result = CS_214_Project.intMain(args);

        assertEquals(-1, result);
    }

    @Test
    public void testHistogram() {

        PGM image = new PGM("input_files/train/class1_8.pgm");
        Histogram histo = image.getHistogram();
        assertNotNull(histo);

    }

    @Test
    public void testPGMCreation() {
        PGM image = new PGM("input_files/train/class1_8.pgm");
        assertNotNull(image);
    }

    @Test
    public void testsHistogram() {
        ArrayList<Integer> pixels = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            pixels.add(i);
        }
        Histogram histogram = new Histogram(pixels);
        assertNotNull(histogram);
    }

    @Test
    public void testNormalize() {
        ArrayList<Integer> pixels = new ArrayList<>();
        for (int i = 0; i < 256; i++) {
            pixels.add(i);
        }
        Histogram histogram = new Histogram(pixels);
        double[] normalizedBuckets = histogram.getNormalizedBuckets();
        assertEquals(64, normalizedBuckets.length);
    }

    @Test
    public void testPairwiseMinimum() {
        double[] histo1 = { 0.1, 0.2, 0.3, 0.4 };
        double[] histo2 = { 0.2, 0.1, 0.4, 0.3 };
        imageComparator test = new imageComparator(new ArrayList<>());
        double result = test.pairwiseMinimum(histo1, histo2);
        assertNotNull(result);
    }

    @Test
    public void testimageReader() {
        imageReader reader = new imageReader("input_files/example2.pgm");
        assertNotNull(reader);
    }

    @Test
    public void testReadImageFiles() {
        imageReader reader = new imageReader("input_files/test.txt");
        List<PGM> images = reader.readImageFiles();
        assertNotNull(images);
        assertTrue(images.size() > 0);
    }

    @Test
    public void testInvalidPGMFormat() {
        try {
            PGM image = new PGM("input_files/emptypgm.pgm");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().startsWith("Error: "));
        }
    }

    @Test
    public void testFNF() {
        try {
            PGM image = new PGM("nonexistent.pgm");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().startsWith("Error: File not found"));
        }
    }

    @Test
    public void testInvalidPGMDimensions() {
        try {
            PGM image = new PGM("input_files/wrongdimensions.pgm");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().startsWith("Error: "));
        }
    }

    @Test
    public void testInvalidPGMsFormat() {
        try {
            PGM image = new PGM("input_files/invalidpgm.pgm");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().startsWith("Error: "));
        }
    }

    @Test
    public void testInvalidPixelSize() {
        try {
            PGM image = new PGM("input_files/invalidpixels.pgm");
            fail("Expected IllegalArgumentException to be thrown due to invalid pixel size");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().startsWith("Error: "));
        }
    }

    @Test
    public void testFindMostDifferentImage() {
        PGM image1 = new PGM("input_files/train/class1_8.pgm");
        PGM image2 = new PGM("input_files/train/class1_9.pgm");
        PGM image3 = new PGM("input_files/train/class1_11.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);

        imageComparator comparer = new imageComparator(images);
        PGM mostDifferent = comparer.findMostDifferentImage(image1);

        assertNotNull(mostDifferent);
        assertNotEquals(image1, mostDifferent);
    }

    @Test
    public void testCalcDiff() {
        PGM image = new PGM("input_files/train/class1_8.pgm");
        PGM otherimage = new PGM("input_files/train/class1_9.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image);
        images.add(otherimage);

        imageComparator comparer = new imageComparator(images);

        double diff = comparer.calculateDifference(image, otherimage);

        assertTrue(diff >= 0 && diff <= 1);
    }

    @Test
    public void testPairwiseMinimum_histo1Smaller() {
        PGM image = new PGM("input_files/train/class1_8.pgm");
        PGM otherimage = new PGM("input_files/train/class1_9.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image);
        images.add(otherimage);
        ClusterFuncs sup = new ClusterFuncs(images);
        double[] histo1 = { 0.1, 0.2, 0.3 };
        double[] histo2 = { 0.5, 0.6, 0.7 }; // All values larger than histo1
        double result = sup.pairwiseMinimum(histo1, histo2);
        assertEquals(0.1 + 0.2 + 0.3, result, 0.0001); // histo1 should be chosen for all
    }

    @Test
    public void testPairwiseMinimum_histo2Smaller() {
        PGM image = new PGM("input_files/train/class1_8.pgm");
        PGM otherimage = new PGM("input_files/train/class1_9.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image);
        images.add(otherimage);
        ClusterFuncs sup = new ClusterFuncs(images);

        double[] histo1 = { 0.6, 0.7, 0.8 };
        double[] histo2 = { 0.5, 0.6, 0.7 }; // All values smaller than histo1
        double result = sup.pairwiseMinimum(histo1, histo2);
        assertEquals(0.5 + 0.6 + 0.7, result, 0.0001); // histo2 should be chosen for all
    }

    @Test
    public void testPairwiseMinimum_equal() {
        PGM image = new PGM("input_files/train/class1_8.pgm");
        PGM otherimage = new PGM("input_files/train/class1_9.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image);
        images.add(otherimage);
        ClusterFuncs sup = new ClusterFuncs(images);
        double[] histo1 = { 0.5, 0.5, 0.5 };
        double[] histo2 = { 0.5, 0.5, 0.5 };
        double result = sup.pairwiseMinimum(histo1, histo2);
        assertEquals(1.5, result, 0.0001); // Both are the same, result should be the sum of either
    }

    @Test
    public void testCompareAllClusters_noMerges() {
        PGM image1 = new PGM("input_files/train/class1_8.pgm");
        PGM image2 = new PGM("input_files/train/class1_9.pgm");
        PGM image3 = new PGM("input_files/train/class1_11.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);
        ClusterFuncs funcs = new ClusterFuncs(images);
        funcs.numClusters = 3;
        funcs.compareAllClusters(3);

        assertEquals(3, funcs.numClusters);
    }

    @Test
    public void testCompareAllClusters_clustersGetMerged() {
        PGM image1 = new PGM("input_files/train/class1_8.pgm");
        PGM image2 = new PGM("input_files/train/class1_9.pgm");
        PGM image3 = new PGM("input_files/train/class1_11.pgm");
        PGM image4 = new PGM("input_files/train/class1_12.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);
        images.add(image4);
        ClusterFuncs funcs = new ClusterFuncs(images);
        funcs.numClusters = 4;

        funcs.compareAllClusters(2);

        assertEquals(2, funcs.numClusters);
    }

    @Test
    public void testCompareAllClustersInverse_clustersGetMerged() {
        PGM image1 = new PGM("input_files/train/class1_8.pgm");
        PGM image2 = new PGM("input_files/train/class1_9.pgm");
        PGM image3 = new PGM("input_files/train/class1_11.pgm");
        PGM image4 = new PGM("input_files/train/class1_12.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);
        images.add(image4);
        ClusterFuncs funcs = new ClusterFuncs(images);
        funcs.numClusters = 4;

        funcs.compareAllClustersInverse(2);

        assertEquals(2, funcs.numClusters);
    }

    @Test
    public void testCompareAllClusters4_clustersGetMerged() {
        PGM image1 = new PGM("input_files/train/class1_8.pgm");
        PGM image2 = new PGM("input_files/train/class1_9.pgm");
        PGM image3 = new PGM("input_files/train/class1_11.pgm");
        PGM image4 = new PGM("input_files/train/class1_12.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);
        images.add(image4);
        ClusterFuncs funcs = new ClusterFuncs(images);
        funcs.numClusters = 4;

        funcs.compareAllClusters4(2);

        assertEquals(2, funcs.numClusters);
    }

    @Test
    public void testCompareAllClusters9_clustersGetMerged() {
        PGM image1 = new PGM("input_files/train/class1_8.pgm");
        PGM image2 = new PGM("input_files/train/class1_9.pgm");
        PGM image3 = new PGM("input_files/train/class1_11.pgm");
        PGM image4 = new PGM("input_files/train/class1_12.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);
        images.add(image4);
        ClusterFuncs funcs = new ClusterFuncs(images);
        funcs.numClusters = 4;

        funcs.compareAllClusters9(2);

        assertEquals(2, funcs.numClusters);
    }

    @Test
    public void testInvalidClusterArgument() {
        String[] args = { "", "4", "6" };
        int simType = Integer.parseInt(args[2]);
        double[][] shmop = new double[2][6];

        assertThrows(IllegalArgumentException.class, () -> CS_214_Project.Clusterize(args[0], simType, 3, shmop));
    }

    @Test
    public void testIntMainInvalidArgumentFormat() {
    String[] args = {"input_files/test.txt", "input_files/train.txt", "two"};
    assertThrows(NumberFormatException.class, () -> CS_214_Project.intMain(args));
}

@Test
    public void testFindOverallQuality() {
        PGM image1 = new PGM("input_files/train/class1_8.pgm");
        PGM image2 = new PGM("input_files/train/class1_9.pgm");
        PGM image3 = new PGM("input_files/train/class1_11.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);

        ClusterFuncs funcs = new ClusterFuncs(images);
        double quality = funcs.findOverallQuality();

        assertTrue(quality >= 0 && quality <= 1, "Quality should be between 0 and 1");
}

@Test
    public void testPerceptron() {
        PGM image1 = new PGM("input_files/train/class1_8.pgm");
        PGM image2 = new PGM("input_files/train/class1_9.pgm");
        PGM image3 = new PGM("input_files/train/class1_10.pgm");

        ArrayList<PGM> images = new ArrayList<>();
        images.add(image1);
        images.add(image2);
        images.add(image3);

        boolean[] isPositive = new boolean[images.size()];
        int i = 0;

        Perceptron perceptron = new Perceptron();
        for (PGM image : images){
            isPositive[i] = (image.getGroundTruth() == 2);
            i++;
        }
        perceptron.trainPerceptron(images, isPositive, 100);

        assertTrue(perceptron.getBias() != 0);
}

}
