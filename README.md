[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/At2SIOll)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=17078859)
# PA9: Putting It All Together

# Motivation
This is the week we put all the pieces together, meaning that we combine learning (perceptrons) 
with clustering (agglomerative). The result will be your largest program yet. In particular, your 
program will train perceptrons based on a training set of (depth) images, and then define a 
distance measure based on those perceptrons and use that distance measure to cluster a test set of 
(depth) images. 
Remember – we are more interested in correct code than fast code. This assignment will be 
graded on correctness, not how much time it takes….

# Task 
Your PA9 program will take three arguments. The first is the training set, expressed as a file of 
depth image file names. The second is the test set, also expressed as a file of (usually different) 
depth image names.  The third argument is K, the number of clusters for your program to make. 
(K, as in PA7, must be an integer greater than zero.) 
Your program must do two things. First, it uses the training images (i.e. the images listed in the 
first argument file) to train perceptrons, similar to PA8. This time, however, there is no argument 
telling you which class to train a perceptron for. Instead, you will train one perceptron for every 
class label in the training set. In other words, if all the training samples are from class1 or class2, 
you will train 2 perceptrons, one for each class. If the training samples  divide into class1, class 2 
and class 3, then you will train 3 perceptrons, and so on. Note: it is an error if there are not at least 
two classes in the training data. As before, you will train each perceptron for 100 epochs. 
Once it has trained the perceptrons, your program will cluster the images in the test set into K 
clusters using these perceptrons. In particular, assume that the training data has N classes, and 
that therefore your program has trained N perceptrons. Let $y_{n,i}$ be the score returned by perceptron 
n on test image i. Then the similarity between two test images i and j is:

$sim(i,j) = \sum_{n=1}^{N} (\frac{1}{(y_{n,i} - y_{n,j})^2})$

Using this similarity measure, your program should use agglomerative clustering (i.e. the 
clustering algorithm from PA6 and PA7) to cluster the test images into K clusters. 
As in PA7, you will output the clusters one line at a time to System.out, with each line containing 
the test file image names for one cluster.

### Output


```
Example output for gradle run -q --args="'input_files/train/train.txt' 'input_files/test/test.txt'  '2'"
class0_1.pgm class0_7.pgm class0_8.pgm class0_9.pgm 
class0_10.pgm class0_11.pgm class0_12.pgm class0_13.pgm class0_14.pgm class0_15.pgm class0_2.pgm class0_3.pgm class0_4.pgm class0_5.pgm class0_6.pgm

Example output for gradle run -q --args="'input_files/train/train.txt' 'input_files/test/test.txt'  '3'" :
class0_1.pgm class0_7.pgm class0_8.pgm class0_9.pgm 
class0_10.pgm class0_11.pgm class0_12.pgm class0_13.pgm class0_2.pgm class0_3.pgm class0_4.pgm class0_5.pgm class0_6.pgm 
class0_14.pgm class0_15.pgm
```


### Submitting Your Work

Your submission must be in the main branch of your GitHub repository.

### Grading Your Homework

Run your code with the following command:

```
gradle run -q --args="'input_files/trainfile' 'input_files/testfile' 'K'"
```
We will pull your code from your repository. It must contain, at a minimum two
`Java` files. One **must** be named `CS_214_Project.java`. We will run your
program by starting it with that class name. The second file **must** be named
`CS_214_Project_Tester.java`. This will be used to run your `JUnit` tests.
All future assignments **must** contain these two files, although they may
contain different code as needed by the particulars of that assignment.

~~~~
public class CS_214_Project {
  public static void main (String[] args) {
    // your code goes here
    // You may just call the main of some other class you wrote.
  }
}
~~~~

~~~~
import org.junit.Test;

// other imports as needed

public CS_214_Project_Tester {
  @Test
  public void test1_OrWhateverNameMakesSense() {
    // code for your first test
  }

  @Test
  public void test2_OrWhateverNameMakesSense() {
    // code for your second test
  }
}
~~~~

### Hints

- Test the clustering from PA7 first, to make sure it has no errors 
- Test the perceptron clustering from PA8 next, to make sure it has no errors. 
- Only put them together once both halves have been tested

---
