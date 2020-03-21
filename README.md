# OCT---MH

That repository is to respond how to get some images features and it has been created to prove that script works correctly, or not. 

The main script generated all features and their values and reported as an excel file .

Firstly, it checks that whether the filenames in the given excel file matches the existing image names, or not.

Then, it controls the correctness with hashing. For example, if two different images can be named the same
but with hashing method, it enables it is the same image, or not. Therefore, it can find dublicate images.

After that, it gets blurriness score, blurriness score by adding gaussiand filter, average pixel width, average intensity,
darkness and whiteness, dominant intensity, noise level, contrast level, and motion estimation between every plane.

It does a histogram, related to feature's values and frequency.

Lastly, in the comparison file, it pictures the differences of every images with regard to feature's minimum, middle and maximum value and then pictures again its every plane.

All algorithms implemented in Python.
