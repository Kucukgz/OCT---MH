# OCT---MH

That repository is to respond how to get some images features and it has been created to prove that script works correctly, or not. 

The main script generated and reported all features and their values.

Firstly, it checks that whether the filenames in the given excel file matches the existing image names, or not.

Then, it controls the correctness with hashing. For example, if two different images can be named the same
but with hashing it enables is it the same image or not. Therefore, it can find dublicate images.

After that, it gets blurriness score, blurriness score by adding gaussiand filter, average pixel width, average intensity
darkness and whiteness, dominant intensity, noise level, contrast level, and motion estimation between every plane.

It recorded as an excel file and do histogram.

Lastly, in the comparison file, it pictures the comparison of every plane with regard to features.

All algorithms implemented in Python.
